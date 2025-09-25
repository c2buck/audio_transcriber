import os
import time
import torch
import threading
import zipfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from utils import get_audio_files, get_file_duration, safe_filename
from backend_manager import BackendManager, UnifiedTranscriber


class AudioTranscriber:
    """Core audio transcription class with support for multiple backends."""
    
    def __init__(self, model_name: str = "base", device: Optional[str] = None, 
                 backend: str = "auto", beam_size: int = 5):
        """
        Initialize the transcriber with a Whisper model.
        
        Args:
            model_name: Whisper model size
            device: Device to use ("cuda", "cpu", "mps", or None for auto-detection)
            backend: Backend to use ("auto", "openai", "faster")
            beam_size: Beam size for faster-whisper (ignored for OpenAI)
        """
        self.model_name = model_name
        self.device = self._get_device() if device is None else device
        self.backend = backend
        self.beam_size = beam_size
        self.backend_manager = BackendManager()
        self.unified_transcriber = None
        self.is_model_loaded = False
        
    def _get_device(self) -> str:
        """Automatically detect the best available device with detailed logging."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 Using CUDA GPU: {device_name} ({memory_gb:.1f} GB)")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 Using Apple Silicon GPU (MPS)")
            return "mps"  # Apple Silicon
        else:
            cpu_cores = os.cpu_count()
            print(f"💾 Using CPU with {cpu_cores} cores")
            print("💡 For faster processing, consider using a GPU-enabled system")
            return "cpu"
    
    def load_model(self, progress_callback: Optional[Callable] = None) -> bool:
        """
        Load the transcription model using the selected backend.
        
        Args:
            progress_callback: Optional callback function for progress updates
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if progress_callback:
                # Log backend selection information
                detection_info = self.backend_manager.get_detection_info()
                if detection_info['recommended_backend'] != "Error":
                    progress_callback(f"🔍 Auto-detected backend: {detection_info['recommended_backend']}")
                
                # Log available backends
                available_backends = self.backend_manager.get_available_backends()
                backend_names = [b.display_name for b in available_backends]
                progress_callback(f"📋 Available backends: {', '.join(backend_names)}")
            
            # Create unified transcriber
            self.unified_transcriber = UnifiedTranscriber(
                backend=self.backend,
                model_name=self.model_name,
                device=self.device,
                beam_size=self.beam_size
            )
            
            # Log which backend was actually selected
            if progress_callback:
                backend_info = self.unified_transcriber.get_backend_info()
                progress_callback(f"🚀 Using: {backend_info['backend_display_name']}")
                progress_callback(f"🎯 Model: {backend_info['model_name']}")
                progress_callback(f"💻 Device: {backend_info.get('device_name', backend_info['device'])}")
                if backend_info.get('beam_size'):
                    progress_callback(f"🔬 Beam size: {backend_info['beam_size']}")
            
            # Load the model
            success = self.unified_transcriber.load_model(progress_callback)
            self.is_model_loaded = success
            
            return success
            
        except Exception as e:
            error_msg = f"Failed to initialize transcriber: {str(e)}"
            
            if progress_callback:
                progress_callback(error_msg)
                
                # Provide troubleshooting suggestions
                if "not available" in str(e).lower():
                    progress_callback("💡 Backend not available. Try installing required dependencies")
                elif "model" in str(e).lower() and "not available" in str(e).lower():
                    progress_callback("💡 Model not supported by selected backend. Try a different model")
                elif "CUDA" in str(e) or "cuda" in str(e):
                    progress_callback("💡 CUDA error detected. Try CPU device or install CUDA drivers")
            
            return False
    
    def _get_detailed_device_info(self) -> str:
        """Get detailed information about the selected device."""
        if self.unified_transcriber:
            backend_info = self.unified_transcriber.get_backend_info()
            device_name = backend_info.get('device_name', 'Unknown')
            backend_name = backend_info.get('backend_display_name', 'Unknown')
            return f"{device_name} (using {backend_name})"
        else:
            # Fallback for when transcriber is not loaded yet
            if self.device == "cuda":
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    return f"CUDA GPU: {gpu_name} ({gpu_memory:.1f} GB)"
                else:
                    return "CUDA (unavailable)"
            elif self.device == "mps":
                return "Apple Silicon GPU (MPS)"
            else:
                return f"CPU ({os.cpu_count()} cores)"
    
    def transcribe_file(self, audio_file: str, 
                       progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Transcribe a single audio file using the selected backend.
        
        Args:
            audio_file: Path to the audio file
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dict containing transcription results and metadata
        """
        if not self.is_model_loaded or not self.unified_transcriber:
            return {
                'file_path': audio_file,
                'success': False,
                'error': 'Model not loaded',
                'transcription': None,
                'duration': 0
            }
        
        # Delegate to unified transcriber which handles all the detailed logging
        return self.unified_transcriber.transcribe_file(audio_file, progress_callback)
    
    def transcribe_batch(self, input_directory: str, output_directory: str,
                        progress_callback: Optional[Callable] = None,
                        file_progress_callback: Optional[Callable] = None,
                        create_zip: bool = True) -> Dict[str, Any]:
        """
        Transcribe all audio files in a directory using the selected backend.
        
        Args:
            input_directory: Directory containing audio files
            output_directory: Directory to save transcriptions
            progress_callback: Callback for overall progress updates
            file_progress_callback: Callback for individual file progress (current, total)
            create_zip: Whether to create a zip file with results and audio files
            
        Returns:
            Dict containing batch transcription results
        """
        if not self.is_model_loaded or not self.unified_transcriber:
            # Try to load model first
            if not self.load_model(progress_callback):
                return {
                    'success': False,
                    'error': 'Failed to load transcription model',
                    'results': [],
                    'total_time': 0,
                    'success_count': 0,
                    'failure_count': 0,
                    'zip_path': None
                }
        
        # Delegate to unified transcriber which handles all the detailed processing
        result = self.unified_transcriber.transcribe_batch(
            input_directory, output_directory, progress_callback, file_progress_callback
        )
        
        # Create a zip file with results and audio files if requested
        zip_path = None
        if create_zip and result.get('success', False):
            if progress_callback:
                progress_callback("Creating results package...")
            
            zip_path = self.create_results_zip(
                input_directory, output_directory, progress_callback
            )
            
            if zip_path:
                result['zip_path'] = zip_path
        
        return result
    
    def get_available_models(self) -> List[str]:
        """Get list of available models for current backend."""
        if self.unified_transcriber:
            return self.unified_transcriber.get_available_models()
        else:
            # Return default models if transcriber not initialized
            return self.backend_manager.get_backend_info('openai').models if self.backend_manager else []
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backend names."""
        return [backend.name for backend in self.backend_manager.get_available_backends()]
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about current backend and device."""
        if self.unified_transcriber:
            return self.unified_transcriber.get_backend_info()
        else:
            # Return basic info if transcriber not initialized
            return {
                'backend_name': self.backend,
                'model_name': self.model_name,
                'device': self.device,
                'model_loaded': self.is_model_loaded
            }
    
    def get_device_info(self) -> Dict[str, str]:
        """Get comprehensive information about the current device."""
        if self.unified_transcriber:
            backend_info = self.unified_transcriber.get_backend_info()
            return {
                'device': backend_info.get('device', self.device),
                'device_type': backend_info.get('device', self.device).upper(),
                'device_name': backend_info.get('device_name', 'Unknown'),
                'backend': backend_info.get('backend_display_name', 'Unknown'),
                'model': backend_info.get('model_name', self.model_name),
                'cuda_available': torch.cuda.is_available(),
                'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            }
        else:
            # Fallback to basic device detection
            info = {
                'device': self.device,
                'device_type': self.device.upper(),
                'cuda_available': torch.cuda.is_available(),
                'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
                'device_name': 'Unknown',
                'backend': 'Not loaded',
                'model': self.model_name
            }
            
            if torch.cuda.is_available():
                try:
                    info['device_name'] = torch.cuda.get_device_name(0)
                except Exception:
                    info['device_name'] = 'CUDA GPU'
            elif self.device == 'mps':
                info['device_name'] = 'Apple Silicon GPU'
            else:
                info['device_name'] = 'CPU'
                
            return info
    
    def get_backend_detection_info(self) -> Dict[str, Any]:
        """Get detailed backend detection information for logging."""
        return self.backend_manager.get_detection_info()
    
    def get_models_for_backend(self, backend_name: str) -> List[str]:
        """Get available models for a specific backend."""
        backend_info = self.backend_manager.get_backend_info(backend_name)
        return backend_info.models if backend_info else []
        
    def create_results_zip(self, input_directory: str, output_directory: str, 
                          progress_callback: Optional[Callable] = None) -> str:
        """
        Create a zip file containing the HTML report and audio files.
        
        Args:
            input_directory: Directory containing audio files
            output_directory: Directory containing transcription results
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Path to the created zip file
        """
        if progress_callback:
            progress_callback("Creating results zip archive...")
            
        # Create a temporary directory for the zip contents
        temp_dir = Path(output_directory) / "temp_zip_contents"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Copy HTML report to temp directory
            html_report = Path(output_directory) / "transcription_report.html"
            if html_report.exists():
                shutil.copy2(html_report, temp_dir)
                
                if progress_callback:
                    progress_callback("Copied HTML report to zip archive")
            else:
                if progress_callback:
                    progress_callback("Warning: HTML report not found")
            
            # Copy audio files to temp directory
            audio_files = get_audio_files(input_directory)
            audio_dir = temp_dir / "audio"
            audio_dir.mkdir(exist_ok=True)
            
            # Keep track of additional web MP3 files
            web_mp3_files = []
            
            for i, audio_file in enumerate(audio_files, 1):
                file_path = Path(audio_file)
                file_name = file_path.name
                shutil.copy2(audio_file, audio_dir / file_name)
                
                # Check for web-compatible MP3 version if this is a WAV file
                if file_path.suffix.lower() == '.wav':
                    potential_mp3 = file_path.parent / f"{file_path.stem}_web.mp3"
                    if potential_mp3.exists():
                        web_mp3_files.append(potential_mp3)
                        shutil.copy2(potential_mp3, audio_dir / potential_mp3.name)
                
                if progress_callback and i % 5 == 0:  # Update every 5 files
                    progress_callback(f"Copying audio files: {i}/{len(audio_files)}")
            
            # Log the total number of files copied
            total_files = len(audio_files) + len(web_mp3_files)
            if progress_callback:
                progress_callback(f"Copied {total_files} audio files to zip archive ({len(web_mp3_files)} web-compatible MP3s)")
            
            # Update HTML file to use relative paths for audio files
            if html_report.exists():
                self._update_html_paths(temp_dir / html_report.name, "audio")
                
                if progress_callback:
                    progress_callback("Updated audio file paths in HTML report")
            
            # Create zip file
            zip_path = Path(output_directory) / "transcription_results.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in temp_dir.rglob('*'):
                    if file.is_file():
                        zipf.write(
                            file, 
                            file.relative_to(temp_dir)
                        )
            
            if progress_callback:
                progress_callback(f"✅ Created zip archive: {zip_path}")
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
            return str(zip_path)
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error creating zip archive: {str(e)}")
            
            # Clean up temporary directory if it exists
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                
            return ""
    
    def _update_html_paths(self, html_file: Path, audio_dir_name: str) -> None:
        """
        Update audio file paths in HTML report to use relative paths.
        
        Args:
            html_file: Path to the HTML file
            audio_dir_name: Name of the audio directory in the zip
        """
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            import re
            
            # Update all source elements to use relative paths
            # Replace any remaining absolute paths in audio sources
            content = re.sub(
                r'<source src="file:///[^"]*?([^/\\]+?)(?:")',
                f'<source src="{audio_dir_name}/\\1"',
                content
            )
            
            # Replace any remaining absolute URIs in onloadedmetadata
            content = re.sub(
                r'onloadedmetadata="initializeAudioPlayer\((\d+), \'file:///[^\']*?([^/\\]+?)(?:\')',
                f'onloadedmetadata="initializeAudioPlayer(\\1, \'{audio_dir_name}/\\2\'',
                content
            )
            
            # First, modify the href links for "Play Audio" buttons
            # Find href="file:///path/to/file.mp3" and replace with href="audio/file.mp3"
            content = re.sub(
                r'href="file:///[^"]*?([^/\\]+?)(?:")',
                f'href="{audio_dir_name}/\\1"',
                content
            )
            
            # Next, modify the onclick handlers for "Open Location" buttons
            # Find onclick="openFileLocation('path/to/file.mp3')" and replace with onclick="openFileLocation('audio/file.mp3')"
            content = re.sub(
                r'onclick="openFileLocation\(\'[^\']*?([^/\\]+?)(?:\'\))',
                f'onclick="openFileLocation(\'{audio_dir_name}/\\1\')',
                content
            )
            
            # Update JavaScript to handle relative paths
            old_js = """function openFileLocation(filePath) {
            // Try to open the file directly (may auto-play depending on system settings)
            window.open(filePath, '_blank');
            
            // For Windows, we can try to open in explorer
            if (navigator.platform.indexOf('Win') !== -1) {
                // This will work in some browsers/contexts
                try {
                    const explorerPath = 'file:///' + filePath.replace(/\\//g, '\\\\\\\\');
                    window.open(explorerPath, '_blank');
                } catch(e) {
                    console.log('Could not open in explorer:', e);
                }
            }
        }"""
            
            new_js = """function openFileLocation(filePath) {
            // Handle both relative and absolute paths
            // For relative paths in the zip archive, just open the file directly
            window.open(filePath, '_blank');
        }"""
            
            content = content.replace(old_js, new_js)
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except Exception as e:
            print(f"Error updating HTML paths: {e}")
            # Continue even if path updating fails