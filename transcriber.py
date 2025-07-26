import os
import time
import torch
import whisper
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from utils import get_audio_files, get_file_duration, safe_filename


class AudioTranscriber:
    """Core audio transcription class using OpenAI Whisper."""
    
    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        """
        Initialize the transcriber with a Whisper model.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to use ("cuda", "cpu", or None for auto-detection)
        """
        self.model_name = model_name
        self.device = self._get_device() if device is None else device
        self.model = None
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
        Load the Whisper model with enhanced device feedback.
        
        Args:
            progress_callback: Optional callback function for progress updates
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if progress_callback:
                progress_callback(f"Loading Whisper model '{self.model_name}' on {self.device.upper()}...")
            
            # Log device information
            if self.device == "cuda":
                device_info = f"CUDA GPU: {torch.cuda.get_device_name(0)}"
            elif self.device == "mps":
                device_info = "Apple Silicon GPU (MPS)"
            else:
                device_info = f"CPU ({os.cpu_count()} cores)"
            
            print(f"🔄 Loading Whisper model '{self.model_name}' on {device_info}")
            
            self.model = whisper.load_model(self.model_name, device=self.device)
            self.is_model_loaded = True
            
            success_msg = f"✅ Model '{self.model_name}' loaded successfully on {device_info}"
            print(success_msg)
            
            if progress_callback:
                progress_callback(success_msg)
            
            return True
        except Exception as e:
            error_msg = f"❌ Error loading model: {str(e)}"
            print(error_msg)
            if progress_callback:
                progress_callback(error_msg)
            return False
    
    def transcribe_file(self, audio_file: str, 
                       progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Transcribe a single audio file.
        
        Args:
            audio_file: Path to the audio file
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dict containing transcription results and metadata
        """
        if not self.is_model_loaded:
            return {
                'file_path': audio_file,
                'success': False,
                'error': 'Model not loaded',
                'transcription': None,
                'duration': 0
            }
        
        start_time = time.time()
        file_name = Path(audio_file).name
        
        try:
            if progress_callback:
                progress_callback(f"Transcribing: {file_name}")
            
            # Get file duration
            duration = get_file_duration(audio_file)
            
            # Transcribe the audio
            result = self.model.transcribe(audio_file)
            
            processing_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(f"Completed: {file_name} ({processing_time:.1f}s)")
            
            return {
                'file_path': audio_file,
                'success': True,
                'error': None,
                'transcription': result["text"],
                'duration': duration,
                'processing_time': processing_time,
                'language': result.get('language', 'unknown')
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            if progress_callback:
                progress_callback(f"Error transcribing {file_name}: {error_msg}")
            
            return {
                'file_path': audio_file,
                'success': False,
                'error': error_msg,
                'transcription': None,
                'duration': get_file_duration(audio_file),
                'processing_time': processing_time
            }
    
    def transcribe_batch(self, input_directory: str, output_directory: str,
                        progress_callback: Optional[Callable] = None,
                        file_progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Transcribe all audio files in a directory.
        
        Args:
            input_directory: Directory containing audio files
            output_directory: Directory to save transcriptions
            progress_callback: Callback for overall progress updates
            file_progress_callback: Callback for individual file progress (current, total)
            
        Returns:
            Dict containing batch transcription results
        """
        start_time = time.time()
        
        # Get all audio files
        audio_files = get_audio_files(input_directory)
        
        if not audio_files:
            return {
                'success': False,
                'error': 'No supported audio files found in the directory',
                'results': [],
                'total_time': 0,
                'success_count': 0,
                'failure_count': 0
            }
        
        if progress_callback:
            progress_callback(f"Found {len(audio_files)} audio files to transcribe")
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Load model if not already loaded
        if not self.is_model_loaded:
            if not self.load_model(progress_callback):
                return {
                    'success': False,
                    'error': 'Failed to load Whisper model',
                    'results': [],
                    'total_time': 0,
                    'success_count': 0,
                    'failure_count': 0
                }
        
        # Process each file
        results = []
        success_count = 0
        failure_count = 0
        
        for i, audio_file in enumerate(audio_files, 1):
            if file_progress_callback:
                file_progress_callback(i, len(audio_files))
            
            # Transcribe the file
            result = self.transcribe_file(audio_file, progress_callback)
            results.append(result)
            
            if result['success']:
                success_count += 1
                # Save individual transcription file
                self._save_individual_transcription(result, output_directory)
            else:
                failure_count += 1
        
        total_time = time.time() - start_time
        
        if progress_callback:
            progress_callback(f"Batch transcription completed. Success: {success_count}, Failed: {failure_count}")
        
        return {
            'success': True,
            'error': None,
            'results': results,
            'total_time': total_time,
            'success_count': success_count,
            'failure_count': failure_count,
            'audio_files_count': len(audio_files)
        }
    
    def _save_individual_transcription(self, result: Dict[str, Any], output_directory: str):
        """Save individual transcription to a text file."""
        try:
            file_name = Path(result['file_path']).stem
            safe_name = safe_filename(file_name)
            output_file = os.path.join(output_directory, f"{safe_name}.txt")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"File: {Path(result['file_path']).name}\n")
                f.write(f"Duration: {result.get('duration', 0):.1f} seconds\n")
                f.write(f"Language: {result.get('language', 'unknown')}\n")
                f.write(f"Processing Time: {result.get('processing_time', 0):.1f} seconds\n")
                f.write("-" * 50 + "\n\n")
                f.write(result['transcription'])
                
        except Exception as e:
            print(f"Error saving individual transcription: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Whisper models."""
        return ["tiny", "base", "small", "medium", "large"]
    
    def get_device_info(self) -> Dict[str, str]:
        """Get comprehensive information about the current device."""
        info = {
            'device': self.device,
            'device_type': self.device.upper(),
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'device_name': 'Unknown',
            'memory_info': 'Unknown',
            'cpu_cores': str(os.cpu_count())
        }
        
        if torch.cuda.is_available():
            try:
                info['device_name'] = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info['memory_info'] = f"{memory_gb:.1f} GB"
                if self.device == 'cuda':
                    # Get current GPU memory usage
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    cached = torch.cuda.memory_reserved(0) / (1024**3)
                    info['memory_usage'] = f"Allocated: {allocated:.1f} GB, Cached: {cached:.1f} GB"
            except Exception as e:
                info['device_name'] = f"CUDA Error: {str(e)}"
        elif self.device == 'mps':
            info['device_name'] = 'Apple Silicon GPU'
            info['memory_info'] = 'Shared with system memory'
        else:
            info['device_name'] = 'CPU'
            info['memory_info'] = 'System RAM'
            
        return info 