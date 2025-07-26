import os
import time
import torch
import whisper
import threading
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
        import time
        load_start_time = time.time()
        
        try:
            if progress_callback:
                progress_callback(f"Initializing Whisper model '{self.model_name}' on {self.device.upper()}...")
            
            # Log detailed device information
            device_info = self._get_detailed_device_info()
            
            if progress_callback:
                progress_callback(f"Target device: {device_info}")
            
            # Check available memory before loading
            if self.device == "cuda" and torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                free_gb = free_memory / (1024**3)
                if progress_callback:
                    progress_callback(f"Available GPU memory: {free_gb:.1f} GB")
            
            if progress_callback:
                progress_callback(f"Downloading/loading model weights for '{self.model_name}'...")
            
            self.model = whisper.load_model(self.model_name, device=self.device)
            self.is_model_loaded = True
            
            load_time = time.time() - load_start_time
            
            # Log post-load memory usage
            memory_info = ""
            if self.device == "cuda" and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                cached = torch.cuda.memory_reserved(0) / (1024**3)
                memory_info = f" (GPU memory: {allocated:.1f} GB allocated, {cached:.1f} GB cached)"
            
            success_msg = f"Model '{self.model_name}' loaded successfully in {load_time:.1f}s{memory_info}"
            
            if progress_callback:
                progress_callback(success_msg)
            
            return True
        except Exception as e:
            load_time = time.time() - load_start_time
            error_msg = f"Failed to load model '{self.model_name}' after {load_time:.1f}s: {str(e)}"
            
            if progress_callback:
                progress_callback(error_msg)
                
            # Provide troubleshooting suggestions
            if "CUDA" in str(e) or "cuda" in str(e):
                if progress_callback:
                    progress_callback("💡 CUDA error detected. Try selecting CPU device or restart the application")
            elif "memory" in str(e).lower():
                if progress_callback:
                    progress_callback("💡 Memory error. Try using a smaller model (tiny/base) or close other applications")
            
            return False
    
    def _get_detailed_device_info(self) -> str:
        """Get detailed information about the selected device."""
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
        Transcribe a single audio file with comprehensive logging.
        
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
        file_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
        
        try:
            if progress_callback:
                progress_callback(f"Processing: {file_name} ({file_size:.1f} MB)")
            
            # Get file duration and format information
            duration = get_file_duration(audio_file)
            file_ext = Path(audio_file).suffix.lower()
            
            if progress_callback:
                progress_callback(f"File info: {duration:.1f}s duration, {file_ext} format")
            
            # Log memory usage before transcription (for GPU)
            if self.device == "cuda" and torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated(0) / (1024**3)
                if progress_callback:
                                         progress_callback(f"GPU memory before transcription: {mem_before:.1f} GB")
            
            # Start transcription with timing
            transcription_start = time.time()
            if progress_callback:
                progress_callback(f"Starting Whisper transcription...")
                
                # Pre-transcription info for longer files
                if duration > 60:
                    progress_callback(f"📂 Loading {duration/60:.1f} minute audio file...")
                if file_size > 10:  # MB
                    progress_callback(f"💾 Processing large file ({file_size:.1f} MB)")
                if duration > 300:  # 5 minutes
                    progress_callback(f"⏱️ Long audio - this may take several minutes")
            
            # Create periodic progress updates for long transcriptions
            progress_finished = threading.Event()
            progress_tracker_thread = None
            
            if duration > 30 and progress_callback:
                def progress_tracker():
                    start_time = time.time()
                    while not progress_finished.wait(8):  # Update every 8 seconds
                        elapsed = time.time() - start_time
                        if elapsed < 30:
                            progress_callback(f"🔄 Transcribing... ({elapsed:.0f}s elapsed)")
                        elif elapsed < 90:
                            progress_callback(f"📝 Processing audio segments... ({elapsed:.0f}s elapsed)")
                        else:
                            progress_callback(f"🧠 Deep analysis... ({elapsed/60:.1f}m elapsed)")
                
                progress_tracker_thread = threading.Thread(target=progress_tracker, daemon=True)
                progress_tracker_thread.start()
            
            try:
                # Transcribe the audio
                result = self.model.transcribe(audio_file)
            finally:
                # Stop progress tracker
                if progress_tracker_thread:
                    progress_finished.set()
            
            transcription_time = time.time() - transcription_start
            processing_time = time.time() - start_time
            
            # Calculate performance metrics
            realtime_factor = duration / transcription_time if transcription_time > 0 else 0
            words_count = len(result["text"].split()) if result["text"] else 0
            
            # Log memory usage after transcription (for GPU)
            if self.device == "cuda" and torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated(0) / (1024**3)
                if progress_callback:
                    progress_callback(f"GPU memory after transcription: {mem_after:.1f} GB")
            
            # Determine detected language and segments
            language = result.get('language', 'unknown')
            segments = result.get("segments", [])
            
            if progress_callback:
                progress_callback(f"✅ Transcription complete: {words_count} words detected")
                progress_callback(f"🌍 Language detected: {language}")
                progress_callback(f"⚡ Processing speed: {realtime_factor:.1f}x realtime")
                progress_callback(f"⏱️ Total time: {processing_time:.1f}s (transcription: {transcription_time:.1f}s)")
                
                # Log segment information if available
                if segments:
                    progress_callback(f"📝 Generated {len(segments)} text segments")
            
            return {
                'file_path': audio_file,
                'success': True,
                'error': None,
                'transcription': result["text"],
                'segments': segments,
                'duration': duration,
                'processing_time': processing_time,
                'transcription_time': transcription_time,
                'language': language,
                'words_count': words_count,
                'realtime_factor': realtime_factor,
                'file_size_mb': file_size
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            if progress_callback:
                progress_callback(f"❌ Error transcribing {file_name}: {error_msg}")
                
                # Provide specific error diagnostics
                if "out of memory" in error_msg.lower() or "memory" in error_msg.lower():
                    progress_callback("💡 Memory error: Try using a smaller model or processing fewer files")
                elif "cuda" in error_msg.lower():
                    progress_callback("💡 CUDA error: Try switching to CPU mode")
                elif "file" in error_msg.lower() or "format" in error_msg.lower():
                    progress_callback("💡 File error: Check if the audio file is corrupted or in an unsupported format")
                elif "ffmpeg" in error_msg.lower():
                    progress_callback("💡 FFmpeg error: Audio format may not be supported")
            
            return {
                'file_path': audio_file,
                'success': False,
                'error': error_msg,
                'transcription': None,
                'duration': get_file_duration(audio_file),
                'processing_time': processing_time,
                'file_size_mb': file_size
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
        
        # Get all audio files with detailed analysis
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
        
        # Analyze the batch before processing
        total_size = sum(os.path.getsize(f) for f in audio_files) / (1024 * 1024)  # MB
        file_types = {}
        total_duration = 0
        
        if progress_callback:
            progress_callback(f"Analyzing {len(audio_files)} audio files...")
        
        for audio_file in audio_files:
            file_ext = Path(audio_file).suffix.lower()
            file_types[file_ext] = file_types.get(file_ext, 0) + 1
            try:
                duration = get_file_duration(audio_file)
                total_duration += duration
            except:
                pass  # Skip files with duration issues
        
        if progress_callback:
            progress_callback(f"Batch analysis complete:")
            progress_callback(f"  └── Total files: {len(audio_files)}")
            progress_callback(f"  └── Total size: {total_size:.1f} MB")
            progress_callback(f"  └── Estimated duration: {total_duration/60:.1f} minutes")
            
            # Log file type breakdown
            type_summary = ", ".join([f"{ext}({count})" for ext, count in sorted(file_types.items())])
            progress_callback(f"  └── File types: {type_summary}")
            
            # Estimate processing time
            if hasattr(self, 'device') and self.device == 'cuda':
                estimated_time = total_duration * 0.2  # Rough GPU estimate
                progress_callback(f"  └── Estimated processing time (GPU): {estimated_time/60:.1f} minutes")
            else:
                estimated_time = total_duration * 1.5  # Rough CPU estimate  
                progress_callback(f"  └── Estimated processing time (CPU): {estimated_time/60:.1f} minutes")
        
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
        
        # Calculate comprehensive batch statistics
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        # Performance metrics
        total_audio_duration = sum(r.get('duration', 0) for r in successful_results)
        total_transcription_time = sum(r.get('transcription_time', 0) for r in successful_results)
        avg_realtime_factor = sum(r.get('realtime_factor', 0) for r in successful_results) / len(successful_results) if successful_results else 0
        total_words = sum(r.get('words_count', 0) for r in successful_results)
        
        # Language detection statistics
        languages = {}
        for result in successful_results:
            lang = result.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
        
        if progress_callback:
            progress_callback("=== BATCH TRANSCRIPTION COMPLETE ===")
            progress_callback(f"Processing summary:")
            progress_callback(f"  └── Total files: {len(audio_files)}")
            progress_callback(f"  └── Successful: {success_count}")
            progress_callback(f"  └── Failed: {failure_count}")
            progress_callback(f"  └── Success rate: {(success_count/len(audio_files)*100):.1f}%")
            
            if successful_results:
                progress_callback(f"Performance metrics:")
                progress_callback(f"  └── Total audio duration: {total_audio_duration/60:.1f} minutes")
                progress_callback(f"  └── Total processing time: {total_time/60:.1f} minutes")
                progress_callback(f"  └── Average speed: {avg_realtime_factor:.1f}x realtime")
                progress_callback(f"  └── Total words transcribed: {total_words:,}")
                progress_callback(f"  └── Average words per minute: {(total_words/(total_time/60)):.0f}" if total_time > 0 else "  └── Average words per minute: N/A")
                
                # Language breakdown
                if languages:
                    lang_summary = ", ".join([f"{lang}({count})" for lang, count in sorted(languages.items())])
                    progress_callback(f"  └── Languages detected: {lang_summary}")
            
            # Report any failures
            if failed_results:
                progress_callback(f"Failed files:")
                for failed in failed_results[:3]:  # Show first 3 failures
                    file_name = Path(failed['file_path']).name
                    error = failed.get('error', 'Unknown error')
                    progress_callback(f"  └── {file_name}: {error}")
                if len(failed_results) > 3:
                    progress_callback(f"  └── ... and {len(failed_results) - 3} more failures")
        
        return {
            'success': True,
            'error': None,
            'results': results,
            'total_time': total_time,
            'success_count': success_count,
            'failure_count': failure_count,
            'audio_files_count': len(audio_files),
            'total_audio_duration': total_audio_duration,
            'total_transcription_time': total_transcription_time,
            'avg_realtime_factor': avg_realtime_factor,
            'total_words': total_words,
            'languages_detected': languages
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