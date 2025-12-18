import os
import time
import torch
import threading
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from utils import get_file_duration


@dataclass
class BackendInfo:
    """Information about a transcription backend."""
    name: str
    display_name: str
    available: bool
    priority: int  # Higher = preferred
    models: List[str]
    requirements: List[str]
    description: str
    gpu_support: bool = True
    cpu_support: bool = True


class BackendManager:
    """Manages and auto-selects transcription backends."""
    
    def __init__(self):
        self.backends = {}
        self.detect_backends()
    
    def detect_backends(self) -> None:
        """Detect available transcription backends."""
        self.backends = {}
        
        # Detect OpenAI Whisper
        openai_available = self._check_openai_whisper()
        self.backends['openai'] = BackendInfo(
            name='openai',
            display_name='OpenAI Whisper',
            available=openai_available,
            priority=10,  # Lower priority, fallback option
            models=['tiny', 'base', 'small', 'medium', 'large'],
            requirements=['openai-whisper', 'torch'],
            description='Original OpenAI Whisper implementation. Reliable and well-tested.',
            gpu_support=True,
            cpu_support=True
        )
        
        # Detect faster-whisper
        faster_available, has_gpu_support = self._check_faster_whisper()
        self.backends['faster'] = BackendInfo(
            name='faster',
            display_name='Faster Whisper',
            available=faster_available,
            priority=20 if has_gpu_support else 15,  # Higher priority if GPU support
            models=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
            requirements=['faster-whisper', 'onnxruntime-gpu'],
            description='Optimized Whisper implementation. Faster processing with GPU acceleration.',
            gpu_support=has_gpu_support,
            cpu_support=True
        )
    
    def _check_openai_whisper(self) -> bool:
        """Check if OpenAI Whisper is available."""
        try:
            import whisper
            return True
        except ImportError:
            return False
    
    def _check_faster_whisper(self) -> Tuple[bool, bool]:
        """Check if faster-whisper is available and has GPU support."""
        try:
            import faster_whisper
            
            # Check for GPU support via onnxruntime-gpu
            gpu_support = False
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                gpu_support = any(provider in providers for provider in [
                    'CUDAExecutionProvider', 
                    'TensorrtExecutionProvider',
                    'DmlExecutionProvider'  # DirectML for Windows GPU
                ])
            except ImportError:
                pass
            
            return True, gpu_support
        except ImportError:
            return False, False
    
    def get_available_backends(self) -> List[BackendInfo]:
        """Get list of available backends sorted by priority."""
        available = [info for info in self.backends.values() if info.available]
        return sorted(available, key=lambda x: x.priority, reverse=True)
    
    def get_backend_info(self, backend_name: str) -> Optional[BackendInfo]:
        """Get information about a specific backend."""
        return self.backends.get(backend_name)
    
    def auto_select_backend(self, device: Optional[str] = None) -> str:
        """
        Auto-select the best backend based on available hardware.
        
        Priority logic:
        1. If NVIDIA GPU + faster-whisper with GPU support -> faster-whisper
        2. If any GPU + faster-whisper available -> faster-whisper  
        3. If GPU available + openai-whisper -> openai-whisper
        4. Fallback to best available backend
        """
        available_backends = self.get_available_backends()
        
        if not available_backends:
            raise RuntimeError("No transcription backends available")
        
        # Detect device if not provided
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        # Check for NVIDIA GPU with CUDA
        has_nvidia_gpu = device == 'cuda' and torch.cuda.is_available()
        has_any_gpu = device in ['cuda', 'mps']
        
        # Selection logic
        for backend in available_backends:
            if backend.name == 'faster':
                # Prefer faster-whisper if GPU support is available
                if has_nvidia_gpu and backend.gpu_support:
                    return backend.name
                elif has_any_gpu and backend.available:
                    return backend.name
            elif backend.name == 'openai':
                # OpenAI as fallback for GPU
                if has_any_gpu and backend.available:
                    return backend.name
        
        # Return the highest priority available backend
        return available_backends[0].name
    
    def get_detection_info(self) -> Dict[str, Any]:
        """Get comprehensive detection information for logging."""
        info = {
            'backends_detected': {},
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'recommended_backend': None,
            'gpu_info': {},
            'compatibility_warnings': []
        }
        
        # Backend information
        for name, backend in self.backends.items():
            info['backends_detected'][name] = {
                'available': backend.available,
                'display_name': backend.display_name,
                'gpu_support': backend.gpu_support,
                'models': backend.models,
                'priority': backend.priority
            }
        
        # GPU information and compatibility check
        if torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                info['gpu_info'] = {
                    'device_name': device_name,
                    'memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    'device_count': torch.cuda.device_count()
                }
                
                # Check for newer GPU compatibility issues
                if "RTX 50" in device_name or "RTX 5080" in device_name:
                    info['compatibility_warnings'].append(
                        "RTX 5080 detected: GPU may have CUDA compatibility issues. CPU fallback available."
                    )
                    
            except Exception as e:
                info['gpu_info'] = {'error': str(e)}
                info['compatibility_warnings'].append(f"GPU detection error: {str(e)}")
        
        # ONNX Runtime information for faster-whisper
        try:
            import onnxruntime as ort
            info['onnx_providers'] = ort.get_available_providers()
            
            # Check ONNX GPU compatibility
            gpu_providers = [p for p in info['onnx_providers'] if 'CUDA' in p or 'Tensorrt' in p or 'Dml' in p]
            if torch.cuda.is_available() and not gpu_providers:
                info['compatibility_warnings'].append(
                    "CUDA GPU detected but no ONNX GPU providers available. Faster-whisper will use CPU."
                )
                
        except ImportError:
            info['onnx_providers'] = []
            if torch.cuda.is_available():
                info['compatibility_warnings'].append(
                    "ONNX Runtime not available. Faster-whisper backend cannot utilize GPU acceleration."
                )
        
        # Auto-selected backend
        try:
            info['recommended_backend'] = self.auto_select_backend()
        except Exception as e:
            info['recommended_backend'] = f"Error: {str(e)}"
        
        return info


class UnifiedTranscriber:
    """Unified interface for both OpenAI Whisper and faster-whisper backends."""
    
    def __init__(self, backend: str = "auto", model_name: str = "base", 
                 device: Optional[str] = None, beam_size: int = 5, language: Optional[str] = None):
        """
        Initialize the unified transcriber.
        
        Args:
            backend: Backend to use ("auto", "openai", "faster")
            model_name: Model size to use
            device: Device to use ("cuda", "cpu", "mps", or None for auto-detection)
            beam_size: Beam size for faster-whisper (ignored for OpenAI)
            language: Language code (e.g., "en" for English) or None for auto-detect
        """
        self.backend_manager = BackendManager()
        self.beam_size = beam_size
        self.device = self._get_device() if device is None else device
        self.language = language
        
        # Resolve backend
        if backend == "auto":
            self.backend_name = self.backend_manager.auto_select_backend(self.device)
        else:
            self.backend_name = backend
        
        # Validate backend
        backend_info = self.backend_manager.get_backend_info(self.backend_name)
        if not backend_info or not backend_info.available:
            raise RuntimeError(f"Backend '{self.backend_name}' is not available")
        
        self.backend_info = backend_info
        self.model_name = model_name
        self.model = None
        self.is_model_loaded = False
        
        # Validate model for backend
        if model_name not in backend_info.models:
            raise ValueError(f"Model '{model_name}' not available for backend '{self.backend_name}'")
    
    def _get_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self, progress_callback: Optional[Callable] = None) -> bool:
        """Load the transcription model."""
        load_start_time = time.time()
        
        try:
            if progress_callback:
                progress_callback(f"Loading {self.backend_info.display_name} model '{self.model_name}' on {self.device.upper()}...")
            
            if self.backend_name == "openai":
                self._load_openai_model(progress_callback)
            elif self.backend_name == "faster":
                self._load_faster_model(progress_callback)
            else:
                raise RuntimeError(f"Unknown backend: {self.backend_name}")
            
            self.is_model_loaded = True
            load_time = time.time() - load_start_time
            
            if progress_callback:
                progress_callback(f"✅ {self.backend_info.display_name} model loaded successfully in {load_time:.1f}s")
            
            return True
            
        except Exception as e:
            load_time = time.time() - load_start_time
            error_msg = f"Failed to load {self.backend_info.display_name} model: {str(e)}"
            
            if progress_callback:
                progress_callback(error_msg)
                
                # Provide troubleshooting suggestions
                if "CUDA" in str(e) or "cuda" in str(e):
                    progress_callback("💡 CUDA error detected. Try CPU device or install CUDA drivers")
                elif "memory" in str(e).lower():
                    progress_callback("💡 Memory error. Try a smaller model or close other applications")
                elif "onnx" in str(e).lower():
                    progress_callback("💡 ONNX error. Try installing onnxruntime-gpu for faster-whisper")
            
            return False
    
    def _load_openai_model(self, progress_callback: Optional[Callable] = None):
        """Load OpenAI Whisper model with GPU fallback."""
        import whisper
        
        if progress_callback:
            progress_callback(f"Downloading/loading OpenAI Whisper '{self.model_name}' weights...")
        
        # Try GPU first, fallback to CPU if needed
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
        except Exception as gpu_error:
            if self.device in ["cuda", "mps"] and "CUDA" in str(gpu_error):
                if progress_callback:
                    progress_callback(f"⚠️ GPU incompatibility detected (RTX 5080 may need newer CUDA). Falling back to CPU...")
                
                # Fallback to CPU
                self.device = "cpu"
                self.model = whisper.load_model(self.model_name, device="cpu")
                
                if progress_callback:
                    progress_callback(f"✅ Successfully loaded on CPU as fallback")
            else:
                raise gpu_error
    
    def _load_faster_model(self, progress_callback: Optional[Callable] = None):
        """Load faster-whisper model with GPU fallback."""
        from faster_whisper import WhisperModel
        
        if progress_callback:
            progress_callback(f"Initializing faster-whisper model '{self.model_name}'...")
        
        # Map device for faster-whisper
        if self.device == "cuda":
            device = "cuda"
            compute_type = "float16"  # Use float16 for better GPU performance
        elif self.device == "mps":
            device = "cpu"  # faster-whisper doesn't support MPS directly
            compute_type = "int8"
        else:
            device = "cpu"
            compute_type = "int8"
        
        if progress_callback:
            progress_callback(f"Using device: {device}, compute_type: {compute_type}")
        
        # Try with requested device first, fallback to CPU if needed
        try:
            self.model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute_type
            )
        except Exception as gpu_error:
            if device == "cuda" and ("CUDA" in str(gpu_error) or "onnxruntime" in str(gpu_error).lower()):
                if progress_callback:
                    progress_callback(f"⚠️ GPU error with faster-whisper (RTX 5080 compatibility). Falling back to CPU...")
                
                # Fallback to CPU
                self.device = "cpu"
                self.model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8"
                )
                
                if progress_callback:
                    progress_callback(f"✅ Faster-whisper loaded successfully on CPU")
            else:
                raise gpu_error
    
    def transcribe_file(self, audio_file: str, 
                       progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Transcribe a single audio file."""
        if not self.is_model_loaded:
            return {
                'file_path': audio_file,
                'success': False,
                'error': 'Model not loaded',
                'transcription': None,
                'duration': 0,
                'backend': self.backend_name
            }
        
        start_time = time.time()
        file_name = Path(audio_file).name
        file_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
        
        try:
            if progress_callback:
                progress_callback(f"Processing: {file_name} ({file_size:.1f} MB) using {self.backend_info.display_name}")
            
            duration = get_file_duration(audio_file)
            
            # Transcribe using appropriate backend
            if self.backend_name == "openai":
                result = self._transcribe_openai(audio_file, progress_callback)
            elif self.backend_name == "faster":
                result = self._transcribe_faster(audio_file, progress_callback)
            else:
                raise RuntimeError(f"Unknown backend: {self.backend_name}")
            
            processing_time = time.time() - start_time
            
            # Normalize result format
            normalized_result = self._normalize_result(result, audio_file, duration, processing_time)
            
            if progress_callback:
                words_count = normalized_result.get('words_count', 0)
                language = normalized_result.get('language', 'unknown')
                progress_callback(f"✅ Transcription complete: {words_count} words, language: {language}")
            
            return normalized_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(f"❌ Error transcribing {file_name}: {str(e)}")
            
            return {
                'file_path': audio_file,
                'success': False,
                'error': str(e),
                'transcription': None,
                'duration': get_file_duration(audio_file),
                'processing_time': processing_time,
                'backend': self.backend_name
            }
    
    def _transcribe_openai(self, audio_file: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper."""
        if progress_callback:
            if self.language:
                progress_callback(f"Starting OpenAI Whisper transcription (language: {self.language})...")
            else:
                progress_callback("Starting OpenAI Whisper transcription (auto-detecting language)...")
        
        result = self.model.transcribe(audio_file, language=self.language)
        return result
    
    def _transcribe_faster(self, audio_file: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Transcribe using faster-whisper."""
        if progress_callback:
            if self.language:
                progress_callback(f"Starting faster-whisper transcription (beam_size: {self.beam_size}, language: {self.language})...")
            else:
                progress_callback(f"Starting faster-whisper transcription (beam_size: {self.beam_size}, auto-detecting language)...")
        
        # Optional initial prompt: set via environment variable WHISPER_INITIAL_PROMPT
        initial_prompt = os.environ.get("WHISPER_INITIAL_PROMPT", None)

        segments, info = self.model.transcribe(
            audio_file,
            beam_size=1,  # greedy, fast
            temperature=[0.0, 0.2],  # deterministic, with light fallback on low-confidence
            best_of=1,
            language=self.language,  # None for auto-detect, "en" for English, etc.
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            word_timestamps=True,
            initial_prompt=initial_prompt if initial_prompt else None,
            task="transcribe"
        )
        
        # Convert segments to list and build transcript
        segments_list = list(segments)
        full_text = " ".join([segment.text.strip() for segment in segments_list])
        
        # Convert to OpenAI Whisper format for compatibility
        result = {
            "text": full_text,
            "segments": [
                {
                    "id": i,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
                for i, segment in enumerate(segments_list)
            ],
            "language": info.language
        }
        
        return result
    
    def _normalize_result(self, result: Dict[str, Any], audio_file: str, 
                         duration: float, processing_time: float) -> Dict[str, Any]:
        """Normalize transcription result to consistent format."""
        text = result.get("text", "")
        segments = result.get("segments", [])
        language = result.get("language", "unknown")
        
        words_count = len(text.split()) if text else 0
        realtime_factor = duration / processing_time if processing_time > 0 else 0
        
        return {
            'file_path': audio_file,
            'success': True,
            'error': None,
            'transcription': text,
            'segments': segments,
            'duration': duration,
            'processing_time': processing_time,
            'language': language,
            'words_count': words_count,
            'realtime_factor': realtime_factor,
            'backend': self.backend_name,
            'model_name': self.model_name,
            'beam_size': self.beam_size if self.backend_name == "faster" else None
        }
    
    def transcribe_batch(self, input_directory: str, output_directory: str,
                        progress_callback: Optional[Callable] = None,
                        file_progress_callback: Optional[Callable] = None,
                        cancellation_check: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        """Transcribe all audio files in a directory."""
        from utils import get_audio_files, safe_filename, create_html_report
        from audio_converter import process_audio_files_for_web_compatibility
        
        start_time = time.time()
        
        # Get all audio files
        audio_files = get_audio_files(input_directory, sort_by_date=True, debug_log=True)
        
        if not audio_files:
            return {
                'success': False,
                'error': 'No supported audio files found in the directory',
                'results': [],
                'total_time': 0,
                'success_count': 0,
                'failure_count': 0,
                'backend': self.backend_name,
                'cancelled': False
            }
        
        # Check for cancellation before starting
        if cancellation_check and cancellation_check():
            return {
                'success': False,
                'error': 'Transcription cancelled by user',
                'results': [],
                'total_time': 0,
                'success_count': 0,
                'failure_count': 0,
                'backend': self.backend_name,
                'cancelled': True
            }
        
        if progress_callback:
            progress_callback(f"Found {len(audio_files)} files to process using {self.backend_info.display_name}")
        
        # Process audio files for web compatibility (convert problematic WAV files)
        if progress_callback:
            progress_callback("🔍 Checking audio files for web compatibility...")
        
        # Check for cancellation during file processing
        if cancellation_check and cancellation_check():
            return {
                'success': False,
                'error': 'Transcription cancelled by user',
                'results': [],
                'total_time': time.time() - start_time,
                'success_count': 0,
                'failure_count': 0,
                'backend': self.backend_name,
                'cancelled': True
            }
        
        processed_audio_files, converted_count = process_audio_files_for_web_compatibility(
            audio_files, progress_callback
        )
        
        # Update the file list to use processed versions
        audio_files = processed_audio_files
        
        # Add conversion metadata to results
        conversion_info = {
            'wav_files_converted': converted_count,
            'total_files_processed': len(audio_files),
            'has_web_compatible_versions': converted_count > 0
        }
        
        if progress_callback and converted_count > 0:
            progress_callback(f"✅ Audio compatibility check complete: {converted_count} file(s) converted to web-compatible format")
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Load model if not already loaded
        if not self.is_model_loaded:
            if not self.load_model(progress_callback):
                return {
                    'success': False,
                    'error': f'Failed to load {self.backend_info.display_name} model',
                    'results': [],
                    'total_time': 0,
                    'success_count': 0,
                    'failure_count': 0,
                    'backend': self.backend_name,
                    'cancelled': False
                }
        
        # Process files
        results = []
        success_count = 0
        failure_count = 0
        
        for i, audio_file in enumerate(audio_files, 1):
            # Check for cancellation before processing each file
            if cancellation_check and cancellation_check():
                if progress_callback:
                    progress_callback("⚠️ Transcription cancelled by user")
                return {
                    'success': False,
                    'error': 'Transcription cancelled by user',
                    'results': results,
                    'total_time': time.time() - start_time,
                    'success_count': success_count,
                    'failure_count': failure_count,
                    'backend': self.backend_name,
                    'cancelled': True,
                    'conversion_info': conversion_info
                }
            
            if file_progress_callback:
                file_progress_callback(i, len(audio_files))
            
            result = self.transcribe_file(audio_file, progress_callback)
            results.append(result)
            
            if result['success']:
                success_count += 1
                # Individual .txt files are no longer created
            else:
                failure_count += 1
        
        total_time = time.time() - start_time
        
        if progress_callback:
            progress_callback(f"✅ Batch complete: {success_count} successful, {failure_count} failed")
            progress_callback(f"Generating output files...")
        
        # Get backend info for reports
        backend_info = self.get_backend_info()
        
        # Create HTML report with progress updates
        try:
            # Create a wrapper for progress callback that handles both formats
            def html_progress_wrapper(message: str, percentage: int = None):
                """Wrapper to handle progress callback with optional percentage."""
                if progress_callback:
                    try:
                        # Try new format with percentage
                        if percentage is not None:
                            progress_callback(f"[{percentage}%] {message}")
                        else:
                            progress_callback(message)
                    except TypeError:
                        # Fallback to old format
                        progress_callback(message)
            
            html_path = create_html_report(
                results, output_directory, total_time, success_count, failure_count,
                progress_callback=html_progress_wrapper
            )
            if html_path and progress_callback:
                progress_callback(f"✅ Created HTML report: {os.path.basename(html_path)}")
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error creating HTML report: {str(e)}")
        
        # JSON transcript creation has been disabled
        
        return {
            'success': True,
            'error': None,
            'results': results,
            'total_time': total_time,
            'success_count': success_count,
            'failure_count': failure_count,
            'backend': self.backend_name,
            'model_name': self.model_name,
            'conversion_info': conversion_info,
            'cancelled': False
        }
    
    def _save_individual_transcription(self, result: Dict[str, Any], output_directory: str):
        """Save individual transcription to a text file."""
        try:
            from utils import safe_filename
            
            file_name = Path(result['file_path']).stem
            safe_name = safe_filename(file_name)
            output_file = os.path.join(output_directory, f"{safe_name}.txt")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"File: {Path(result['file_path']).name}\n")
                f.write(f"Backend: {result.get('backend', 'unknown')}\n")
                f.write(f"Model: {result.get('model_name', 'unknown')}\n")
                f.write(f"Duration: {result.get('duration', 0):.1f} seconds\n")
                f.write(f"Language: {result.get('language', 'unknown')}\n")
                f.write(f"Processing Time: {result.get('processing_time', 0):.1f} seconds\n")
                if result.get('beam_size'):
                    f.write(f"Beam Size: {result.get('beam_size')}\n")
                f.write("-" * 50 + "\n\n")
                f.write(result['transcription'])
                
        except Exception as e:
            print(f"Error saving individual transcription: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models for current backend."""
        return self.backend_info.models if self.backend_info else []
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about current backend and device."""
        info = {
            'backend_name': self.backend_name,
            'backend_display_name': self.backend_info.display_name if self.backend_info else 'Unknown',
            'model_name': self.model_name,
            'device': self.device,
            'beam_size': self.beam_size if self.backend_name == "faster" else None,
            'model_loaded': self.is_model_loaded
        }
        
        # Add device-specific information
        if self.device == 'cuda' and torch.cuda.is_available():
            info['device_name'] = torch.cuda.get_device_name(0)
            info['device_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        elif self.device == 'mps':
            info['device_name'] = 'Apple Silicon GPU'
        else:
            info['device_name'] = 'CPU'
            info['cpu_cores'] = os.cpu_count()
        
        return info 