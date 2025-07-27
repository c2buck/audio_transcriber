#!/usr/bin/env python3
"""
Audio Transcriber Application
Main entry point for the desktop audio transcription app using OpenAI Whisper.

Features:
- Modern PySide6 GUI with dark/light theme support
- Batch transcription of audio files
- Multiple Whisper model sizes (tiny, base, small, medium, large)
- Automatic GPU/CPU detection
- Real-time progress tracking and logging
- HTML report generation with audio file links
- Individual text file outputs for each transcription

Author: AI Assistant
Version: 1.0
"""

import sys
import os
import torch
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Optional imports for backend detection
try:
    import faster_whisper
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

def detect_and_log_gpu():
    """Detect available computing devices and transcription backends."""
    print("=" * 60)
    print("🔍 SYSTEM HARDWARE & BACKEND DETECTION")
    print("=" * 60)
    
    # Check PyTorch installation details
    print(f"🐍 PyTorch Version: {torch.__version__}")
    if hasattr(torch.version, 'cuda') and torch.version.cuda:
        print(f"⚡ PyTorch CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  PyTorch CUDA Version: Not available (CPU-only installation)")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"🖥️  CUDA Available: {cuda_available}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"🎮 GPU Count: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"   └── GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # RTX 5080 specific information
            if "RTX 5080" in gpu_name or "RTX 50" in gpu_name:
                print(f"   ⚡ RTX 5080 detected! This GPU requires:")
                print(f"      • CUDA 12.4+ and newer drivers")
                print(f"      • onnxruntime-gpu 1.19.0+")
                print(f"      • PyTorch with CUDA 12.4 support")
        
        # Test GPU accessibility
        try:
            test_tensor = torch.tensor([1.0]).cuda()
            current_device = torch.cuda.current_device()
            print(f"✅ GPU {current_device} is ready for processing")
        except Exception as e:
            print(f"⚠️  GPU detected but not accessible: {e}")
            print(f"   💡 Try running: python install_gpu_support.py")
            cuda_available = False
    else:
        print("💾 No CUDA GPU detected")
        print("💡 For RTX 5080 support, ensure you have:")
        print("   • Latest NVIDIA drivers")
        print("   • PyTorch with CUDA 12.4+ support")
        print("   • Run: python install_gpu_support.py")
    
    # Check for Apple Silicon (MPS)
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if mps_available:
        print("🍎 Apple Silicon GPU (MPS) Available: True")
    elif not cuda_available:
        print("🍎 Apple Silicon GPU (MPS): Not available")
    
    # CPU information
    cpu_count = os.cpu_count()
    print(f"🔧 CPU Cores: {cpu_count}")
    
    # Check transcription backends
    print("\n📦 TRANSCRIPTION BACKENDS:")
    
    # Check OpenAI Whisper
    try:
        import whisper
        print("✅ OpenAI Whisper: Available")
    except ImportError:
        print("❌ OpenAI Whisper: Not Available")
    
    # Check faster-whisper
    if FASTER_WHISPER_AVAILABLE:
        print("✅ Faster-Whisper: Available")
        
        # Check ONNX Runtime for GPU support
        if ONNXRUNTIME_AVAILABLE:
            providers = ort.get_available_providers()
            gpu_providers = [p for p in providers if 'CUDA' in p or 'Tensorrt' in p or 'Dml' in p]
            
            if gpu_providers:
                print(f"🚀 ONNX GPU Acceleration: Available ({', '.join(gpu_providers)})")
                if cuda_available:
                    print("🎯 Recommended: Faster-Whisper with GPU acceleration")
            else:
                print("⚠️  ONNX GPU Acceleration: Not Available (CPU only)")
        else:
            print("❌ ONNX Runtime: Not Available")
    else:
        print("❌ Faster-Whisper: Not Available")
    
    # Provide recommendations
    print("\n💡 RECOMMENDATIONS:")
    if cuda_available:
        print("• Use Faster-Whisper with GPU acceleration for best performance")
        print("• Install onnxruntime-gpu for optimal GPU support")
    elif mps_available:
        print("• Use OpenAI Whisper with MPS for Apple Silicon acceleration")
        print("• Faster-Whisper may also work but with CPU fallback")
    else:
        print("• CPU processing will be used (slower but functional)")
        print("• Consider using smaller models (tiny/base) for faster processing")
        print("• For best performance, use a system with NVIDIA GPU")
    
    print("=" * 60)
    return cuda_available or mps_available

try:
    from gui import main
    
    if __name__ == "__main__":
        # Ensure default directories exist
        audio_dir = current_dir / "audio_files"
        output_dir = current_dir / "outputs"
        icons_dir = current_dir / "icons"
        
        audio_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        icons_dir.mkdir(exist_ok=True)
        
        # Create a sample README in audio_files directory
        readme_path = audio_dir / "README.txt"
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("""Audio Files Directory
====================

Place your audio files here for transcription.

Supported formats:
- MP3, WAV, M4A, FLAC, AAC, OGG, WMA
- MP4, AVI, MOV, MKV (audio will be extracted)

The application will automatically scan this directory and subdirectories
for supported audio files when you select it as the input folder.

Example usage:
1. Copy your audio files to this folder
2. Select this folder as the input folder in the application
3. Choose an output folder for the transcriptions
4. Select a Whisper model size
5. Click "Start Transcription"

The application will create:
- Individual .txt files for each transcription
- A combined HTML report with links to the original audio files
""")
        
        print("🎙️  Starting Audio Transcriber Application...")
        print(f"📁 Default audio directory: {audio_dir}")
        print(f"📂 Default output directory: {output_dir}")
        
        # Detect and log GPU/hardware information
        gpu_available = detect_and_log_gpu()
        
        # Launch the GUI application
        main()
        
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("\nPlease install the required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting application: {e}")
    sys.exit(1) 