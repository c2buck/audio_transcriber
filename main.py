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

def detect_and_log_gpu():
    """Detect available computing devices and log information for user."""
    print("=" * 60)
    print("🔍 SYSTEM HARDWARE DETECTION")
    print("=" * 60)
    
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
        
        # Test GPU accessibility
        try:
            test_tensor = torch.tensor([1.0]).cuda()
            current_device = torch.cuda.current_device()
            print(f"✅ GPU {current_device} is ready for processing")
            print("🚀 Whisper will use GPU acceleration for faster transcription!")
        except Exception as e:
            print(f"⚠️  GPU detected but not accessible: {e}")
            print("🐌 Falling back to CPU processing")
    else:
        print("💾 No CUDA GPU detected")
    
    # Check for Apple Silicon (MPS)
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if mps_available:
        print("🍎 Apple Silicon GPU (MPS) Available: True")
        print("🚀 Whisper will use Apple Silicon acceleration!")
    elif not cuda_available:
        print("🍎 Apple Silicon GPU (MPS): Not available")
    
    # CPU information
    cpu_count = os.cpu_count()
    print(f"🔧 CPU Cores: {cpu_count}")
    
    if not cuda_available and not mps_available:
        print("🐌 Using CPU for processing (slower but still functional)")
        print("💡 Consider using a smaller model (tiny/base) for faster CPU processing")
    
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