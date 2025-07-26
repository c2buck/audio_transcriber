# Audio Transcriber - Whisper AI Desktop App

A modern desktop application built with PySide6 that allows you to transcribe audio files using OpenAI's Whisper AI model. Features a clean, professional interface with dark/light theme support and comprehensive batch processing capabilities.

## Features

### 🎯 Core Functionality
- **Batch Transcription**: Process multiple audio files simultaneously
- **Multiple Formats**: Support for MP3, WAV, M4A, FLAC, AAC, OGG, WMA, MP4, AVI, MOV, MKV
- **Model Selection**: Choose from 5 Whisper model sizes (tiny, base, small, medium, large)
- **Smart Device Detection**: Automatically uses GPU if available, falls back to CPU

### 🎨 User Interface
- **Modern Design**: Clean, professional PySide6 interface
- **Dark/Light Themes**: Toggle between themes for comfortable viewing
- **Real-time Progress**: Live progress bars and detailed logging
- **Responsive Layout**: Resizable panels and organized sections

### 📊 Output & Reports
- **HTML Reports**: Beautiful reports with hyperlinks to original audio files
- **Individual Files**: Separate .txt files for each transcription
- **Statistics**: Processing time, success/failure counts, file durations
- **Audio Playback**: Direct links to play original files from HTML report

### 🔧 Advanced Features
- **Background Processing**: Non-blocking transcription with cancellation support
- **Settings Persistence**: Remembers your folder selections and preferences
- **Error Handling**: Robust error handling with detailed error messages
- **Status Updates**: Real-time status bar and device information

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (required for audio processing)

### Install FFmpeg
**Windows:**
```bash
# Using chocolatey
choco install ffmpeg

# Using winget
winget install FFmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg  # Ubuntu/Debian
sudo yum install ffmpeg  # CentOS/RHEL
```

### Install Python Dependencies
```bash
# Clone or download the project
cd transcriber_app

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. **Launch the Application**
   ```bash
   python main.py
   ```

2. **Select Input Folder**
   - Click "Browse..." next to "Input Folder"
   - Choose a folder containing your audio files
   - The app will automatically scan for supported formats

3. **Select Output Folder**
   - Click "Browse..." next to "Output Folder"
   - Choose where you want the transcriptions saved

4. **Choose Whisper Model**
   - Select from the dropdown: tiny, base, small, medium, or large
   - Larger models are more accurate but slower

5. **Start Transcription**
   - Click "Start Transcription"
   - Monitor progress in real-time
   - View logs for detailed information

6. **View Results**
   - Individual .txt files are saved for each audio file
   - Click "Open HTML Report" to view the comprehensive report
   - HTML report includes hyperlinks to play original audio files

## Whisper Model Comparison

| Model  | Size | Speed | Accuracy | Use Case |
|--------|------|-------|----------|----------|
| tiny   | 39MB | ~32x realtime | Lower | Quick drafts, testing |
| base   | 74MB | ~16x realtime | Good | General purpose |
| small  | 244MB | ~6x realtime | Better | Professional transcription |
| medium | 769MB | ~2x realtime | High | High-quality transcription |
| large  | 1550MB | ~1x realtime | Best | Maximum accuracy needed |

## Supported Audio Formats

### Audio Files
- MP3, WAV, M4A, FLAC, AAC, OGG, WMA

### Video Files (audio extraction)
- MP4, AVI, MOV, MKV

## Directory Structure

```
transcriber_app/
├── main.py              # Application entry point
├── gui.py               # PySide6 GUI interface
├── transcriber.py       # Core Whisper transcription logic
├── utils.py             # Helper functions and utilities
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── audio_files/        # Default input folder (with sample README)
├── outputs/            # Default output folder
└── icons/              # UI icons (optional)
```

## GPU Support

The application automatically detects and uses the best available compute device:

- **NVIDIA GPU**: CUDA support for faster processing
- **Apple Silicon**: MPS backend for M1/M2 Macs
- **CPU**: Fallback for universal compatibility

Device information is displayed in the "Device Information" section of the interface.

## HTML Report Features

The generated HTML report includes:
- **Statistics Dashboard**: Total files, success/failure counts, processing time
- **File Listings**: Each transcription with metadata
- **Audio Playback Links**: Click to play original files directly
- **Responsive Design**: Works on desktop and mobile browsers
- **Professional Styling**: Clean, modern appearance

## Troubleshooting

### Common Issues

**"No module named 'whisper'"**
```bash
pip install openai-whisper
```

**"FFmpeg not found"**
- Install FFmpeg using instructions above
- Ensure FFmpeg is in your system PATH

**"CUDA out of memory"**
- Try a smaller model (tiny, base, or small)
- Close other GPU-intensive applications
- The app will automatically fall back to CPU if needed

**"Permission denied" errors**
- Ensure you have write permissions to the output folder
- Try running as administrator (Windows) or with sudo (Linux/Mac)

### Performance Tips

1. **Use GPU**: Install CUDA for NVIDIA GPUs for significant speed improvements
2. **Choose Right Model**: Balance speed vs accuracy based on your needs
3. **Close Other Apps**: Free up system resources for better performance
4. **SSD Storage**: Use SSD drives for faster file I/O

## Development

### Project Structure
- `gui.py`: Complete PySide6 interface with threading support
- `transcriber.py`: Core Whisper transcription engine
- `utils.py`: Utility functions for file handling and HTML generation
- `main.py`: Application entry point with error handling

### Key Technologies
- **PySide6**: Modern Qt-based GUI framework
- **OpenAI Whisper**: State-of-the-art speech recognition
- **PyTorch**: Deep learning backend with GPU support
- **FFmpeg**: Audio/video processing pipeline

## License

This project is provided as-is for educational and personal use. Please respect the licenses of the underlying technologies:
- OpenAI Whisper: MIT License
- PySide6: LGPL/Commercial License
- PyTorch: BSD-style License

## Contributing

Feel free to submit issues and enhancement requests! This application was designed to be:
- User-friendly for non-technical users
- Extensible for developers
- Professional-grade for business use

---

**Enjoy transcribing your audio files with AI!** 🎵➡️📝 