# Audio Transcriber - Whisper AI Desktop App

A modern desktop application built with PySide6 that allows you to transcribe audio files using OpenAI's Whisper AI model and analyze transcripts with local AI models. Features a clean, professional interface with dark/light theme support, comprehensive batch processing capabilities, and intelligent AI-powered transcript review.

## Features

### 🎯 Core Functionality
- **Batch Transcription**: Process multiple audio files simultaneously with real-time progress tracking
- **Multiple Formats**: Support for MP3, WAV, M4A, FLAC, AAC, OGG, WMA, MP4, AVI, MOV, MKV
- **Advanced Backend Management**: Automatic selection between OpenAI Whisper and faster-whisper with GPU optimization
- **Model Selection**: Choose from 5 Whisper model sizes (tiny, base, small, medium, large) with performance guidance
- **Smart Device Detection**: Automatically detects and uses GPU (CUDA/Apple Silicon) if available, falls back to CPU
- **Manual Device Selection**: Override automatic detection to choose specific processing device

### 🤖 AI Review & Analysis
- **Local AI Integration**: Built-in support for Ollama-powered AI models for transcript analysis
- **Segment-by-Segment Review**: Intelligent analysis of transcribed content with contextual understanding
- **Case-Specific Analysis**: Customizable analysis based on case facts and specific requirements
- **Dedicated Review Interface**: Separate tab for AI review functionality with comprehensive logging
- **Model Flexibility**: Support for various local AI models through Ollama integration
- **Privacy-First**: All AI analysis runs locally, ensuring data privacy and security

### 🚀 Enhanced Backend System
- **Dual Backend Support**: Seamless switching between OpenAI Whisper and faster-whisper
- **Automatic Optimization**: Intelligent backend selection based on available hardware
- **GPU Acceleration**: Enhanced ONNX Runtime GPU support for faster-whisper
- **Performance Monitoring**: Real-time backend performance tracking and optimization
- **Fallback Mechanisms**: Robust error handling with automatic backend switching

### 🎨 User Interface
- **Tabbed Interface**: Organized tabs for transcription and AI review functionality
- **Modern Design**: Clean, professional PySide6 interface with organized panels
- **Dark/Light Themes**: Toggle between themes with settings persistence
- **Real-time Progress**: Live progress bars, file counters, and detailed logging with color coding
- **Responsive Layout**: Resizable splitter panels and organized groupings
- **Status Updates**: Comprehensive status bar and device information display
- **Full Menu System**: Complete menu bar with file operations, view options, and help

### 📊 Output & Reports
- **Enhanced HTML Reports**: Beautiful reports with timestamped segments and interactive features
- **Audio Playback Links**: Direct links to play original files from HTML report
- **Segment Toggle**: Switch between full text view and timestamped segments
- **Individual Files**: Separate .txt files for each transcription with metadata
- **Statistics Dashboard**: Processing time, success/failure counts, performance metrics
- **Language Detection**: Automatic language identification and reporting

### 🔧 Advanced Features
- **Background Processing**: Non-blocking transcription with thread-based processing
- **Cancellation Support**: Stop transcription mid-process safely
- **Settings Persistence**: Remembers folder selections, theme preferences, and device choices
- **Comprehensive Error Handling**: Detailed error messages with troubleshooting suggestions
- **Memory Monitoring**: GPU memory usage tracking and optimization tips
- **Performance Metrics**: Real-time speed calculations and processing statistics
- **File Analysis**: Pre-processing analysis of batch size, duration, and file types
- **System Information**: Startup logging of hardware capabilities and system specs
- **AI Review Threading**: Non-blocking AI analysis with progress tracking

### 📈 Performance & Monitoring
- **Real-time Metrics**: Processing speed in realtime factor (e.g., 4x realtime)
- **Memory Usage**: GPU memory allocation tracking and warnings
- **Progress Tracking**: Detailed progress updates for long transcriptions
- **Batch Statistics**: Comprehensive reporting on batch processing results
- **Device Optimization**: Automatic selection of best available processing device
- **Performance Guidance**: Model selection recommendations based on hardware

### 🛠️ Technical Features
- **Threaded Architecture**: UI remains responsive during processing and AI analysis
- **Robust File Handling**: Recursive directory scanning with format validation
- **Safe Filename Generation**: Automatic handling of special characters in filenames
- **HTML Report Generation**: Advanced report creation with embedded styling and JavaScript
- **Settings Management**: QSettings-based configuration persistence
- **Cross-platform Compatibility**: Windows, macOS, and Linux support
- **Ollama Integration**: Native support for local AI model deployment and management
- **Backend Abstraction**: Unified interface for multiple transcription backends

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (required for audio processing)
- Ollama (optional, for AI review features)

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

### Install Ollama (Optional - for AI Review)
**Windows:**
Download and install from [ollama.ai](https://ollama.ai)

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Install Python Dependencies
```bash
# Clone or download the project
cd transcriber_app

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Transcription

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

4. **Choose Backend and Model**
   - The app automatically selects the best available backend (faster-whisper preferred)
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

### AI Review (Optional)

1. **Setup Ollama**
   - Ensure Ollama is installed and running
   - Download a compatible model (e.g., `ollama pull llama2`)

2. **Access AI Review Tab**
   - Switch to the "🤖 AI Review" tab in the application

3. **Load Transcript**
   - Select the combined transcript file from your transcription output
   - The app will parse segments for analysis

4. **Configure Analysis**
   - Enter case facts or specific analysis requirements
   - Select the AI model for analysis

5. **Start Review**
   - Click "Start AI Review" to begin analysis
   - Monitor progress as each segment is analyzed
   - View detailed logs and results

## Backend Comparison

| Backend | Speed | GPU Support | Memory Usage | Best For |
|---------|-------|-------------|--------------|----------|
| faster-whisper | ~3-5x faster | ONNX GPU acceleration | Lower | Production use, batch processing |
| OpenAI Whisper | Standard | CUDA/MPS | Higher | Development, compatibility |

## Whisper Model Comparison

| Model  | Size | Speed | Accuracy | Use Case |
|--------|------|-------|----------|----------|
| tiny   | 39MB | ~32x realtime | Lower | Quick drafts, testing |
| base   | 74MB | ~16x realtime | Good | General purpose |
| small  | 244MB | ~6x realtime | Better | Professional transcription |
| medium | 769MB | ~2x realtime | High | High-quality transcription |
| large  | 1550MB | ~1x realtime | Best | Maximum accuracy needed |

*Note: Speeds are approximate and depend on the selected backend and available hardware.*

## Supported Audio Formats

### Audio Files
- MP3, WAV, M4A, FLAC, AAC, OGG, WMA

### Video Files (audio extraction)
- MP4, AVI, MOV, MKV

## Directory Structure

```
transcriber_app/
├── main.py                 # Application entry point with hardware detection
├── gui.py                  # PySide6 GUI with tabbed interface (transcription + AI review)
├── transcriber.py          # Core Whisper transcription logic with backend management
├── backend_manager.py      # Advanced backend selection and management system
├── ai_review.py           # AI-powered transcript analysis using Ollama
├── ollama_client.py       # Client for local Ollama AI model communication
├── utils.py               # Helper functions and utilities
├── requirements.txt       # Python dependencies with enhanced backends
├── build_executable.py   # PyInstaller build script for distribution
├── transcriber_app.spec   # PyInstaller configuration
├── README.md             # This file
├── audio_files/          # Default input folder (with sample README)
├── outputs/              # Default output folder
└── icons/                # UI icons (optional)
```

## GPU Support

The application automatically detects and uses the best available compute device:

- **NVIDIA GPU**: CUDA support with ONNX GPU acceleration for faster-whisper
- **Apple Silicon**: MPS backend for M1/M2 Macs
- **CPU**: Fallback for universal compatibility

Device information is displayed in the "Device Information" section of the interface.

## HTML Report Features

The generated HTML report includes:
- **Statistics Dashboard**: Total files, success/failure counts, processing time with visual cards
- **Timestamped Segments**: Interactive view showing transcription segments with precise timing
- **Dual View Modes**: Toggle between full text and timestamped segments view
- **Audio Playback Links**: Direct links to play original files with system player
- **File Location Access**: JavaScript-powered file explorer integration
- **Professional Styling**: Modern responsive design with hover effects and animations
- **Interactive Elements**: Clickable buttons and smooth transitions
- **Metadata Display**: File duration, language detection, and processing statistics
- **Error Reporting**: Clear display of failed transcriptions with error details

## AI Review Features

The AI review system provides intelligent analysis of transcribed content:

### Core Capabilities
- **Local Processing**: All AI analysis runs on your machine via Ollama
- **Contextual Analysis**: AI considers case facts and specific requirements
- **Segment-by-Segment Review**: Detailed analysis of individual transcript segments
- **Flexible Models**: Support for various AI models (Llama, CodeLlama, Mistral, etc.)
- **Progress Tracking**: Real-time progress updates during analysis
- **Comprehensive Logging**: Detailed logs of AI interactions and results

### Setup Requirements
1. Install Ollama on your system
2. Download a compatible AI model (e.g., `ollama pull llama2`)
3. Ensure Ollama service is running (usually starts automatically)
4. Configure the model in the AI Review tab

### Usage Workflow
1. Complete transcription of your audio files
2. Switch to the AI Review tab
3. Load the combined transcript file
4. Enter relevant case facts or analysis criteria
5. Select your preferred AI model
6. Start the review process
7. Monitor progress and review results

## 🕵️ Enhanced AI Crime Investigation Features

The AI review feature has been specially enhanced for crime investigation scenarios. When analyzing transcripts, the system now provides:

### Crime-Relevant Section Detection
- **Intelligent Analysis**: AI specifically looks for content relevant to crime investigations including:
  - Direct evidence related to the crime
  - Witness statements and observations
  - Suspect behavior and statements
  - Timeline information
  - Physical evidence descriptions
  - Names, locations, and identifying details
  - Contradictions or inconsistencies
  - Suspicious activities

### Timestamp-Based Navigation
- **Precise Timing**: Each relevant section includes estimated timestamps
- **Direct Audio Links**: Click-able links that jump directly to specific moments in recordings
- **Time Ranges**: Clear start and end times for each relevant section

### Crime Investigation Report
- **Dedicated HTML Report**: Automatically generates a crime-focused investigation report
- **Audio Integration**: Direct links to play audio from specific timestamps
- **Organized by Recording**: Sections grouped by source recording for easy reference
- **Visual Highlighting**: Crime-relevant content clearly marked and formatted

### Output Files
When running AI analysis, three types of reports are generated:

1. **Individual Analysis Files** (`.ai.txt`): Detailed analysis for each recording
2. **Combined Summary** (`ai_review_summary_[timestamp].txt`): Complete analysis of all recordings
3. **Crime Investigation Report** (`crime_investigation_report_[timestamp].html`): 🆕 **Enhanced HTML report with:**
   - Quick navigation to relevant audio sections
   - Timestamp-based audio links
   - Visual crime-relevant content highlighting
   - Executive summary of findings

### Usage for Crime Investigation

1. **Load Transcript**: Select your combined transcript file
2. **Enter Case Facts**: Provide relevant case details in the text area
3. **Run Analysis**: AI will analyze all recordings for crime-relevant content
4. **Review Results**: Open the generated crime investigation report
5. **Navigate Audio**: Click timestamp links to jump to specific moments

The system automatically estimates timestamps and creates direct links to audio files, making it easy to quickly navigate to the most important sections of recordings during an investigation.

## Troubleshooting

### Built-in Diagnostics
The application includes comprehensive diagnostic features:
- **Automatic Error Detection**: Smart error categorization with specific solutions
- **Memory Monitoring**: Real-time GPU memory usage tracking and warnings
- **Device Validation**: Startup hardware detection and capability testing
- **File Analysis**: Pre-processing validation of audio files and formats
- **Progress Tracking**: Detailed logging of each processing step

### Common Issues

**"No module named 'whisper'" or "No module named 'faster_whisper'"**
```bash
pip install openai-whisper faster-whisper
```

**"FFmpeg not found"**
- Install FFmpeg using instructions above
- Ensure FFmpeg is in your system PATH
- The app will detect and report FFmpeg availability

**"CUDA out of memory"**
- The app automatically monitors GPU memory and provides warnings
- Try a smaller model (tiny, base, or small)
- Switch to faster-whisper backend for better memory management
- Close other GPU-intensive applications
- The app will automatically fall back to CPU if needed

**"Ollama connection failed"**
- Ensure Ollama is installed and running (`ollama serve`)
- Check that the Ollama service is accessible at `http://localhost:11434`
- Verify you have downloaded at least one AI model (`ollama list`)
- Check firewall settings if connection issues persist

**"Permission denied" errors**
- Ensure you have write permissions to the output folder
- Try running as administrator (Windows) or with sudo (Linux/Mac)
- The app validates folder permissions before processing

**Backend Selection Issues**
- The app automatically selects the best available backend
- Check the device information panel for backend status
- Review startup logs for backend detection details
- Manual backend override available in advanced settings

### Performance Tips

1. **Monitor Device Usage**: Check the device information panel for optimal settings
2. **Use GPU Acceleration**: The app automatically detects and recommends the best device
3. **Model Selection Guidance**: Built-in descriptions help choose the right model
4. **Memory Optimization**: Real-time memory monitoring prevents out-of-memory errors
5. **Batch Analysis**: Pre-processing analysis helps estimate processing time
6. **Background Processing**: UI remains responsive during long transcriptions

## Development

### Project Structure
- `main.py`: Application entry point with hardware detection and error handling
- `gui.py`: Complete PySide6 interface with threading, themes, and settings persistence
- `transcriber.py`: Core Whisper transcription engine with device management and performance monitoring
- `utils.py`: Utility functions for file handling, HTML report generation, and time formatting
- `build_executable.py`: PyInstaller build script for standalone executable creation
- `requirements.txt`: Python dependencies with version specifications
- `transcriber_app.spec`: PyInstaller configuration for building

### Key Technologies
- **PySide6**: Modern Qt-based GUI framework with dark/light theme support
- **OpenAI Whisper**: State-of-the-art speech recognition with multiple model sizes
- **faster-whisper**: Optimized Whisper implementation with ONNX acceleration
- **PyTorch**: Deep learning backend with CUDA, MPS, and CPU support
- **FFmpeg**: Audio/video processing pipeline for format support
- **QSettings**: Cross-platform settings persistence
- **Threading**: Background processing with cancellation support
- **Ollama**: Local AI model deployment and management platform
- **ONNX Runtime**: GPU-accelerated inference for faster processing

## License

This project is provided as-is for educational and personal use. Please respect the licenses of the underlying technologies:
- OpenAI Whisper: MIT License
- faster-whisper: MIT License
- PySide6: LGPL/Commercial License
- PyTorch: BSD-style License
- Ollama: MIT License

## Contributing

Feel free to submit issues and enhancement requests! This application was designed to be:
- User-friendly for non-technical users
- Extensible for developers
- Professional-grade for business use
- Privacy-focused with local AI processing

---

**Enjoy transcribing and analyzing your audio files with AI!** 🎵➡️📝🤖 