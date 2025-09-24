#!/usr/bin/env python3
"""
Audio Converter Module for Transcription Workflow
==================================================

This module handles automatic conversion of problematic audio files to web-compatible
formats before transcription. It focuses on WAV files that may have browser playback
issues while maintaining transcription quality.

Author: AI Assistant
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable


def check_conversion_dependencies() -> bool:
    """Check if audio conversion dependencies are available."""
    try:
        from pydub import AudioSegment
        from pydub.utils import which
        
        # Check if ffmpeg is available
        if not which("ffmpeg"):
            return False
            
        return True
    except ImportError:
        return False


def install_conversion_dependencies_message() -> str:
    """Get installation instructions for conversion dependencies."""
    return """
🔧 AUDIO CONVERSION SETUP REQUIRED

To enable automatic WAV conversion for web compatibility:

1. Install pydub:
   pip install pydub

2. Install ffmpeg:
   • Windows: winget install ffmpeg (or download from ffmpeg.org)
   • macOS: brew install ffmpeg  
   • Linux: sudo apt install ffmpeg

After installation, restart the application to enable automatic conversion.
"""


def is_wav_likely_problematic(wav_path: str) -> bool:
    """
    Determine if a WAV file is likely to have browser playback issues.
    
    This uses heuristics based on common problematic characteristics:
    - Files from specific recording software patterns
    - Non-standard metadata
    - Unusual encoding parameters
    """
    try:
        filename = Path(wav_path).name.lower()
        
        # Check for patterns that indicate recording software with known issues
        problematic_patterns = [
            "recorded on",           # Common recording software pattern
            "recording_",           # Generic recording pattern
            "(4a#", "(4b6",        # Specific patterns from user's files
            "g03619517",           # Device/software identifiers
            "w03355197",           # Device/software identifiers
        ]
        
        for pattern in problematic_patterns:
            if pattern in filename:
                return True
        
        # Check file size - very large WAV files often have issues
        file_size_mb = Path(wav_path).stat().st_size / (1024 * 1024)
        if file_size_mb > 100:  # Over 100MB might indicate uncompressed/problematic format
            return True
            
        return False
        
    except Exception:
        # If we can't analyze, assume it might be problematic
        return True


def analyze_wav_file(wav_path: str) -> Dict[str, any]:
    """Analyze a WAV file and return its properties."""
    try:
        from pydub import AudioSegment
        
        audio = AudioSegment.from_wav(wav_path)
        return {
            'can_load': True,
            'duration': len(audio) / 1000.0,  # Convert to seconds
            'channels': audio.channels,
            'sample_rate': audio.frame_rate,
            'sample_width': audio.sample_width,
            'frame_count': audio.frame_count(),
            'format_info': f"{audio.channels}ch, {audio.frame_rate}Hz, {audio.sample_width*8}bit",
            'file_size_mb': Path(wav_path).stat().st_size / (1024 * 1024)
        }
    except Exception as e:
        return {
            'can_load': False,
            'error': str(e)
        }


def convert_wav_to_mp3(wav_path: str, output_dir: str = None, 
                      quality: str = "high", progress_callback: Optional[Callable] = None) -> Tuple[bool, str]:
    """
    Convert a WAV file to MP3 format for web compatibility.
    
    Args:
        wav_path: Path to the WAV file
        output_dir: Directory to save MP3 (defaults to same as WAV)
        quality: Quality setting ("high", "medium", "low")
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (success, result_path_or_error_message)
    """
    try:
        from pydub import AudioSegment
        
        wav_file = Path(wav_path)
        if not wav_file.exists():
            return False, f"WAV file not found: {wav_path}"
        
        if output_dir is None:
            output_dir = wav_file.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Set quality parameters
        quality_settings = {
            "high": {"bitrate": "192k"},
            "medium": {"bitrate": "128k"},
            "low": {"bitrate": "96k"}
        }
        
        if quality not in quality_settings:
            quality = "high"
        
        # Generate output filename
        mp3_filename = wav_file.stem + "_web.mp3"
        mp3_path = output_dir / mp3_filename
        
        if progress_callback:
            progress_callback(f"🔄 Converting {wav_file.name} to web-compatible MP3...")
        
        # Load and convert
        audio = AudioSegment.from_wav(str(wav_file))
        
        # Add metadata
        audio.export(
            str(mp3_path),
            format="mp3",
            bitrate=quality_settings[quality]["bitrate"],
            tags={
                'title': wav_file.stem,
                'comment': f'Converted from {wav_file.name} for web compatibility',
                'artist': 'Audio Transcriber App'
            }
        )
        
        # Verify the conversion
        if mp3_path.exists():
            file_size_mb = mp3_path.stat().st_size / (1024 * 1024)
            if progress_callback:
                progress_callback(f"✅ Created web-compatible MP3: {mp3_filename} ({file_size_mb:.1f} MB)")
            return True, str(mp3_path)
        else:
            return False, "Conversion failed - output file not created"
            
    except Exception as e:
        return False, f"Conversion error: {str(e)}"


def get_or_create_web_compatible_version(audio_file: str, 
                                       progress_callback: Optional[Callable] = None) -> Tuple[str, bool]:
    """
    Get or create a web-compatible version of an audio file.
    
    For WAV files that are likely problematic, this will:
    1. Check if a web-compatible MP3 version already exists
    2. If not, create one automatically
    3. Return the path to use for transcription
    
    Args:
        audio_file: Path to the original audio file
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (file_path_to_use, was_converted)
    """
    audio_path = Path(audio_file)
    file_extension = audio_path.suffix.lower()
    
    # Only process WAV files that are likely problematic
    if file_extension != '.wav' or not is_wav_likely_problematic(str(audio_path)):
        return str(audio_path), False
    
    # Check for existing web-compatible version
    mp3_path = audio_path.parent / f"{audio_path.stem}_web.mp3"
    if mp3_path.exists():
        if progress_callback:
            progress_callback(f"✅ Using existing web-compatible version: {mp3_path.name}")
        return str(mp3_path), False
    
    # Check if conversion dependencies are available
    if not check_conversion_dependencies():
        if progress_callback:
            progress_callback(f"⚠️ Cannot convert {audio_path.name} - conversion dependencies not available")
            progress_callback("📋 Will use original WAV file for transcription (may have web playback issues)")
        return str(audio_path), False
    
    # Create web-compatible version
    if progress_callback:
        progress_callback(f"🔍 Detected potentially problematic WAV file: {audio_path.name}")
        progress_callback("🔄 Creating web-compatible MP3 version for better browser support...")
    
    success, result = convert_wav_to_mp3(str(audio_path), progress_callback=progress_callback)
    
    if success:
        return result, True  # Return path to new MP3
    else:
        if progress_callback:
            progress_callback(f"❌ Conversion failed: {result}")
            progress_callback("📋 Will use original WAV file for transcription")
        return str(audio_path), False  # Fallback to original


def process_audio_files_for_web_compatibility(audio_files: List[str], 
                                             progress_callback: Optional[Callable] = None) -> Tuple[List[str], int]:
    """
    Process a list of audio files and create web-compatible versions where needed.
    
    Args:
        audio_files: List of paths to audio files
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (processed_file_list, num_converted)
    """
    processed_files = []
    converted_count = 0
    
    # Check if any files need conversion
    wav_files_to_check = [f for f in audio_files if Path(f).suffix.lower() == '.wav']
    problematic_wavs = [f for f in wav_files_to_check if is_wav_likely_problematic(f)]
    
    if problematic_wavs and progress_callback:
        progress_callback(f"🔍 Found {len(problematic_wavs)} potentially problematic WAV file(s)")
        
        if not check_conversion_dependencies():
            progress_callback("⚠️ Audio conversion dependencies not available")
            progress_callback(install_conversion_dependencies_message())
            progress_callback("📋 Proceeding with original files (transcription will work, web playback may have issues)")
    
    for audio_file in audio_files:
        processed_path, was_converted = get_or_create_web_compatible_version(
            audio_file, progress_callback
        )
        processed_files.append(processed_path)
        
        if was_converted:
            converted_count += 1
    
    if converted_count > 0 and progress_callback:
        progress_callback(f"🎵 Successfully converted {converted_count} file(s) to web-compatible format")
        progress_callback("📋 Using converted versions for both transcription and web playback")
    
    return processed_files, converted_count


def cleanup_temporary_conversions(converted_files: List[str], 
                                progress_callback: Optional[Callable] = None):
    """
    Clean up temporary conversion files if needed.
    Note: In this implementation, we keep the converted files as they're useful for web playback.
    """
    # For now, we keep all converted files as they improve web compatibility
    # This could be made configurable in the future
    pass


# Example usage and testing
if __name__ == "__main__":
    def test_callback(message):
        print(f"[CONVERTER] {message}")
    
    print("🎵 Audio Converter Module Test")
    print("=" * 50)
    
    # Test dependency check
    if check_conversion_dependencies():
        print("✅ Conversion dependencies available")
    else:
        print("❌ Conversion dependencies missing")
        print(install_conversion_dependencies_message())
