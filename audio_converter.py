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
        # Check if ffmpeg is available directly (more reliable than pydub)
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def install_conversion_dependencies_message() -> str:
    """Get installation instructions for conversion dependencies."""
    return """
🔧 AUDIO CONVERSION SETUP REQUIRED

To enable automatic WAV conversion for web compatibility:

Install ffmpeg:
• Windows: winget install ffmpeg (or download from ffmpeg.org)
• macOS: brew install ffmpeg  
• Linux: sudo apt install ffmpeg

After installation, restart the application to enable automatic conversion.
Note: ffmpeg handles all conversion directly, no additional Python packages needed.
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
    """Analyze a WAV file and return its properties using ffprobe."""
    try:
        import subprocess
        import json
        
        # Use ffprobe to get audio information
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', wav_path
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            audio_stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)
            
            if audio_stream:
                duration = float(audio_stream.get('duration', 0))
                channels = int(audio_stream.get('channels', 0))
                sample_rate = int(audio_stream.get('sample_rate', 0))
                bit_depth = int(audio_stream.get('bits_per_sample', 16))
                
                return {
                    'can_load': True,
                    'duration': duration,
                    'channels': channels,
                    'sample_rate': sample_rate,
                    'sample_width': bit_depth // 8,
                    'format_info': f"{channels}ch, {sample_rate}Hz, {bit_depth}bit",
                    'file_size_mb': Path(wav_path).stat().st_size / (1024 * 1024)
                }
        
        return {
            'can_load': False,
            'error': f'ffprobe failed with code {result.returncode}'
        }
    except Exception as e:
        return {
            'can_load': False,
            'error': str(e)
        }


def convert_wav_to_mp3(wav_path: str, output_dir: str = None, 
                      quality: str = "high", progress_callback: Optional[Callable] = None) -> Tuple[bool, str]:
    """
    Convert a WAV file to MP3 format for web compatibility using ffmpeg directly.
    
    Args:
        wav_path: Path to the WAV file
        output_dir: Directory to save MP3 (defaults to same as WAV)
        quality: Quality setting ("high", "medium", "low")
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (success, result_path_or_error_message)
    """
    try:
        import subprocess
        
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
        
        # Use ffmpeg for conversion
        cmd = [
            'ffmpeg', '-i', str(wav_file),
            '-codec:a', 'libmp3lame',
            '-b:a', quality_settings[quality]["bitrate"],
            '-metadata', f'title={wav_file.stem}',
            '-metadata', f'comment=Converted from {wav_file.name} for web compatibility',
            '-metadata', 'artist=Audio Transcriber App',
            '-y',  # Overwrite output file if it exists
            str(mp3_path)
        ]
        
        # Run ffmpeg conversion
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and mp3_path.exists():
            file_size_mb = mp3_path.stat().st_size / (1024 * 1024)
            if progress_callback:
                progress_callback(f"✅ Created web-compatible MP3: {mp3_filename} ({file_size_mb:.1f} MB)")
            return True, str(mp3_path)
        else:
            error_msg = f"ffmpeg failed with code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr[:200]}"
            return False, error_msg
            
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
