#!/usr/bin/env python3
"""
WAV to MP3 Converter for Web Playback Compatibility
====================================================

This utility converts WAV files that have browser playback issues to MP3 format
for better web compatibility while preserving the original files.

Usage:
    python wav_to_mp3_converter.py <directory>
    python wav_to_mp3_converter.py <single_wav_file>

Requirements:
    pip install pydub

Author: AI Assistant
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        from pydub import AudioSegment
        from pydub.utils import which
        
        # Check if ffmpeg is available
        if not which("ffmpeg"):
            print("❌ Error: ffmpeg is required but not found in PATH")
            print("💡 Install ffmpeg:")
            print("   - Windows: Download from https://ffmpeg.org/ or use: winget install ffmpeg")
            print("   - macOS: brew install ffmpeg")
            print("   - Linux: sudo apt install ffmpeg")
            return False
            
        return True
    except ImportError:
        print("❌ Error: pydub is required but not installed")
        print("💡 Install with: pip install pydub")
        return False

def analyze_wav_file(wav_path: str) -> dict:
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
            'format_info': f"{audio.channels}ch, {audio.frame_rate}Hz, {audio.sample_width*8}bit"
        }
    except Exception as e:
        return {
            'can_load': False,
            'error': str(e)
        }

def convert_wav_to_mp3(wav_path: str, output_dir: str = None, quality: str = "high") -> Tuple[bool, str]:
    """
    Convert a WAV file to MP3 format.
    
    Args:
        wav_path: Path to the WAV file
        output_dir: Directory to save MP3 (defaults to same as WAV)
        quality: Quality setting ("high", "medium", "low")
        
    Returns:
        Tuple of (success, message/path)
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
        
        print(f"🔄 Converting: {wav_file.name} → {mp3_filename}")
        
        # Load and convert
        audio = AudioSegment.from_wav(str(wav_file))
        audio.export(
            str(mp3_path),
            format="mp3",
            bitrate=quality_settings[quality]["bitrate"],
            tags={
                'title': wav_file.stem,
                'comment': f'Converted from {wav_file.name} for web compatibility'
            }
        )
        
        # Verify the conversion
        if mp3_path.exists():
            file_size_mb = mp3_path.stat().st_size / (1024 * 1024)
            print(f"✅ Created: {mp3_path.name} ({file_size_mb:.1f} MB)")
            return True, str(mp3_path)
        else:
            return False, "Conversion failed - output file not created"
            
    except Exception as e:
        return False, f"Conversion error: {str(e)}"

def find_wav_files(directory: str) -> List[str]:
    """Find all WAV files in a directory and subdirectories."""
    wav_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"❌ Directory not found: {directory}")
        return []
    
    for wav_file in directory_path.rglob("*.wav"):
        wav_files.append(str(wav_file))
    
    for wav_file in directory_path.rglob("*.WAV"):
        wav_files.append(str(wav_file))
    
    return sorted(wav_files)

def main():
    """Main function to handle command line usage."""
    print("🎵 WAV to MP3 Converter for Web Playback Compatibility")
    print("=" * 60)
    
    if len(sys.argv) != 2:
        print("Usage: python wav_to_mp3_converter.py <directory_or_file>")
        print("\nExamples:")
        print("  python wav_to_mp3_converter.py ./audio_files")
        print("  python wav_to_mp3_converter.py my_recording.wav")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    target_path = sys.argv[1]
    target = Path(target_path)
    
    if not target.exists():
        print(f"❌ Path not found: {target_path}")
        sys.exit(1)
    
    wav_files = []
    
    if target.is_file() and target.suffix.lower() == '.wav':
        wav_files = [str(target)]
    elif target.is_dir():
        wav_files = find_wav_files(str(target))
    else:
        print(f"❌ Invalid target: {target_path} (must be WAV file or directory)")
        sys.exit(1)
    
    if not wav_files:
        print("❌ No WAV files found")
        sys.exit(1)
    
    print(f"📁 Found {len(wav_files)} WAV file(s)")
    print()
    
    successful_conversions = 0
    failed_conversions = 0
    
    for wav_file in wav_files:
        print(f"🔍 Analyzing: {Path(wav_file).name}")
        
        # Analyze the WAV file
        analysis = analyze_wav_file(wav_file)
        
        if analysis['can_load']:
            print(f"   ℹ️  Format: {analysis['format_info']}")
            print(f"   ⏱️  Duration: {analysis['duration']:.1f} seconds")
            
            # Convert to MP3
            success, result = convert_wav_to_mp3(wav_file, quality="high")
            
            if success:
                successful_conversions += 1
                print(f"   ✅ Conversion successful: {Path(result).name}")
            else:
                failed_conversions += 1
                print(f"   ❌ Conversion failed: {result}")
        else:
            failed_conversions += 1
            print(f"   ❌ Cannot analyze WAV file: {analysis['error']}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("📊 CONVERSION SUMMARY")
    print(f"✅ Successful: {successful_conversions}")
    print(f"❌ Failed: {failed_conversions}")
    print(f"📁 Total processed: {len(wav_files)}")
    
    if successful_conversions > 0:
        print()
        print("💡 Next steps:")
        print("1. Use the generated *_web.mp3 files in your transcription HTML reports")
        print("2. These MP3 files should have better browser compatibility")
        print("3. Keep original WAV files for transcription (Whisper works fine with them)")

if __name__ == "__main__":
    main()
