#!/usr/bin/env python3
"""
Build script for Audio Transcriber Application
This script automates the process of building the application into an executable.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"🔨 {text}")
    print("=" * 60)

def print_step(step_num, text):
    """Print a formatted step."""
    print(f"\n📋 Step {step_num}: {text}")

def run_command(command, description=""):
    """Run a command and handle errors."""
    print(f"🚀 Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error {description}: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_dependencies():
    """Check if all required dependencies are available."""
    print_step(1, "Checking dependencies")
    
    try:
        import PyInstaller
        print(f"✅ PyInstaller {PyInstaller.__version__} is available")
    except ImportError:
        print("❌ PyInstaller not found. Installing...")
        if not run_command("pip install pyinstaller", "installing PyInstaller"):
            return False
    
    # Check main dependencies
    dependencies = ["whisper", "torch", "PySide6", "numpy", "ffmpeg"]
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} is available")
        except ImportError:
            print(f"❌ {dep} not found. Please install requirements.txt first")
            return False
    
    return True

def clean_build_directories():
    """Clean previous build directories."""
    print_step(2, "Cleaning previous build directories")
    
    dirs_to_clean = ["build", "dist", "__pycache__"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"🗑️  Removing {dir_name}/")
            shutil.rmtree(dir_name)
        
    # Also clean any .spec files that might have been auto-generated
    for spec_file in Path(".").glob("*.spec"):
        if spec_file.name != "transcriber_app.spec":
            print(f"🗑️  Removing {spec_file}")
            spec_file.unlink()

def create_icon():
    """Create a basic icon if none exists."""
    print_step(3, "Checking for application icon")
    
    icons_dir = Path("icons")
    icons_dir.mkdir(exist_ok=True)
    
    icon_path = icons_dir / "app_icon.ico"
    if not icon_path.exists():
        print("📎 No icon found, creating a basic one...")
        # For now, we'll skip icon creation as it requires additional dependencies
        # The app will build without an icon
        print("💡 Building without custom icon (you can add icons/app_icon.ico later)")

def build_executable():
    """Build the executable using PyInstaller."""
    print_step(4, "Building executable with PyInstaller")
    
    if not os.path.exists("transcriber_app.spec"):
        print("❌ transcriber_app.spec file not found!")
        return False
    
    # Run PyInstaller with the spec file
    command = "pyinstaller transcriber_app.spec --clean --noconfirm"
    return run_command(command, "building executable")

def create_distribution():
    """Create a distribution package."""
    print_step(5, "Creating distribution package")
    
    dist_dir = Path("dist/AudioTranscriber")
    if not dist_dir.exists():
        print("❌ Distribution directory not found!")
        return False
    
    # Copy additional files to the distribution
    files_to_copy = [
        ("README.md", "README.md"),
        ("requirements.txt", "requirements.txt"),
    ]
    
    for src, dst in files_to_copy:
        if os.path.exists(src):
            dst_path = dist_dir / dst
            shutil.copy2(src, dst_path)
            print(f"📄 Copied {src} to distribution")
    
    # Create audio_files and outputs directories
    (dist_dir / "audio_files").mkdir(exist_ok=True)
    (dist_dir / "outputs").mkdir(exist_ok=True)
    print("📁 Created audio_files and outputs directories")
    
    # Create a launcher script
    launcher_script = dist_dir / "run_transcriber.bat"
    with open(launcher_script, 'w') as f:
        f.write("""@echo off
echo Starting Audio Transcriber...
AudioTranscriber.exe
pause
""")
    print("🚀 Created launcher script")
    
    return True

def show_completion_info():
    """Show information about the completed build."""
    print_header("BUILD COMPLETED SUCCESSFULLY!")
    
    dist_dir = Path("dist/AudioTranscriber")
    exe_path = dist_dir / "AudioTranscriber.exe"
    
    if exe_path.exists():
        exe_size = exe_path.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"📦 Executable created: {exe_path}")
        print(f"📏 Size: {exe_size:.1f} MB")
        print(f"📁 Distribution folder: {dist_dir}")
        
        print("\n🎯 Next steps:")
        print("1. Test the executable by running:")
        print(f"   {exe_path}")
        print("2. Copy the entire 'dist/AudioTranscriber' folder to share the app")
        print("3. The app requires the entire folder structure to work properly")
        
        print("\n💡 Tips:")
        print("• Place audio files in the 'audio_files' folder")
        print("• Transcriptions will be saved in the 'outputs' folder")
        print("• Use 'run_transcriber.bat' for easier launching")
    else:
        print("❌ Executable not found! Check the build output for errors.")

def main():
    """Main build process."""
    print_header("AUDIO TRANSCRIBER - EXECUTABLE BUILD")
    print("This script will build the Audio Transcriber into a standalone executable.")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Step 1: Check dependencies
        if not check_dependencies():
            print("❌ Dependency check failed!")
            return 1
        
        # Step 2: Clean build directories
        clean_build_directories()
        
        # Step 3: Create icon
        create_icon()
        
        # Step 4: Build executable
        if not build_executable():
            print("❌ Build failed!")
            return 1
        
        # Step 5: Create distribution
        if not create_distribution():
            print("⚠️  Distribution creation failed, but executable should still work")
        
        # Show completion info
        show_completion_info()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Build cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 