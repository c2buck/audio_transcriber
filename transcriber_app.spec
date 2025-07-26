# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from pathlib import Path

# Get the current directory
current_dir = Path.cwd()

# Define the main application file
main_script = 'main.py'

# Hidden imports for dependencies that PyInstaller might miss
hidden_imports = [
    'whisper',
    'whisper.model',
    'whisper.audio',
    'whisper.decoding',
    'whisper.normalizers',
    'whisper.tokenizer',
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torchvision',
    'torchaudio',
    'PySide6.QtCore',
    'PySide6.QtGui', 
    'PySide6.QtWidgets',
    'numpy',
    'ffmpeg',
    'pathlib',
    'threading',
    'datetime',
    'webbrowser',
    'json',
    're',
    'tiktoken',
    'tiktoken_ext',
    'tiktoken_ext.openai_public',
    'regex',
    'ftfy',
    'more_itertools',
]

# Data files to include
datas = [
    # Include any icon files if they exist
    ('icons', 'icons') if os.path.exists('icons') else None,
    # Include README files
    ('README.md', '.') if os.path.exists('README.md') else None,
    ('audio_files/README.txt', 'audio_files') if os.path.exists('audio_files/README.txt') else None,
]

# Filter out None entries
datas = [d for d in datas if d is not None]

# Binary excludes to reduce size (optional, can be removed if needed)
excludes = [
    'matplotlib',
    'IPython',
    'jupyter',
    'notebook',
    'scipy',
    'pandas',
    'PIL',
    'tkinter',
    '_tkinter',
    'tcl',
    'tk',
]

a = Analysis(
    [main_script],
    pathex=[str(current_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remove duplicate files
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Create the standalone executable (onefile mode)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AudioTranscriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want to see console output for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icons/app_icon.ico' if os.path.exists('icons/app_icon.ico') else None,
) 