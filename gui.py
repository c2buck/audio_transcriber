import os
import sys
import threading
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFileDialog, QComboBox,
    QTextEdit, QProgressBar, QGroupBox, QFrame, QSplitter,
    QMessageBox, QStatusBar, QMenuBar, QMenu, QCheckBox,
    QTabWidget, QScrollArea, QLineEdit, QSizePolicy
)
from PySide6.QtCore import QThread, Signal, QTimer, Qt
from PySide6.QtGui import QFont, QIcon, QAction, QPalette, QTextCursor

from transcriber import AudioTranscriber
from backend_manager import BackendManager
from utils import get_audio_files, create_html_report, format_time
from dv_review import DVWordListAnalyzer


class TranscriptionWorker(QThread):
    """Worker thread for running transcription in the background."""
    
    progress_update = Signal(str)
    file_progress = Signal(int, int)
    finished = Signal(dict)
    
    def __init__(self, transcriber: AudioTranscriber, input_dir: str, output_dir: str, create_zip: bool = True, filename_prefix: Optional[str] = None):
        super().__init__()
        self.transcriber = transcriber
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.create_zip = create_zip
        self.filename_prefix = filename_prefix
        self.is_cancelled = False
    
    def run(self):
        """Run the transcription process."""
        def progress_callback(message):
            if not self.is_cancelled:
                self.progress_update.emit(message)
        
        def file_progress_callback(current, total):
            if not self.is_cancelled:
                self.file_progress.emit(current, total)
        
        def cancellation_check():
            """Check if cancellation has been requested."""
            return self.is_cancelled
        
        try:
            result = self.transcriber.transcribe_batch(
                self.input_dir,
                self.output_dir,
                progress_callback,
                file_progress_callback,
                self.create_zip,
                cancellation_check,
                self.filename_prefix
            )
            
            if not self.is_cancelled:
                self.finished.emit(result)
            else:
                # Emit cancelled result
                result['cancelled'] = True
                self.finished.emit(result)
        except Exception as e:
            if not self.is_cancelled:
                self.finished.emit({
                    'success': False,
                    'error': str(e),
                    'results': [],
                    'total_time': 0,
                    'success_count': 0,
                    'failure_count': 0,
                    'zip_path': None,
                    'cancelled': False
                })
            else:
                # Cancelled during exception
                self.finished.emit({
                    'success': False,
                    'error': 'Transcription cancelled',
                    'results': [],
                    'total_time': 0,
                    'success_count': 0,
                    'failure_count': 0,
                    'zip_path': None,
                    'cancelled': True
                })
    
    def cancel(self):
        """Cancel the transcription process."""
        self.is_cancelled = True


class AudioTranscriberGUI(QMainWindow):
    """Main GUI application for the Audio Transcriber."""
    
    def __init__(self):
        super().__init__()
        
        self.transcriber = None
        self.worker_thread = None
        self.backend_manager = BackendManager()
        
        # Initialize DV Review Analyzer
        self.dv_analyzer = DVWordListAnalyzer()
        self.dv_analysis_results = None
        
        # Wordlist feature toggle state (default to False - disabled)
        self.wordlist_enabled = False
        
        # Initialize UI
        self.init_ui()
        
        # Initialize backend and device selection with defaults
        self.populate_backend_combo()
        self.populate_device_combo()
        self.update_model_combo()
        self.update_device_info()
        self.update_status()
        
        # Set up timer for periodic updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(1000)  # Update every second
        
        # Log system information on startup
        self.log_system_info()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Audio Transcriber - Whisper AI")
        self.setMinimumSize(1200, 800)
        self.resize(1600, 1000)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create header
        self.create_header(main_layout)
        
        # Create tabbed interface
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Transcription Tab
        transcription_tab = self.create_transcription_tab()
        self.tab_widget.addTab(transcription_tab, "🎙️ Transcription")
        
        # Apply initial theme
        self.apply_theme()
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_input_action = QAction("Select Input Folder", self)
        open_input_action.triggered.connect(self.select_input_folder)
        file_menu.addAction(open_input_action)
        
        open_output_action = QAction("Select Output Folder", self)
        open_output_action.triggered.connect(self.select_output_folder)
        file_menu.addAction(open_output_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        self.theme_action = QAction("Toggle Dark/Light Theme", self)
        self.theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(self.theme_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_transcription_tab(self):
        """Create the transcription tab with existing functionality."""
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        
        # Create splitter for config and progress panels
        splitter = QSplitter(Qt.Horizontal)
        tab_layout.addWidget(splitter)
        
        # Left panel - Configuration
        config_widget = self.create_config_panel()
        splitter.addWidget(config_widget)
        
        # Right panel - Progress and logs
        progress_widget = self.create_progress_panel()
        splitter.addWidget(progress_widget)
        
        # Set splitter proportions (wider initial sizes to prevent squishing)
        splitter.setSizes([600, 800])
        
        return tab_widget
    
    def create_header(self, layout):
        """Create the application header."""
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.StyledPanel)
        header_layout = QVBoxLayout(header_frame)
        
        title_label = QLabel("Audio Transcriber")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)
        
        subtitle_label = QLabel("Powered by OpenAI Whisper & Faster-Whisper")
        subtitle_label.setFont(QFont("Arial", 12))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: gray;")
        header_layout.addWidget(subtitle_label)
        
        layout.addWidget(header_frame)
    
    def create_config_panel(self):
        """Create the configuration panel."""
        config_widget = QWidget()
        layout = QVBoxLayout(config_widget)
        
        # File Selection Group
        file_group = QGroupBox("File Selection")
        file_layout = QGridLayout(file_group)
        
        # Input folder
        file_layout.addWidget(QLabel("Input Folder:"), 0, 0)
        self.input_folder_label = QLabel("No folder selected")
        self.input_folder_label.setStyleSheet("background: #f0f0f0; padding: 8px; border: 1px solid #ccc; color: black;")
        self.input_folder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.input_folder_label.setMinimumWidth(200)
        file_layout.addWidget(self.input_folder_label, 0, 1)
        file_layout.setColumnStretch(1, 1)  # Make folder label column expandable
        
        self.select_input_btn = QPushButton("Browse...")
        self.select_input_btn.clicked.connect(self.select_input_folder)
        file_layout.addWidget(self.select_input_btn, 0, 2)
        
        # Output folder
        file_layout.addWidget(QLabel("Output Folder:"), 1, 0)
        self.output_folder_label = QLabel("No folder selected")
        self.output_folder_label.setStyleSheet("background: #f0f0f0; padding: 8px; border: 1px solid #ccc; color: black;")
        self.output_folder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.output_folder_label.setMinimumWidth(200)
        file_layout.addWidget(self.output_folder_label, 1, 1)
        
        self.select_output_btn = QPushButton("Browse...")
        self.select_output_btn.clicked.connect(self.select_output_folder)
        file_layout.addWidget(self.select_output_btn, 1, 2)
        
        layout.addWidget(file_group)
        
        # File Naming Group
        naming_group = QGroupBox("File Naming")
        naming_layout = QGridLayout(naming_group)
        
        # Subject name
        naming_layout.addWidget(QLabel("Subject Name:"), 0, 0)
        self.subject_name_input = QLineEdit()
        self.subject_name_input.setPlaceholderText("Enter subject name e.g. SMITH, John")
        self.subject_name_input.textChanged.connect(self.clear_subject_name_error)
        self.subject_name_input.setMinimumWidth(250)
        naming_layout.addWidget(self.subject_name_input, 0, 1)
        naming_layout.setColumnStretch(1, 1)  # Make input column expandable
        
        # Case No.
        naming_layout.addWidget(QLabel("Case No.:"), 1, 0)
        self.case_no_input = QLineEdit()
        self.case_no_input.setPlaceholderText("Enter case number e.g. CO25XXXXX, QP25XXXX")
        self.case_no_input.textChanged.connect(self.clear_case_no_error)
        self.case_no_input.setMinimumWidth(250)
        naming_layout.addWidget(self.case_no_input, 1, 1)
        
        layout.addWidget(naming_group)
        
        # Backend Selection Group
        backend_group = QGroupBox("Transcription Backend")
        backend_layout = QVBoxLayout(backend_group)
        
        # Backend selection
        backend_selection_layout = QHBoxLayout()
        backend_selection_layout.addWidget(QLabel("Backend:"))
        
        self.backend_combo = QComboBox()
        self.backend_combo.setMinimumWidth(200)
        self.backend_combo.currentTextChanged.connect(self.on_backend_changed)
        backend_selection_layout.addWidget(self.backend_combo)
        
        backend_layout.addLayout(backend_selection_layout)
        
        # Backend description
        self.backend_description = QLabel("Auto-select best available backend")
        self.backend_description.setWordWrap(True)
        self.backend_description.setStyleSheet("color: gray; font-size: 11px;")
        backend_layout.addWidget(self.backend_description)
        
        layout.addWidget(backend_group)
        
        # Model Selection Group
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout(model_group)
        
        # Model selection
        model_info_layout = QHBoxLayout()
        model_info_layout.addWidget(QLabel("Model:"))
        
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_info_layout.addWidget(self.model_combo)
        
        model_layout.addLayout(model_info_layout)
        
        # Beam size setting (for faster-whisper)
        self.beam_size_widget = QWidget()
        beam_layout = QHBoxLayout(self.beam_size_widget)
        beam_layout.setContentsMargins(0, 0, 0, 0)
        beam_layout.addWidget(QLabel("Beam Size:"))
        
        self.beam_size_combo = QComboBox()
        self.beam_size_combo.setMinimumWidth(100)
        self.beam_size_combo.addItems(["1", "3", "5", "7", "10"])
        self.beam_size_combo.setCurrentText("1")
        self.beam_size_combo.currentTextChanged.connect(self.on_beam_size_changed)
        beam_layout.addWidget(self.beam_size_combo)
        
        # Beam size help
        beam_help = QLabel("(Higher = better accuracy, slower)")
        beam_help.setStyleSheet("color: gray; font-size: 10px;")
        beam_layout.addWidget(beam_help)
        
        model_layout.addWidget(self.beam_size_widget)
        
        # Model description
        self.model_description = QLabel("")
        self.model_description.setWordWrap(True)
        self.model_description.setStyleSheet("color: gray; font-size: 11px;")
        model_layout.addWidget(self.model_description)
        
        layout.addWidget(model_group)
        
        # Language Selection Group
        language_group = QGroupBox("Language & Translation")
        language_layout = QVBoxLayout(language_group)
        
        # Language selection
        language_selection_layout = QHBoxLayout()
        language_selection_layout.addWidget(QLabel("Language:"))
        
        self.language_combo = QComboBox()
        self.language_combo.setMinimumWidth(200)
        self.language_combo.addItem("Auto Detect", None)
        self.language_combo.addItem("English", "en")
        self.language_combo.currentTextChanged.connect(self.on_language_changed)
        language_selection_layout.addWidget(self.language_combo)
        
        language_layout.addLayout(language_selection_layout)
        
        # Language description
        self.language_description = QLabel("Auto-detect language and transcribe in detected language")
        self.language_description.setWordWrap(True)
        self.language_description.setStyleSheet("color: gray; font-size: 11px;")
        language_layout.addWidget(self.language_description)
        
        layout.addWidget(language_group)
        
        # Device Info Group
        device_group = QGroupBox("Device Information")
        device_layout = QVBoxLayout(device_group)
        
        # Device selection
        device_selection_layout = QHBoxLayout()
        device_selection_layout.addWidget(QLabel("Device:"))
        
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(200)
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        device_selection_layout.addWidget(self.device_combo)
        
        device_layout.addLayout(device_selection_layout)
        
        # Device info display
        self.device_label = QLabel("Detecting available devices...")
        self.device_label.setStyleSheet("color: gray; font-size: 11px;")
        device_layout.addWidget(self.device_label)
        
        layout.addWidget(device_group)
        
        # Word List Detection Group
        wordlist_group = QGroupBox("Word List Detection")
        wordlist_layout = QVBoxLayout(wordlist_group)
        
        # Wordlist status label
        self.wordlist_status_label = QLabel("Disabled")
        self.wordlist_status_label.setStyleSheet("color: gray; font-size: 11px;")
        wordlist_layout.addWidget(self.wordlist_status_label)
        
        # Wordlist toggle button
        self.wordlist_toggle_btn = QPushButton("Enable Word List Detection")
        self.wordlist_toggle_btn.setCheckable(True)
        self.wordlist_toggle_btn.setChecked(False)
        self.wordlist_toggle_btn.clicked.connect(self.toggle_wordlist)
        self.wordlist_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                font-weight: bold;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:checked {
                background-color: #28a745;
            }
            QPushButton:checked:hover {
                background-color: #218838;
            }
        """)
        wordlist_layout.addWidget(self.wordlist_toggle_btn)
        
        # Wordlist description
        wordlist_desc = QLabel("Analyzes transcriptions for specific keywords and phrases")
        wordlist_desc.setStyleSheet("color: gray; font-size: 10px;")
        wordlist_desc.setWordWrap(True)
        wordlist_layout.addWidget(wordlist_desc)
        
        layout.addWidget(wordlist_group)
        
        # Control Buttons
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Transcription")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                padding: 12px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.start_btn.clicked.connect(self.start_transcription)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-weight: bold;
                padding: 12px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_transcription)
        control_layout.addWidget(self.stop_btn)
        
        layout.addLayout(control_layout)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return config_widget
    
    def create_progress_panel(self):
        """Create the progress and logs panel."""
        progress_widget = QWidget()
        layout = QVBoxLayout(progress_widget)
        
        # Progress Group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Overall progress
        progress_layout.addWidget(QLabel("Overall Progress:"))
        self.overall_progress = QProgressBar()
        self.overall_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 6px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 5px;
            }
        """)
        progress_layout.addWidget(self.overall_progress)
        
        # File progress info
        self.file_progress_label = QLabel("Ready to start...")
        progress_layout.addWidget(self.file_progress_label)
        
        layout.addWidget(progress_group)
        
        # Logs Group
        logs_group = QGroupBox("Transcription Logs")
        logs_layout = QVBoxLayout(logs_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        logs_layout.addWidget(self.log_text)
        
        # Clear logs button
        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self.clear_logs)
        logs_layout.addWidget(clear_btn)
        
        layout.addWidget(logs_group)
        
        # Results Group
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_label = QLabel("No transcription completed yet.")
        results_layout.addWidget(self.results_label)
        
        # Create a horizontal layout for buttons
        buttons_layout = QHBoxLayout()
        
        self.open_html_btn = QPushButton("Open HTML Report")
        self.open_html_btn.setEnabled(False)
        self.open_html_btn.clicked.connect(self.open_html_report)
        buttons_layout.addWidget(self.open_html_btn)
        
        self.open_zip_btn = QPushButton("Open Results Package")
        self.open_zip_btn.setEnabled(False)
        self.open_zip_btn.clicked.connect(self.open_zip_file)
        self.open_zip_btn.setToolTip("Open the zip file containing HTML report and audio files")
        buttons_layout.addWidget(self.open_zip_btn)
        
        results_layout.addLayout(buttons_layout)
        
        layout.addWidget(results_group)
        
        return progress_widget
    
    def get_model_description(self, model_name: str, backend_name: str = "openai") -> str:
        """Get description for the selected model and backend."""
        if backend_name == "faster":
            descriptions = {
                "tiny": "Fastest model, lower accuracy (Faster-Whisper optimized)",
                "base": "Good balance of speed and accuracy (Faster-Whisper optimized)",
                "small": "Better accuracy, moderate speed (Faster-Whisper optimized)",
                "medium": "High accuracy, slower processing (Faster-Whisper optimized)",
                "large-v2": "Very high accuracy (Faster-Whisper large-v2)",
                "large-v3": "Best accuracy available (Faster-Whisper large-v3)"
            }
        else:  # openai
            descriptions = {
                "tiny": "Fastest model, lower accuracy (~32x realtime)",
                "base": "Good balance of speed and accuracy (~16x realtime)", 
                "small": "Better accuracy, moderate speed (~6x realtime)",
                "medium": "High accuracy, slower processing (~2x realtime)",
                "large": "Best accuracy, slowest processing (~1x realtime)"
            }
        return descriptions.get(model_name, "Unknown model")
    
    def get_backend_description(self, backend_name: str) -> str:
        """Get description for the selected backend."""
        backend_info = self.backend_manager.get_backend_info(backend_name)
        if backend_info:
            return backend_info.description
        else:
            return "Unknown backend"
    
    def populate_backend_combo(self):
        """Populate the backend selection combo box."""
        self.backend_combo.clear()
        
        # Add Auto option
        self.backend_combo.addItem("Auto (Recommended)", "auto")
        
        # Add available backends
        available_backends = self.backend_manager.get_available_backends()
        for backend in available_backends:
            self.backend_combo.addItem(backend.display_name, backend.name)
        
        # Default to faster-whisper if available, otherwise auto
        faster_index = -1
        for i in range(self.backend_combo.count()):
            if self.backend_combo.itemData(i) == "faster":
                faster_index = i
                break
        if faster_index >= 0:
            self.backend_combo.setCurrentIndex(faster_index)
        else:
            self.backend_combo.setCurrentIndex(0)  # Fallback to auto
    
    def update_model_combo(self):
        """Update model combo based on selected backend."""
        backend_name = self.backend_combo.currentData()
        if not backend_name:
            return
        
        self.model_combo.clear()
        
        if backend_name == "auto":
            # For auto mode, use OpenAI models as default display
            # The actual backend will be determined at runtime
            backend_info = self.backend_manager.get_backend_info('openai')
            models = backend_info.models if backend_info else ["tiny", "base", "small", "medium", "large"]
        else:
            backend_info = self.backend_manager.get_backend_info(backend_name)
            models = backend_info.models if backend_info else []
        
        self.model_combo.addItems(models)
        
        # Set default based on backend - default to large-v3 for faster-whisper
        if backend_name == "faster" or backend_name == "auto":
            # Default to large-v3 for faster-whisper
            if "large-v3" in models:
                self.model_combo.setCurrentText("large-v3")
            elif "base" in models:
                self.model_combo.setCurrentText("base")
            elif models:
                self.model_combo.setCurrentIndex(0)
        else:
            # For other backends, default to base
            if "base" in models:
                self.model_combo.setCurrentText("base")
            elif models:
                self.model_combo.setCurrentIndex(0)
        
        # Update beam size visibility
        self.update_beam_size_visibility()
        
        # Update model description
        self.update_model_description()
    
    def update_beam_size_visibility(self):
        """Show/hide beam size setting based on backend."""
        backend_name = self.backend_combo.currentData()
        
        # Show beam size only for faster-whisper backend
        is_faster_backend = backend_name == "faster"
        self.beam_size_widget.setVisible(is_faster_backend)
    
    def update_model_description(self):
        """Update the model description based on current selections."""
        model_name = self.model_combo.currentText()
        backend_name = self.backend_combo.currentData()
        
        if model_name:
            # Determine actual backend for description
            if backend_name == "auto":
                # For auto mode, try to predict which backend would be used
                try:
                    auto_backend = self.backend_manager.auto_select_backend()
                    description = self.get_model_description(model_name, auto_backend)
                    description += f" (Auto-selected: {auto_backend})"
                except Exception:
                    description = self.get_model_description(model_name, "openai")
            else:
                description = self.get_model_description(model_name, backend_name)
            
            self.model_description.setText(description)
    
    def on_backend_changed(self):
        """Handle backend selection change."""
        backend_name = self.backend_combo.currentData()
        backend_display = self.backend_combo.currentText()
        
        # Update backend description
        if backend_name == "auto":
            self.backend_description.setText("Auto-select best available backend based on hardware")
        else:
            description = self.get_backend_description(backend_name)
            self.backend_description.setText(description)
        
        # Update model combo and related UI
        self.update_model_combo()
        
        # Reset transcriber to use new backend
        self.transcriber = None
        
        self.log(f"Backend changed to: {backend_display}", "INFO")
    
    def on_model_changed(self):
        """Handle model selection change."""
        model_name = self.model_combo.currentText()
        backend_name = self.backend_combo.currentData()
        
        # Update model description
        self.update_model_description()
        
        # Reset transcriber to reload model
        self.transcriber = None
        
        self.log(f"Model changed to: {model_name}", "INFO")
    
    def on_beam_size_changed(self):
        """Handle beam size change."""
        beam_size = self.beam_size_combo.currentText()
        
        # Reset transcriber to use new beam size
        self.transcriber = None
        
        self.log(f"Beam size changed to: {beam_size}", "INFO")
    
    def on_language_changed(self):
        """Handle language selection change."""
        language = self.language_combo.currentData()
        language_display = self.language_combo.currentText()
        
        # Update language description
        if language is None:
            self.language_description.setText("Auto-detect language and transcribe in detected language")
        else:
            self.language_description.setText(f"Force {language_display} language transcription")
        
        # Reset transcriber to use new language setting
        self.transcriber = None
        
        self.log(f"Language setting changed to: {language_display}", "INFO")
    
    def on_device_changed(self):
        """Handle device selection change."""
        self.transcriber = None  # Reset transcriber to use new device
        self.update_device_info()
        
        # Log device change
        device_name = self.device_combo.currentText()
        device = self.device_combo.currentData()
        self.log(f"Processing device changed to: {device_name}", "INFO")
        
        # Provide device-specific guidance
        if device == "cuda":
            self.log("💡 GPU acceleration enabled - expect faster processing", "SUCCESS")
        elif device == "mps":
            self.log("💡 Apple Silicon acceleration enabled", "SUCCESS")
        else:
            self.log("💡 CPU processing selected - consider GPU for faster performance", "INFO")
    
    def get_available_devices(self):
        """Get list of available devices with their display names."""
        devices = [("cpu", "CPU")]
        
        try:
            # Check for CUDA GPU
            if torch.cuda.is_available():
                try:
                    # Test CUDA functionality with a simple operation
                    test_tensor = torch.tensor([1.0]).cuda()
                    del test_tensor  # Clean up
                    gpu_name = torch.cuda.get_device_name(0)
                    devices.append(("cuda", f"GPU - {gpu_name}"))
                except Exception:
                    # If there's any CUDA error, skip GPU and use CPU only
                    pass
        except Exception:
            # If CUDA check fails entirely, continue with CPU only
            pass
        
        try:
            # Check for Apple Silicon GPU (for Mac users)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                devices.append(("mps", "Apple Silicon GPU"))
        except Exception:
            pass
        
        return devices
    
    def populate_device_combo(self):
        """Populate the device selection combo box."""
        self.device_combo.clear()
        devices = self.get_available_devices()
        
        for device_id, display_name in devices:
            self.device_combo.addItem(display_name, device_id)
        
        # Default to GPU (cuda) if available, otherwise use auto-detection
        gpu_index = -1
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == "cuda":
                gpu_index = i
                break
        
        if gpu_index >= 0:
            # GPU available, select it by default
            self.device_combo.setCurrentIndex(gpu_index)
        else:
            # No GPU, use auto-detection
            try:
                temp_transcriber = AudioTranscriber()
                auto_device = temp_transcriber._get_device()
                for i in range(self.device_combo.count()):
                    if self.device_combo.itemData(i) == auto_device:
                        self.device_combo.setCurrentIndex(i)
                        break
            except Exception:
                # If auto-detection fails, just use the first device (CPU)
                if self.device_combo.count() > 0:
                    self.device_combo.setCurrentIndex(0)
    
    def update_device_info(self):
        """Update the device information display."""
        selected_device = self.device_combo.currentData()
        if not selected_device:
            self.device_label.setText("No device selected")
            return
        
        try:
            # Create a temporary transcriber to get device info
            temp_transcriber = AudioTranscriber(device=selected_device)
            device_info = temp_transcriber.get_device_info()
            
            # Build device info text
            info_parts = []
            if device_info['cuda_available'] and selected_device == 'cuda':
                info_parts.append("✓ CUDA acceleration enabled")
            elif selected_device == 'mps':
                info_parts.append("✓ Apple Silicon acceleration enabled")
            else:
                info_parts.append("Using CPU processing")
            
            if device_info['cuda_available'] and selected_device != 'cuda':
                info_parts.append("💡 GPU available but not selected")
            
            self.device_label.setText(" | ".join(info_parts))
        except Exception as e:
            self.device_label.setText(f"Error getting device info: {str(e)}")
    
    def select_input_folder(self):
        """Select input folder containing audio files."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", ""
        )
        if folder:
            self.input_folder_label.setText(folder)
            self.update_file_count()
    
    def select_output_folder(self):
        """Select output folder for transcriptions."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", ""
        )
        if folder:
            self.output_folder_label.setText(folder)
    
    def highlight_error(self, line_edit: QLineEdit):
        """Highlight a line edit field in red to indicate an error."""
        line_edit.setStyleSheet("background-color: #ffcccc; border: 2px solid #ff0000; padding: 5px;")
    
    def clear_subject_name_error(self):
        """Clear error highlighting from subject name field."""
        self.subject_name_input.setStyleSheet("")
    
    def clear_case_no_error(self):
        """Clear error highlighting from case no field."""
        self.case_no_input.setStyleSheet("")
    
    def update_file_count(self):
        """Update the count of audio files in the selected folder."""
        input_folder = self.input_folder_label.text()
        if input_folder and input_folder != "No folder selected":
            try:
                audio_files = get_audio_files(input_folder, sort_by_date=True, debug_log=True)
                if len(audio_files) > 0:
                    self.log(f"Found {len(audio_files)} audio files in selected folder", "SUCCESS")
                    # Log supported file types found
                    file_types = {}
                    for file in audio_files:
                        ext = Path(file).suffix.lower()
                        file_types[ext] = file_types.get(ext, 0) + 1
                    
                    type_summary = ", ".join([f"{ext}({count})" for ext, count in sorted(file_types.items())])
                    self.log(f"File types: {type_summary}", "INFO")
                else:
                    self.log("No supported audio files found in selected folder", "WARNING")
            except Exception as e:
                self.log(f"Error scanning folder: {e}", "ERROR")
    
    def start_transcription(self):
        """Start the transcription process."""
        input_folder = self.input_folder_label.text()
        output_folder = self.output_folder_label.text()
        
        if input_folder == "No folder selected":
            QMessageBox.warning(self, "Warning", "Please select an input folder.")
            return
        
        if output_folder == "No folder selected":
            QMessageBox.warning(self, "Warning", "Please select an output folder.")
            return
        
        # Validate Subject Name and Case No. fields
        subject_name = self.subject_name_input.text().strip()
        case_no = self.case_no_input.text().strip()
        
        has_error = False
        if not subject_name:
            self.highlight_error(self.subject_name_input)
            has_error = True
        
        if not case_no:
            self.highlight_error(self.case_no_input)
            has_error = True
        
        if has_error:
            QMessageBox.warning(
                self, 
                "Validation Error", 
                "Both Subject Name and Case No. are required to proceed.\n\n"
                "Please fill in both fields before starting transcription."
            )
            return
        
        # Create transcriber if needed
        if not self.transcriber:
            model_name = self.model_combo.currentText()
            selected_device = self.device_combo.currentData()
            backend_name = self.backend_combo.currentData()
            beam_size = int(self.beam_size_combo.currentText())
            
            # Get language setting
            language = self.language_combo.currentData()
            
            self.transcriber = AudioTranscriber(
                model_name=model_name,
                device=selected_device,
                backend=backend_name,
                beam_size=beam_size,
                language=language
            )
        
        # Reset progress
        self.overall_progress.setValue(0)
        self.file_progress_label.setText("Initializing...")
        self.log("=== TRANSCRIPTION SESSION STARTED ===", "SYSTEM")
        self.log(f"Input folder: {input_folder}", "INFO")
        self.log(f"Output folder: {output_folder}", "INFO")
        self.log(f"Backend: {self.backend_combo.currentText()}", "INFO")
        self.log(f"Model: {self.model_combo.currentText()}", "INFO")
        self.log(f"Processing device: {self.device_combo.currentText()}", "INFO")
        self.log(f"Language: {self.language_combo.currentText()}", "INFO")
        if self.backend_combo.currentData() == "faster":
            self.log(f"Beam size: {self.beam_size_combo.currentText()}", "INFO")
        
        # Update UI state
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.select_input_btn.setEnabled(False)
        self.select_output_btn.setEnabled(False)
        self.backend_combo.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.beam_size_combo.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.language_combo.setEnabled(False)
        
        # Get filename prefix from input fields (already validated above)
        filename_prefix = f"{subject_name} {case_no}"
        
        # Start worker thread
        self.worker_thread = TranscriptionWorker(self.transcriber, input_folder, output_folder, create_zip=True, filename_prefix=filename_prefix)
        self.worker_thread.progress_update.connect(self.log)
        self.worker_thread.file_progress.connect(self.update_file_progress)
        self.worker_thread.finished.connect(self.transcription_finished)
        self.worker_thread.start()
    
    def stop_transcription(self):
        """Stop the transcription process."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.log("Stopping transcription...", "WARNING")
            self.worker_thread.cancel()
            
            # Update UI immediately to show response
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stopping...")
            
            # Don't wait here - let the cancellation check in the transcription loop handle it
            # The finished signal will be emitted when cancellation is detected
            # This prevents the UI from freezing
    
    def update_file_progress(self, current: int, total: int):
        """Update file progress display."""
        # Calculate progress based on completed files, not current file
        # If we're on file 1 of 10, we've completed 0 files, so progress should be 0%
        # If we're on file 2 of 10, we've completed 1 file, so progress should be 10%
        # Only show 100% when we've actually completed all files
        if total > 0:
            # Progress is based on files completed (current - 1), not current file
            # But we want to show some progress for the current file being processed
            # So we'll show progress as (current - 1) / total, but cap it at 99% until all files are done
            completed = max(0, current - 1)
            progress = int((completed / total) * 100) if total > 0 else 0
            # Cap at 99% until transcription is actually finished
            progress = min(99, progress)
        else:
            progress = 0
        
        self.overall_progress.setValue(progress)
        self.file_progress_label.setText(f"Processing file {current} of {total}")
    
    def transcription_finished(self, result: Dict[str, Any]):
        """Handle transcription completion."""
        # Update UI state
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stop")
        self.select_input_btn.setEnabled(True)
        self.select_output_btn.setEnabled(True)
        self.backend_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.beam_size_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.language_combo.setEnabled(True)
        
        # Check if transcription was cancelled
        if result.get('cancelled', False):
            self.log("=== TRANSCRIPTION CANCELLED ===", "WARNING")
            self.file_progress_label.setText("Transcription cancelled")
            self.overall_progress.setValue(0)
            
            # Show partial results if any
            if result.get('results'):
                success_count = result.get('success_count', 0)
                failure_count = result.get('failure_count', 0)
                self.log(f"Partial results: {success_count} successful, {failure_count} failed", "INFO")
            
            self.worker_thread = None
            return
        
        if result['success']:
            self.overall_progress.setValue(100)
            self.file_progress_label.setText("Transcription completed!")
            
            # Update results
            total_time = format_time(result['total_time'])
            success_count = result['success_count']
            failure_count = result['failure_count']
            
            # Build results text with conversion info
            results_lines = [
                "Transcription Summary:",
                f"• Total files processed: {success_count + failure_count}",
                f"• Successful: {success_count}",
                f"• Failed: {failure_count}",
                f"• Total processing time: {total_time}"
            ]
            
            # Add conversion information if applicable
            conversion_info = result.get('conversion_info', {})
            if conversion_info.get('wav_files_converted', 0) > 0:
                results_lines.append(f"• WAV files converted to MP3: {conversion_info['wav_files_converted']}")
                results_lines.append("• Enhanced web playback compatibility")
            
            results_text = "\n".join(results_lines)
            
            self.results_label.setText(results_text)
            self.log("=== TRANSCRIPTION SESSION COMPLETED ===", "SYSTEM")
            self.log(f"Total files processed: {success_count + failure_count}", "INFO")
            self.log(f"Successfully transcribed: {success_count}", "SUCCESS")
            if failure_count > 0:
                self.log(f"Failed transcriptions: {failure_count}", "WARNING")
            self.log(f"Total processing time: {total_time}", "INFO")
            
            # Log conversion information if available
            conversion_info = result.get('conversion_info', {})
            if conversion_info.get('wav_files_converted', 0) > 0:
                self.log("🎵 AUDIO CONVERSION SUMMARY:", "SUCCESS")
                self.log(f"WAV files converted to web-compatible MP3: {conversion_info['wav_files_converted']}", "SUCCESS")
                self.log("Converted files provide better browser playback compatibility", "INFO")
            
            # Calculate average processing speed
            if result['total_time'] > 0 and success_count > 0:
                avg_time_per_file = result['total_time'] / (success_count + failure_count)
                self.log(f"Average time per file: {avg_time_per_file:.1f} seconds", "INFO")
            
            # Run DV analysis on transcriptions only if wordlist is enabled
            if self.wordlist_enabled:
                self.log("🔍 Starting Domestic Violence word list analysis...", "INFO")
                dv_analysis = self.dv_analyzer.analyze_batch(result['results'])
                self.dv_analysis_results = dv_analysis
                
                # Add DV scores to results
                dv_score_map = {a['filename']: a for a in dv_analysis['analyses']}
                for res in result['results']:
                    if res.get('success', False):
                        filename = Path(res.get('file_path', '')).name
                        if filename in dv_score_map:
                            res['dv_score'] = dv_score_map[filename]['total_score']
                            res['dv_match_count'] = dv_score_map[filename]['match_count']
                
                # Log DV analysis summary
                self.log(f"✅ DV Analysis complete: {dv_analysis['recordings_with_matches']} recordings with matches", "SUCCESS")
                if dv_analysis['top_10']:
                    self.log(f"⚠️  Top scoring recording: {dv_analysis['top_10'][0]['filename']} (Score: {dv_analysis['top_10'][0]['total_score']})", "WARNING")
                
                # Update results text with DV summary and individual scores
                if dv_analysis['recordings_with_matches'] > 0:
                    results_lines.append(f"• Recordings flagged for review: {dv_analysis['recordings_with_matches']}")
                    
                    # Add top 3 scores to summary
                    if dv_analysis['top_10']:
                        top_3 = dv_analysis['top_10'][:3]
                        results_lines.append("• Top scoring recordings:")
                        for i, rec in enumerate(top_3, 1):
                            filename = rec['filename'][:40] + "..." if len(rec['filename']) > 40 else rec['filename']
                            results_lines.append(f"  {i}. {filename} (Score: {rec['total_score']:.1f})")
                    
                    results_text = "\n".join(results_lines)
                    self.results_label.setText(results_text)
                
                # Create HTML report (with DV scores and highlighting)
                output_folder = self.output_folder_label.text()
                
                # Create progress callback wrapper for HTML report generation
                def html_progress_callback(message: str, percentage: int = None):
                    """Progress callback for HTML report generation."""
                    if percentage is not None:
                        self.log(f"[{percentage}%] {message}", "INFO")
                    else:
                        self.log(message, "INFO")
                
                # Get filename prefix from input fields
                subject_name = self.subject_name_input.text().strip()
                case_no = self.case_no_input.text().strip()
                filename_prefix = None
                if subject_name or case_no:
                    parts = []
                    if subject_name:
                        parts.append(subject_name)
                    if case_no:
                        parts.append(case_no)
                    filename_prefix = " ".join(parts)
                
                html_path = create_html_report(
                    result['results'], output_folder, result['total_time'],
                    success_count, failure_count, dv_analysis=self.dv_analysis_results,
                    progress_callback=html_progress_callback,
                    filename_prefix=filename_prefix
                )
                
                # Recreate zip file with updated HTML report that includes wordlist results
                if html_path and hasattr(self, 'transcriber') and self.transcriber:
                    input_folder = self.input_folder_label.text()
                    if input_folder and 'zip_path' in result and result['zip_path']:
                        self.log("Updating zip file with wordlist analysis results...", "INFO")
                        try:
                            # Recreate the zip file with the updated HTML
                            def zip_progress_callback(message: str, percentage: int = None):
                                """Progress callback for zip file creation."""
                                if percentage is not None:
                                    self.log(f"[{percentage}%] {message}", "INFO")
                                else:
                                    self.log(message, "INFO")
                            
                            # Get filename prefix from input fields
                            subject_name = self.subject_name_input.text().strip()
                            case_no = self.case_no_input.text().strip()
                            filename_prefix = None
                            if subject_name or case_no:
                                parts = []
                                if subject_name:
                                    parts.append(subject_name)
                                if case_no:
                                    parts.append(case_no)
                                filename_prefix = " ".join(parts)
                            
                            updated_zip_path = self.transcriber.create_results_zip(
                                input_folder, output_folder, 
                                progress_callback=zip_progress_callback,
                                filename_prefix=filename_prefix
                            )
                            if updated_zip_path:
                                result['zip_path'] = updated_zip_path
                                self.zip_file_path = updated_zip_path
                                self.log("✅ Zip file updated with wordlist analysis results", "SUCCESS")
                        except Exception as e:
                            self.log(f"Warning: Could not update zip file: {str(e)}", "WARNING")
            else:
                # Create HTML report without DV analysis
                output_folder = self.output_folder_label.text()
                
                # Create progress callback wrapper for HTML report generation
                def html_progress_callback(message: str, percentage: int = None):
                    """Progress callback for HTML report generation."""
                    if percentage is not None:
                        self.log(f"[{percentage}%] {message}", "INFO")
                    else:
                        self.log(message, "INFO")
                
                # Get filename prefix from input fields
                subject_name = self.subject_name_input.text().strip()
                case_no = self.case_no_input.text().strip()
                filename_prefix = None
                if subject_name or case_no:
                    parts = []
                    if subject_name:
                        parts.append(subject_name)
                    if case_no:
                        parts.append(case_no)
                    filename_prefix = " ".join(parts)
                
                html_path = create_html_report(
                    result['results'], output_folder, result['total_time'],
                    success_count, failure_count, dv_analysis=None,
                    progress_callback=html_progress_callback,
                    filename_prefix=filename_prefix
                )
            
            if html_path:
                self.html_report_path = html_path
                self.open_html_btn.setEnabled(True)
                self.log(f"HTML report created: {html_path}", "SUCCESS")
            else:
                self.log("Failed to create HTML report", "WARNING")
            
            # JSON transcript creation has been disabled
            
            # Handle zip file if created
            if hasattr(self, 'transcriber') and self.transcriber:
                if 'zip_path' in result and result['zip_path']:
                    self.zip_file_path = result['zip_path']
                    self.open_zip_btn.setEnabled(True)
                    self.log(f"Results package created: {Path(result['zip_path']).name}", "SUCCESS")
                    self.log("Package contains HTML report and audio files for sharing", "INFO")
        else:
            error_msg = result.get('error', 'Unknown error')
            self.log("=== TRANSCRIPTION SESSION FAILED ===", "SYSTEM")
            self.log(f"Error: {error_msg}", "ERROR")
            
            # Log additional diagnostic information if available
            if 'results' in result and result['results']:
                failed_files = [r for r in result['results'] if not r['success']]
                if failed_files:
                    self.log(f"Files that failed: {len(failed_files)}", "ERROR")
                    for failed_file in failed_files[:5]:  # Show first 5 failed files
                        file_name = Path(failed_file['file_path']).name
                        self.log(f"  └── {file_name}: {failed_file.get('error', 'Unknown error')}", "ERROR")
                    if len(failed_files) > 5:
                        self.log(f"  └── ... and {len(failed_files) - 5} more files", "ERROR")
            
            QMessageBox.critical(self, "Error", f"Transcription failed:\n{error_msg}")
        
        # Clean up worker thread
        self.worker_thread = None
    
    def open_html_report(self):
        """Open the HTML report in the default browser."""
        if hasattr(self, 'html_report_path') and self.html_report_path:
            import webbrowser
            webbrowser.open(f"file:///{self.html_report_path}")
    
    def open_zip_file(self):
        """Open the zip file or its containing folder."""
        if hasattr(self, 'zip_file_path') and self.zip_file_path:
            import os
            import platform
            import subprocess
            
            try:
                # Try to open the folder containing the zip file
                zip_folder = os.path.dirname(self.zip_file_path)
                
                if platform.system() == "Windows":
                    os.startfile(zip_folder)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.call(["open", zip_folder])
                else:  # Linux
                    subprocess.call(["xdg-open", zip_folder])
                    
                self.log(f"Opened folder containing results package: {zip_folder}", "INFO")
            except Exception as e:
                self.log(f"Error opening results package: {str(e)}", "ERROR")
    
    def log(self, message: str, level: str = "INFO"):
        """Add a message to the log display with enhanced formatting."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        
        # Color coding based on log level
        color_map = {
            "INFO": "#ffffff",      # White
            "SUCCESS": "#28a745",   # Green
            "WARNING": "#ffc107",   # Yellow
            "ERROR": "#dc3545",     # Red
            "DEBUG": "#6c757d",     # Gray
            "SYSTEM": "#17a2b8"     # Cyan
        }
        
        # Icon mapping for different log levels
        icon_map = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔍",
            "SYSTEM": "🖥️"
        }
        
        color = color_map.get(level, "#ffffff")
        icon = icon_map.get(level, "•")
        
        # Format message with HTML for colored output
        formatted_message = f'<span style="color: {color};">[{timestamp}] {icon} {level}: {message}</span>'
        
        # Append as HTML to preserve colors
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(formatted_message + "<br>")
        
        # Auto-scroll to bottom
        self.log_text.setTextCursor(cursor)
        
        # Also print to console for debugging
        print(f"[{timestamp}] {level}: {message}")
    
    def log_system_info(self):
        """Log comprehensive system information at startup."""
        import platform
        
        self.log("=== SYSTEM INFORMATION ===", "SYSTEM")
        self.log(f"Operating System: {platform.system()} {platform.release()}", "SYSTEM")
        self.log(f"Python Version: {platform.python_version()}", "SYSTEM")
        self.log(f"CPU: {platform.processor()}", "SYSTEM")
        self.log(f"CPU Cores: {os.cpu_count()}", "SYSTEM")
        
        # Try to get memory information with psutil
        try:
            import psutil
            memory = psutil.virtual_memory()
            self.log(f"Total RAM: {memory.total / (1024**3):.1f} GB", "SYSTEM")
            self.log(f"Available RAM: {memory.available / (1024**3):.1f} GB", "SYSTEM")
            self.log(f"RAM Usage: {memory.percent:.1f}%", "SYSTEM")
        except ImportError:
            self.log("Extended memory info unavailable (psutil not installed)", "DEBUG")
        except Exception as e:
            self.log(f"Error getting memory info: {str(e)}", "DEBUG")
        
        # Device information
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.log(f"CUDA GPUs Available: {gpu_count}", "SYSTEM")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                self.log(f"  └── GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)", "SYSTEM")
        else:
            self.log("No CUDA GPUs detected", "SYSTEM")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.log("Apple Silicon GPU (MPS) Available", "SYSTEM")
        
        # Backend detection information
        self.log("=== BACKEND DETECTION ===", "SYSTEM")
        detection_info = self.backend_manager.get_detection_info()
        
        for backend_name, backend_info in detection_info['backends_detected'].items():
            status = "✓ Available" if backend_info['available'] else "✗ Not Available"
            self.log(f"{backend_info['display_name']}: {status}", "SYSTEM")
            if backend_info['available']:
                models = ", ".join(backend_info['models'])
                self.log(f"  └── Models: {models}", "SYSTEM")
                self.log(f"  └── GPU Support: {'Yes' if backend_info['gpu_support'] else 'No'}", "SYSTEM")
        
        # ONNX Runtime information
        if detection_info['onnx_providers']:
            providers = ", ".join(detection_info['onnx_providers'])
            self.log(f"ONNX Providers: {providers}", "SYSTEM")
        else:
            self.log("ONNX Runtime: Not available", "SYSTEM")
        
        # Recommended backend
        recommended = detection_info['recommended_backend']
        if recommended and not recommended.startswith("Error"):
            self.log(f"Recommended Backend: {recommended}", "SYSTEM")
        
        # Compatibility warnings
        warnings = detection_info.get('compatibility_warnings', [])
        if warnings:
            self.log("=== COMPATIBILITY WARNINGS ===", "WARNING")
            for warning in warnings:
                self.log(warning, "WARNING")
        
        self.log("=== END BACKEND INFO ===", "SYSTEM")
    
    def clear_logs(self):
        """Clear the log display."""
        self.log_text.clear()
    
    def update_status(self):
        """Update the status bar."""
        # Update status bar
        if self.worker_thread and self.worker_thread.isRunning():
            self.status_bar.showMessage("Transcription in progress...")
        else:
            self.status_bar.showMessage("Ready")
    
    def toggle_theme(self):
        """Toggle between dark and light themes."""
        # Get current theme from stylesheet or default to light
        current_stylesheet = self.styleSheet()
        current_theme = "dark" if current_stylesheet and "background-color: #2b2b2b" in current_stylesheet else "light"
        new_theme = "dark" if current_theme == "light" else "light"
        self.apply_theme(new_theme)
    
    def toggle_wordlist(self):
        """Toggle wordlist analysis feature on/off."""
        self.wordlist_enabled = self.wordlist_toggle_btn.isChecked()
        
        if self.wordlist_enabled:
            self.wordlist_toggle_btn.setText("Disable Word List Detection")
            self.wordlist_status_label.setText("Enabled - Will analyze transcriptions")
            self.wordlist_status_label.setStyleSheet("color: #28a745; font-size: 11px; font-weight: bold;")
            self.log("Wordlist analysis enabled - will run on next transcription", "INFO")
        else:
            self.wordlist_toggle_btn.setText("Enable Word List Detection")
            self.wordlist_status_label.setText("Disabled")
            self.wordlist_status_label.setStyleSheet("color: gray; font-size: 11px;")
            self.log("Wordlist analysis disabled", "INFO")
    
    def apply_theme(self, theme="light"):
        """Apply the selected theme."""
        
        if theme == "dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #555555;
                    border-radius: 8px;
                    margin-top: 1ex;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
                QLabel {
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #404040;
                    border: 1px solid #555555;
                    padding: 8px;
                    border-radius: 4px;
                    color: #ffffff;
                }
                QPushButton:hover {
                    background-color: #505050;
                }
                QComboBox {
                    background-color: #404040;
                    border: 1px solid #555555;
                    padding: 5px;
                    border-radius: 4px;
                    color: #ffffff;
                }
                QTextEdit {
                    background-color: #1e1e1e;
                    border: 1px solid #555555;
                    color: #ffffff;
                }
            """)
        else:
            self.setStyleSheet("")
    
    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(self, "About Audio Transcriber", 
                         "Audio Transcriber v1.0\n\n"
                         "A modern desktop application for transcribing audio files "
                         "using OpenAI Whisper.\n\n"
                         "Features:\n"
                         "• Batch transcription of audio files\n"
                         "• Multiple Whisper model sizes\n"
                         "• Automatic GPU/CPU detection\n"
                         "• HTML report generation\n"
                         "• Dark/Light theme support")
    
    def closeEvent(self, event):
        """Handle application close event."""
        try:
            # Stop the timer first
            if hasattr(self, 'timer') and self.timer:
                self.timer.stop()
            
            # Check for running operations
            operations_running = []
            
            # Check transcription thread
            if hasattr(self, 'worker_thread') and self.worker_thread and self.worker_thread.isRunning():
                operations_running.append("transcription")
            
            if operations_running:
                operations_text = " and ".join(operations_running)
                reply = QMessageBox.question(self, "Confirm Exit", 
                                           f"{operations_text.title()} is in progress. Are you sure you want to exit?",
                                           QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    # Stop transcription
                    if hasattr(self, 'worker_thread') and self.worker_thread:
                        self.worker_thread.cancel()
                        # Wait for graceful shutdown
                        if not self.worker_thread.wait(3000):  # Wait up to 3 seconds
                            # Force terminate if it doesn't respond
                            self.worker_thread.terminate()
                            self.worker_thread.wait(2000)  # Wait for termination
                        self.worker_thread = None
                else:
                    event.ignore()
                    return
            
            # Clean up transcriber resources (release GPU memory, etc.)
            if hasattr(self, 'transcriber') and self.transcriber:
                try:
                    # Release model resources
                    if hasattr(self.transcriber, 'unified_transcriber') and self.transcriber.unified_transcriber:
                        if hasattr(self.transcriber.unified_transcriber, 'model') and self.transcriber.unified_transcriber.model:
                            # Delete model to free memory
                            del self.transcriber.unified_transcriber.model
                            self.transcriber.unified_transcriber.model = None
                            self.transcriber.unified_transcriber.is_model_loaded = False
                    
                    # Clear transcriber reference
                    self.transcriber = None
                except Exception as e:
                    print(f"Error cleaning up transcriber: {e}")
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            
            event.accept()
        except Exception as e:
            # If there's any error during close, log it but still allow closing
            print(f"Error during close event: {e}")
            event.accept()


def main():
    """Main application entry point."""
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Audio Transcriber")
        app.setApplicationVersion("1.0")
        
        # Set application icon (if available)
        try:
            app.setWindowIcon(QIcon("icons/app_icon.png"))
        except Exception:
            pass  # Icon not found, continue without it
        
        window = AudioTranscriberGUI()
        window.show()
        
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1) 