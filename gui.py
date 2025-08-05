import os
import sys
import threading
import torch
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFileDialog, QComboBox,
    QTextEdit, QProgressBar, QGroupBox, QFrame, QSplitter,
    QMessageBox, QStatusBar, QMenuBar, QMenu, QCheckBox,
    QTabWidget, QScrollArea
)
from PySide6.QtCore import QThread, Signal, QTimer, Qt, QSettings
from PySide6.QtGui import QFont, QIcon, QAction, QPalette, QTextCursor

from transcriber import AudioTranscriber
from backend_manager import BackendManager
from utils import get_audio_files, create_html_report, create_json_transcript, format_time
from ai_review import AIReviewManager, TranscriptSegment


class TranscriptionWorker(QThread):
    """Worker thread for running transcription in the background."""
    
    progress_update = Signal(str)
    file_progress = Signal(int, int)
    finished = Signal(dict)
    
    def __init__(self, transcriber: AudioTranscriber, input_dir: str, output_dir: str, create_zip: bool = True):
        super().__init__()
        self.transcriber = transcriber
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.create_zip = create_zip
        self.is_cancelled = False
    
    def run(self):
        """Run the transcription process."""
        def progress_callback(message):
            if not self.is_cancelled:
                self.progress_update.emit(message)
        
        def file_progress_callback(current, total):
            if not self.is_cancelled:
                self.file_progress.emit(current, total)
        
        try:
            result = self.transcriber.transcribe_batch(
                self.input_dir,
                self.output_dir,
                progress_callback,
                file_progress_callback,
                self.create_zip
            )
            
            if not self.is_cancelled:
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
                    'zip_path': None
                })
    
    def cancel(self):
        """Cancel the transcription process."""
        self.is_cancelled = True


class AIReviewWorker(QThread):
    """Worker thread for running AI review in the background with comprehensive logging."""
    
    progress_update = Signal(str, str)  # message, level
    segment_complete = Signal(dict, int, int)  # result, current, total
    finished = Signal(list)  # all results
    
    def __init__(self, ai_manager: AIReviewManager, segments, case_facts: str, model_name: str, log_callback=None):
        super().__init__()
        self.ai_manager = ai_manager
        self.segments = segments
        self.case_facts = case_facts
        self.model_name = model_name
        self.log_callback = log_callback
        self.is_cancelled = False
    
    def run(self):
        """Run the AI review process with comprehensive logging."""
        def progress_callback(message):
            """Internal progress callback that forwards to the log_callback."""
            try:
                if not self.is_cancelled:
                    # Parse log level from enhanced logging format
                    if "] DEBUG:" in message:
                        level = "DEBUG"
                        msg = message.split("] DEBUG:", 1)[1].strip()
                    elif "] ERROR:" in message:
                        level = "ERROR" 
                        msg = message.split("] ERROR:", 1)[1].strip()
                    elif "] INFO:" in message:
                        level = "INFO"
                        msg = message.split("] INFO:", 1)[1].strip()
                    elif "] WARNING:" in message:
                        level = "WARNING"
                        msg = message.split("] WARNING:", 1)[1].strip()
                    else:
                        level = "INFO"
                        msg = message
                    
                    # Emit both to the progress signal and direct logging
                    try:
                        self.progress_update.emit(msg, level)
                    except Exception as signal_error:
                        print(f"[SIGNAL_ERROR] {signal_error}: {msg}", file=sys.stderr)
                    
                    if self.log_callback:
                        try:
                            self.log_callback(msg, level)
                        except Exception as callback_error:
                            print(f"[CALLBACK_ERROR] {callback_error}: {msg}", file=sys.stderr)
            except Exception as e:
                print(f"[PROGRESS_CALLBACK_ERROR] {e}: {message}", file=sys.stderr)
        
        def segment_complete_callback(result, current, total):
            """Internal segment completion callback."""
            try:
                if not self.is_cancelled:
                    self.segment_complete.emit(result, current, total)
            except Exception as e:
                print(f"[SEGMENT_CALLBACK_ERROR] {e}: {current}/{total}", file=sys.stderr)
        
        try:
            # Validate inputs first
            if not self.segments:
                error_msg = "No segments provided for analysis"
                progress_callback(error_msg)
                self.finished.emit([])
                return
                
            if not self.case_facts or not self.case_facts.strip():
                error_msg = "No case facts provided for analysis"
                progress_callback(error_msg)
                self.finished.emit([])
                return
                
            if not self.model_name:
                error_msg = "No model name provided for analysis"
                progress_callback(error_msg)
                self.finished.emit([])
                return
            
            # Reset cancellation flag in the AI manager
            try:
                self.ai_manager.reset_cancellation(progress_callback)
            except Exception as e:
                progress_callback(f"Error resetting AI manager: {str(e)}")
                # Continue anyway
            
            # Start the comprehensive analysis
            progress_callback("Starting comprehensive AI analysis...")
            
            results = self.ai_manager.analyze_all_segments(
                segments=self.segments,
                case_facts=self.case_facts,
                model_name=self.model_name,
                progress_callback=progress_callback,
                segment_complete_callback=segment_complete_callback
            )
            
            if not self.is_cancelled:
                progress_callback("AI analysis workflow completed successfully")
                try:
                    self.finished.emit(results)
                except Exception as finish_error:
                    print(f"[FINISH_EMIT_ERROR] {finish_error}", file=sys.stderr)
            else:
                progress_callback("AI analysis was cancelled")
                try:
                    self.finished.emit([])
                except Exception as cancel_emit_error:
                    print(f"[CANCEL_EMIT_ERROR] {cancel_emit_error}", file=sys.stderr)
                
        except Exception as e:
            if not self.is_cancelled:
                error_msg = f"Error during AI review workflow: {str(e)}"
                try:
                    progress_callback(error_msg)
                except Exception as progress_error:
                    print(f"[PROGRESS_ERROR] {progress_error}: {error_msg}", file=sys.stderr)
                
                # Log additional error details
                import traceback
                try:
                    progress_callback(f"Full traceback: {traceback.format_exc()}")
                except Exception as traceback_error:
                    print(f"[TRACEBACK_ERROR] {traceback_error}", file=sys.stderr)
                    print(f"Full traceback: {traceback.format_exc()}", file=sys.stderr)
                
                try:
                    self.finished.emit([])
                except Exception as error_emit_error:
                    print(f"[ERROR_EMIT_ERROR] {error_emit_error}", file=sys.stderr)
    
    def cancel(self):
        """Cancel the AI review process with logging."""
        self.is_cancelled = True
        self.ai_manager.cancel_analysis()
        if self.log_callback:
            self.log_callback("AI review cancellation requested", "WARNING")


class AudioTranscriberGUI(QMainWindow):
    """Main GUI application for the Audio Transcriber."""
    
    def __init__(self):
        super().__init__()
        
        self.transcriber = None
        self.worker_thread = None
        self.ai_worker_thread = None
        self.ai_review_manager = None
        self.settings = QSettings("AudioTranscriber", "Settings")
        self.backend_manager = BackendManager()
        
        # Initialize AI Review Manager
        self.ai_review_manager = AIReviewManager()
        
        # Initialize UI
        self.init_ui()
        self.load_settings()
        
        # Initialize backend and device selection
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
        self.setMinimumSize(900, 700)
        self.resize(1200, 800)
        
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
        
        # AI Review Tab
        ai_review_tab = self.create_ai_review_tab()
        self.tab_widget.addTab(ai_review_tab, "🤖 AI Review")
        
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
        
        # Set splitter proportions
        splitter.setSizes([400, 500])
        
        return tab_widget
    
    def create_ai_review_tab(self):
        """Create the AI Review tab."""
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        tab_layout.setSpacing(15)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create main splitter for layout
        main_splitter = QSplitter(Qt.Horizontal)
        tab_layout.addWidget(main_splitter)
        
        # Left Panel - Configuration and Controls
        left_panel = self.create_ai_config_panel()
        main_splitter.addWidget(left_panel)
        
        # Right Panel - Results and Logs
        right_panel = self.create_ai_results_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 600])
        
        return tab_widget
    
    def create_ai_config_panel(self):
        """Create the AI Review configuration panel."""
        config_widget = QWidget()
        layout = QVBoxLayout(config_widget)
        
        # Ollama Connection Group
        connection_group = QGroupBox("Ollama Connection")
        connection_layout = QVBoxLayout(connection_group)
        
        # Connection status
        connection_status_layout = QHBoxLayout()
        self.ai_connection_label = QLabel("Not connected")
        self.ai_connection_label.setStyleSheet("color: red; font-weight: bold;")
        connection_status_layout.addWidget(QLabel("Status:"))
        connection_status_layout.addWidget(self.ai_connection_label)
        connection_status_layout.addStretch()
        
        self.test_connection_btn = QPushButton("Test Connection")
        self.test_connection_btn.clicked.connect(self.test_ollama_connection)
        connection_status_layout.addWidget(self.test_connection_btn)
        
        connection_layout.addLayout(connection_status_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("AI Model:"))
        self.ai_model_combo = QComboBox()
        self.ai_model_combo.addItem("mistral")  # Default model
        model_layout.addWidget(self.ai_model_combo)
        connection_layout.addLayout(model_layout)
        
        layout.addWidget(connection_group)
        
        # Transcript Source Group
        transcript_group = QGroupBox("Transcript Source")
        transcript_layout = QVBoxLayout(transcript_group)
        
        # Transcript file selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Transcript File:"))
        self.ai_transcript_label = QLabel("No file selected")
        self.ai_transcript_label.setStyleSheet("background: #f0f0f0; padding: 8px; border: 1px solid #ccc; color: black;")
        file_layout.addWidget(self.ai_transcript_label, 1)
        
        self.select_transcript_btn = QPushButton("Browse...")
        self.select_transcript_btn.clicked.connect(self.select_transcript_file)
        file_layout.addWidget(self.select_transcript_btn)
        
        transcript_layout.addLayout(file_layout)
        
        # Segment info
        self.segment_info_label = QLabel("No transcript loaded")
        self.segment_info_label.setStyleSheet("color: gray; font-size: 11px;")
        transcript_layout.addWidget(self.segment_info_label)
        
        layout.addWidget(transcript_group)
        
        # Case Facts Group
        case_facts_group = QGroupBox("Case Facts")
        case_facts_layout = QVBoxLayout(case_facts_group)
        
        self.case_facts_text = QTextEdit()
        self.case_facts_text.setPlaceholderText("Enter the case facts that the AI should look for in the transcripts...")
        self.case_facts_text.setMaximumHeight(150)
        self.case_facts_text.textChanged.connect(self.update_analyze_button_state)
        case_facts_layout.addWidget(self.case_facts_text)
        
        layout.addWidget(case_facts_group)
        
        # Save Options Group
        save_group = QGroupBox("Save Options")
        save_layout = QVBoxLayout(save_group)
        
        self.save_individual_checkbox = QCheckBox("Save individual .ai.txt files per recording")
        self.save_individual_checkbox.setChecked(True)
        save_layout.addWidget(self.save_individual_checkbox)
        
        self.save_combined_checkbox = QCheckBox("Save combined summary file")
        self.save_combined_checkbox.setChecked(True)
        save_layout.addWidget(self.save_combined_checkbox)
        
        layout.addWidget(save_group)
        
        # Control Buttons
        control_layout = QHBoxLayout()
        
        self.analyze_btn = QPushButton("Analyze All Recordings")
        self.analyze_btn.setStyleSheet("""
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
        self.analyze_btn.clicked.connect(self.start_ai_analysis)
        self.analyze_btn.setEnabled(False)
        control_layout.addWidget(self.analyze_btn)
        
        self.stop_ai_btn = QPushButton("Stop Analysis")
        self.stop_ai_btn.setEnabled(False)
        self.stop_ai_btn.setStyleSheet("""
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
        self.stop_ai_btn.clicked.connect(self.stop_ai_analysis)
        control_layout.addWidget(self.stop_ai_btn)
        
        layout.addLayout(control_layout)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return config_widget
    
    def create_ai_results_panel(self):
        """Create the AI Review results panel."""
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        
        # Progress Group
        progress_group = QGroupBox("Analysis Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Overall progress
        progress_layout.addWidget(QLabel("Overall Progress:"))
        self.ai_progress = QProgressBar()
        self.ai_progress.setStyleSheet("""
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
        progress_layout.addWidget(self.ai_progress)
        
        # Current segment info
        self.ai_progress_label = QLabel("Ready to analyze...")
        progress_layout.addWidget(self.ai_progress_label)
        
        layout.addWidget(progress_group)
        
        # Results Group
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        # Results area with scroll
        self.ai_results_area = QScrollArea()
        self.ai_results_area.setWidgetResizable(True)
        self.ai_results_area.setMinimumHeight(300)
        
        self.ai_results_widget = QWidget()
        self.ai_results_layout = QVBoxLayout(self.ai_results_widget)
        self.ai_results_layout.addStretch()
        
        self.ai_results_area.setWidget(self.ai_results_widget)
        results_layout.addWidget(self.ai_results_area)
        
        layout.addWidget(results_group)
        
        # Logs Group
        logs_group = QGroupBox("Analysis Logs")
        logs_layout = QVBoxLayout(logs_group)
        
        self.ai_log_text = QTextEdit()
        self.ai_log_text.setReadOnly(True)
        self.ai_log_text.setFont(QFont("Consolas", 10))
        self.ai_log_text.setMaximumHeight(200)
        self.ai_log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        logs_layout.addWidget(self.ai_log_text)
        
        # Clear logs button
        clear_ai_logs_btn = QPushButton("Clear Logs")
        clear_ai_logs_btn.clicked.connect(self.clear_ai_logs)
        logs_layout.addWidget(clear_ai_logs_btn)
        
        layout.addWidget(logs_group)
        
        return results_widget
    
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
        file_layout.addWidget(self.input_folder_label, 0, 1)
        
        self.select_input_btn = QPushButton("Browse...")
        self.select_input_btn.clicked.connect(self.select_input_folder)
        file_layout.addWidget(self.select_input_btn, 0, 2)
        
        # Output folder
        file_layout.addWidget(QLabel("Output Folder:"), 1, 0)
        self.output_folder_label = QLabel("No folder selected")
        self.output_folder_label.setStyleSheet("background: #f0f0f0; padding: 8px; border: 1px solid #ccc; color: black;")
        file_layout.addWidget(self.output_folder_label, 1, 1)
        
        self.select_output_btn = QPushButton("Browse...")
        self.select_output_btn.clicked.connect(self.select_output_folder)
        file_layout.addWidget(self.select_output_btn, 1, 2)
        
        layout.addWidget(file_group)
        
        # Backend Selection Group
        backend_group = QGroupBox("Transcription Backend")
        backend_layout = QVBoxLayout(backend_group)
        
        # Backend selection
        backend_selection_layout = QHBoxLayout()
        backend_selection_layout.addWidget(QLabel("Backend:"))
        
        self.backend_combo = QComboBox()
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
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_info_layout.addWidget(self.model_combo)
        
        model_layout.addLayout(model_info_layout)
        
        # Beam size setting (for faster-whisper)
        self.beam_size_widget = QWidget()
        beam_layout = QHBoxLayout(self.beam_size_widget)
        beam_layout.setContentsMargins(0, 0, 0, 0)
        beam_layout.addWidget(QLabel("Beam Size:"))
        
        self.beam_size_combo = QComboBox()
        self.beam_size_combo.addItems(["1", "3", "5", "7", "10"])
        self.beam_size_combo.setCurrentText("5")
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
        
        # Device Info Group
        device_group = QGroupBox("Device Information")
        device_layout = QVBoxLayout(device_group)
        
        # Device selection
        device_selection_layout = QHBoxLayout()
        device_selection_layout.addWidget(QLabel("Device:"))
        
        self.device_combo = QComboBox()
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        device_selection_layout.addWidget(self.device_combo)
        
        device_layout.addLayout(device_selection_layout)
        
        # Device info display
        self.device_label = QLabel("Detecting available devices...")
        self.device_label.setStyleSheet("color: gray; font-size: 11px;")
        device_layout.addWidget(self.device_label)
        
        layout.addWidget(device_group)
        
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
        
        # Set default to auto
        self.backend_combo.setCurrentIndex(0)
    
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
        
        # Set default to base if available
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
        
        # Save backend preference
        self.settings.setValue("preferred_backend", backend_name)
    
    def on_model_changed(self):
        """Handle model selection change."""
        model_name = self.model_combo.currentText()
        backend_name = self.backend_combo.currentData()
        
        # Update model description
        self.update_model_description()
        
        # Reset transcriber to reload model
        self.transcriber = None
        
        self.log(f"Model changed to: {model_name}", "INFO")
        
        # Save model preference
        self.settings.setValue("preferred_model", model_name)
    
    def on_beam_size_changed(self):
        """Handle beam size change."""
        beam_size = self.beam_size_combo.currentText()
        
        # Reset transcriber to use new beam size
        self.transcriber = None
        
        self.log(f"Beam size changed to: {beam_size}", "INFO")
        
        # Save beam size preference
        self.settings.setValue("beam_size", beam_size)
    
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
        
        # Save device preference
        if device:
            self.settings.setValue("preferred_device", device)
    
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
        
        # Set default selection based on saved preference or auto-detection
        preferred_device = self.settings.value("preferred_device", None)
        
        if preferred_device:
            # Try to find and select the preferred device
            for i in range(self.device_combo.count()):
                if self.device_combo.itemData(i) == preferred_device:
                    self.device_combo.setCurrentIndex(i)
                    break
        else:
            # Auto-select the best available device
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
            self, "Select Input Folder", self.settings.value("input_folder", "")
        )
        if folder:
            self.input_folder_label.setText(folder)
            self.settings.setValue("input_folder", folder)
            self.update_file_count()
    
    def select_output_folder(self):
        """Select output folder for transcriptions."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self.settings.value("output_folder", "")
        )
        if folder:
            self.output_folder_label.setText(folder)
            self.settings.setValue("output_folder", folder)
    
    def update_file_count(self):
        """Update the count of audio files in the selected folder."""
        input_folder = self.input_folder_label.text()
        if input_folder and input_folder != "No folder selected":
            try:
                audio_files = get_audio_files(input_folder)
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
        
        # Create transcriber if needed
        if not self.transcriber:
            model_name = self.model_combo.currentText()
            selected_device = self.device_combo.currentData()
            backend_name = self.backend_combo.currentData()
            beam_size = int(self.beam_size_combo.currentText())
            
            self.transcriber = AudioTranscriber(
                model_name=model_name,
                device=selected_device,
                backend=backend_name,
                beam_size=beam_size
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
        
        # Start worker thread
        self.worker_thread = TranscriptionWorker(self.transcriber, input_folder, output_folder, create_zip=True)
        self.worker_thread.progress_update.connect(self.log)
        self.worker_thread.file_progress.connect(self.update_file_progress)
        self.worker_thread.finished.connect(self.transcription_finished)
        self.worker_thread.start()
    
    def stop_transcription(self):
        """Stop the transcription process."""
        if self.worker_thread:
            self.worker_thread.cancel()
            self.log("Transcription process cancelled by user", "WARNING")
    
    def update_file_progress(self, current: int, total: int):
        """Update file progress display."""
        progress = int((current / total) * 100) if total > 0 else 0
        self.overall_progress.setValue(progress)
        self.file_progress_label.setText(f"Processing file {current} of {total}")
    
    def transcription_finished(self, result: Dict[str, Any]):
        """Handle transcription completion."""
        # Update UI state
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.select_input_btn.setEnabled(True)
        self.select_output_btn.setEnabled(True)
        self.backend_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.beam_size_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        
        if result['success']:
            self.overall_progress.setValue(100)
            self.file_progress_label.setText("Transcription completed!")
            
            # Update results
            total_time = format_time(result['total_time'])
            success_count = result['success_count']
            failure_count = result['failure_count']
            
            results_text = f"""
Transcription Summary:
• Total files processed: {success_count + failure_count}
• Successful: {success_count}
• Failed: {failure_count}
• Total processing time: {total_time}
            """.strip()
            
            self.results_label.setText(results_text)
            self.log("=== TRANSCRIPTION SESSION COMPLETED ===", "SYSTEM")
            self.log(f"Total files processed: {success_count + failure_count}", "INFO")
            self.log(f"Successfully transcribed: {success_count}", "SUCCESS")
            if failure_count > 0:
                self.log(f"Failed transcriptions: {failure_count}", "WARNING")
            self.log(f"Total processing time: {total_time}", "INFO")
            
            # Calculate average processing speed
            if result['total_time'] > 0 and success_count > 0:
                avg_time_per_file = result['total_time'] / (success_count + failure_count)
                self.log(f"Average time per file: {avg_time_per_file:.1f} seconds", "INFO")
            
            # Create HTML report
            output_folder = self.output_folder_label.text()
            html_path = create_html_report(
                result['results'], output_folder, result['total_time'],
                success_count, failure_count
            )
            
            if html_path:
                self.html_report_path = html_path
                self.open_html_btn.setEnabled(True)
                self.log(f"HTML report created: {html_path}", "SUCCESS")
            else:
                self.log("Failed to create HTML report", "WARNING")
            
            # Create JSON transcript for AI review
            if hasattr(self, 'transcriber') and self.transcriber:
                model_info = self.transcriber.get_backend_info()
                json_path = create_json_transcript(
                    result['results'], output_folder, result['total_time'],
                    success_count, failure_count, model_info
                )
                
                if json_path:
                    self.log(f"JSON transcript created: {json_path}", "SUCCESS")
                    self.log("JSON file is ready for AI review processing", "INFO")
                else:
                    self.log("Failed to create JSON transcript", "WARNING")
                
                # Handle zip file if created
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
        current_theme = self.settings.value("theme", "light")
        new_theme = "dark" if current_theme == "light" else "light"
        self.settings.setValue("theme", new_theme)
        self.apply_theme()
    
    def apply_theme(self):
        """Apply the selected theme."""
        theme = self.settings.value("theme", "light")
        
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
    
    def load_settings(self):
        """Load application settings."""
        input_folder = self.settings.value("input_folder", "")
        output_folder = self.settings.value("output_folder", "")
        
        if input_folder:
            self.input_folder_label.setText(input_folder)
            self.update_file_count()
        
        if output_folder:
            self.output_folder_label.setText(output_folder)
        
        # Load backend preference
        preferred_backend = self.settings.value("preferred_backend", "auto")
        for i in range(self.backend_combo.count()):
            if self.backend_combo.itemData(i) == preferred_backend:
                self.backend_combo.setCurrentIndex(i)
                break
        
        # Load model preference
        preferred_model = self.settings.value("preferred_model", "base")
        for i in range(self.model_combo.count()):
            if self.model_combo.itemText(i) == preferred_model:
                self.model_combo.setCurrentIndex(i)
                break
        
        # Load beam size preference
        preferred_beam_size = self.settings.value("beam_size", "5")
        for i in range(self.beam_size_combo.count()):
            if self.beam_size_combo.itemText(i) == preferred_beam_size:
                self.beam_size_combo.setCurrentIndex(i)
                break
        
        # Device preference will be loaded in populate_device_combo()
    
    # AI Review Methods
    def test_ollama_connection(self):
        """Test connection to Ollama instance with comprehensive logging."""
        self.ai_log("=== TESTING OLLAMA CONNECTION ===", "SYSTEM")
        
        def log_callback(message):
            """Callback to capture detailed connection logs."""
            # Parse log level from message if present
            if "] DEBUG:" in message:
                level = "DEBUG"
                msg = message.split("] DEBUG:", 1)[1].strip()
            elif "] ERROR:" in message:
                level = "ERROR" 
                msg = message.split("] ERROR:", 1)[1].strip()
            elif "] INFO:" in message:
                level = "INFO"
                msg = message.split("] INFO:", 1)[1].strip()
            else:
                level = "INFO"
                msg = message
            
            self.ai_log(msg, level)
        
        try:
            result = self.ai_review_manager.test_ollama_connection(log_callback)
            
            if result['success']:
                self.ai_connection_label.setText("Connected ✓")
                self.ai_connection_label.setStyleSheet("color: green; font-weight: bold;")
                self.ai_log(f"✓ Successfully connected to Ollama", "SUCCESS")
                
                # Log detailed connection info
                if 'response_time' in result:
                    self.ai_log(f"Response time: {result['response_time']:.3f}s", "DEBUG")
                if 'ollama_version' in result:
                    self.ai_log(f"Ollama version: {result['ollama_version']}", "DEBUG")
                
                # Update model combo with available models
                self.update_ai_model_combo(result.get('models', []))
                
                # Enable analyze button if we have transcript loaded
                self.update_analyze_button_state()
            else:
                self.ai_connection_label.setText("Failed ✗")
                self.ai_connection_label.setStyleSheet("color: red; font-weight: bold;")
                self.ai_log(f"✗ Connection failed: {result['message']}", "ERROR")
                
                # Log additional error details
                if 'error_type' in result:
                    self.ai_log(f"Error type: {result['error_type']}", "DEBUG")
                if 'error_details' in result:
                    self.ai_log(f"Error details: {result['error_details']}", "DEBUG")
                    
                self.analyze_btn.setEnabled(False)
                
        except Exception as e:
            self.ai_connection_label.setText("Error ✗")
            self.ai_connection_label.setStyleSheet("color: red; font-weight: bold;")
            self.ai_log(f"Connection test exception: {str(e)}", "ERROR")
            import traceback
            self.ai_log(f"Full traceback: {traceback.format_exc()}", "DEBUG")
    
    def update_ai_model_combo(self, available_models):
        """Update AI model combo with available models and detailed logging."""
        self.ai_log(f"Updating model list with {len(available_models)} models", "DEBUG")
        self.ai_model_combo.clear()
        
        if available_models:
            # Add available models
            for i, model in enumerate(available_models):
                if isinstance(model, dict):
                    display_name = f"{model['name']} ({model.get('size', 'Unknown size')})"
                    self.ai_model_combo.addItem(display_name, model['name'])
                    self.ai_log(f"  Model {i+1}: {model['name']} ({model.get('size', 'Unknown size')})", "DEBUG")
                else:
                    self.ai_model_combo.addItem(str(model), str(model))
                    self.ai_log(f"  Model {i+1}: {model}", "DEBUG")
            
            # Set mistral as default if available
            mistral_found = False
            for i in range(self.ai_model_combo.count()):
                model_data = self.ai_model_combo.itemData(i)
                if model_data and 'mistral' in model_data.lower():
                    self.ai_model_combo.setCurrentIndex(i)
                    mistral_found = True
                    self.ai_log(f"Set default model to: {model_data}", "INFO")
                    break
            
            if not mistral_found and self.ai_model_combo.count() > 0:
                # Use first available model if mistral not found
                first_model = self.ai_model_combo.itemData(0)
                self.ai_log(f"Mistral not found, using first available: {first_model}", "INFO")
        else:
            # Add default models if none found
            self.ai_model_combo.addItem("mistral (Download required)", "mistral")
            self.ai_model_combo.addItem("llama2 (Download required)", "llama2")
            self.ai_log("No models found, added default options", "WARNING")
    
    def select_transcript_file(self):
        """Select transcript file for AI analysis with comprehensive logging."""
        self.ai_log("Opening transcript file selection dialog", "DEBUG")
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Transcript File", 
            self.settings.value("ai_transcript_folder", ""),
            "Text files (*.txt);;JSON files (*.json);;All files (*.*)"
        )
        
        if file_path:
            self.ai_log(f"Selected transcript file: {file_path}", "INFO")
            self.ai_transcript_label.setText(file_path)
            self.settings.setValue("ai_transcript_folder", str(Path(file_path).parent))
            
            # Load and segment the transcript
            self.load_transcript_segments(file_path)
            
            # Update button state
            self.update_analyze_button_state()
        else:
            self.ai_log("File selection cancelled", "DEBUG")
    
    def load_transcript_segments(self, file_path):
        """Load and segment the transcript file with comprehensive logging."""
        self.ai_log("=== LOADING TRANSCRIPT SEGMENTS ===", "SYSTEM")
        
        def log_callback(message):
            """Callback to capture detailed loading logs."""
            # Parse log level from message if present
            if "] DEBUG:" in message:
                level = "DEBUG"
                msg = message.split("] DEBUG:", 1)[1].strip()
            elif "] ERROR:" in message:
                level = "ERROR" 
                msg = message.split("] ERROR:", 1)[1].strip()
            elif "] INFO:" in message:
                level = "INFO"
                msg = message.split("] INFO:", 1)[1].strip()
            elif "] WARNING:" in message:
                level = "WARNING"
                msg = message.split("] WARNING:", 1)[1].strip()
            else:
                level = "INFO"
                msg = message
            
            self.ai_log(msg, level)
        
        try:
            # Check if this is a chunked JSON file
            is_chunked_json = Path(file_path).name == "recordings_chunked.json"
            
            # Load the transcript based on file type
            if is_chunked_json:
                self.ai_log("Detected chunked JSON format optimized for AI review", "INFO")
                result = self.ai_review_manager.load_chunked_json_transcript(file_path, log_callback)
            else:
                # Use standard transcript loader
                result = self.ai_review_manager.load_combined_transcript(file_path, log_callback)
            
            if result['success']:
                # Log detailed file information
                self.ai_log(f"File size: {result.get('file_size', 0)} bytes", "DEBUG")
                self.ai_log(f"Character count: {result.get('character_count', 0)}", "DEBUG")
                self.ai_log(f"Word count: {result.get('word_count', 0)}", "DEBUG")
                self.ai_log(f"Line count: {result.get('line_count', 0)}", "DEBUG")
                if 'read_time' in result:
                    self.ai_log(f"File read time: {result['read_time']:.3f}s", "DEBUG")
                
                # Extract JSON metadata for enhanced segmentation
                json_metadata = None
                if result.get('is_json', False):
                    json_metadata = {
                        'is_json': result['is_json'],
                        'original_json': result.get('original_json'),
                        'json_structure': result.get('json_structure')
                    }
                    self.ai_log(f"JSON file detected: {result.get('json_structure', {}).get('estimated_segments', 0)} estimated segments", "INFO")
                
                # Segment the transcript with JSON metadata
                try:
                    segments = self.ai_review_manager.segment_transcript(
                        transcript_content=result['content'], 
                        progress_callback=log_callback, 
                        json_metadata=json_metadata
                    )
                    self.current_segments = segments
                except Exception as e:
                    self.ai_log(f"Error segmenting transcript: {str(e)}", "ERROR")
                    import traceback
                    self.ai_log(f"Segmentation traceback: {traceback.format_exc()}", "DEBUG")
                    self.current_segments = []
                    return
                
                # Update UI with segment info
                total_words = sum(segment.word_count for segment in segments)
                self.segment_info_label.setText(
                    f"Found {len(segments)} recording segments, {total_words:,} total words"
                )
                
                self.ai_log(f"✓ Successfully loaded {len(segments)} segments with {total_words:,} words", "SUCCESS")
                
                # Log detailed segment information
                for i, segment in enumerate(segments):
                    if i < 10:  # Show first 10 segments
                        self.ai_log(f"  Segment {i+1}: {segment.filename} ({segment.word_count} words)", "DEBUG")
                    elif i == 10:
                        self.ai_log(f"  ... and {len(segments) - 10} more segments", "DEBUG")
                        break
                        
            else:
                self.segment_info_label.setText("Failed to load transcript")
                self.ai_log(f"✗ Failed to load transcript: {result['error']}", "ERROR")
                
                # Log additional error details
                if 'error_type' in result:
                    self.ai_log(f"Error type: {result['error_type']}", "DEBUG")
                if 'traceback' in result:
                    self.ai_log(f"Full traceback: {result['traceback']}", "DEBUG")
                    
                self.current_segments = []
                
        except Exception as e:
            self.segment_info_label.setText("Error loading transcript")
            self.ai_log(f"Exception loading transcript: {str(e)}", "ERROR")
            import traceback
            self.ai_log(f"Full traceback: {traceback.format_exc()}", "DEBUG")
            self.current_segments = []
    
    def update_analyze_button_state(self):
        """Update the analyze button enabled state with logging."""
        # Enable if we have Ollama connection, transcript loaded, and case facts
        has_connection = "Connected" in self.ai_connection_label.text()
        has_transcript = hasattr(self, 'current_segments') and self.current_segments
        has_case_facts = bool(self.case_facts_text.toPlainText().strip())
        
        button_enabled = has_connection and has_transcript and has_case_facts
        self.analyze_btn.setEnabled(button_enabled)
        
        # Log button state reasoning
        status_parts = []
        if not has_connection:
            status_parts.append("no Ollama connection")
        if not has_transcript:
            status_parts.append("no transcript loaded")
        if not has_case_facts:
            status_parts.append("no case facts entered")
            
        if status_parts:
            self.ai_log(f"Analyze button disabled: {', '.join(status_parts)}", "DEBUG")
        else:
                    self.ai_log("Analyze button enabled: all requirements met", "DEBUG")

    def start_ai_analysis(self):
        """Start AI analysis of transcript segments with comprehensive logging."""
        try:
            # Validate AI Review Manager
            if not self.ai_review_manager:
                QMessageBox.critical(self, "Error", "AI Review Manager not initialized.")
                return
                
            case_facts = self.case_facts_text.toPlainText().strip()
            if not case_facts:
                QMessageBox.warning(self, "Warning", "Please enter case facts to analyze.")
                return

            if not hasattr(self, 'current_segments') or not self.current_segments:
                QMessageBox.warning(self, "Warning", "Please select and load a transcript file.")
                return
                
            # Validate that segments are actually TranscriptSegment objects
            if not all(hasattr(seg, 'filename') and hasattr(seg, 'content') for seg in self.current_segments):
                QMessageBox.critical(self, "Error", "Invalid transcript segments loaded.")
                return

            # Get selected model
            model_name = self.ai_model_combo.currentData() or "mistral"
            if not model_name:
                QMessageBox.warning(self, "Warning", "Please select an AI model.")
                return
            
            # Clear previous results
            self.clear_ai_results()
            
            # Reset progress
            self.ai_progress.setValue(0)
            self.ai_progress_label.setText("Starting AI analysis...")
            
            # Update UI state
            self.analyze_btn.setEnabled(False)
            self.stop_ai_btn.setEnabled(True)
            self.test_connection_btn.setEnabled(False)
            self.select_transcript_btn.setEnabled(False)
            
            # Log comprehensive analysis start information
            self.ai_log("=== AI ANALYSIS SESSION STARTED ===", "SYSTEM")
            self.ai_log(f"Model: {model_name}", "INFO")
            self.ai_log(f"Segments to analyze: {len(self.current_segments)}", "INFO")
            self.ai_log(f"Total words to analyze: {sum(seg.word_count for seg in self.current_segments):,}", "INFO")
            self.ai_log(f"Case facts length: {len(case_facts)} characters", "DEBUG")
            self.ai_log(f"Case facts preview: {case_facts[:200]}{'...' if len(case_facts) > 200 else ''}", "DEBUG")
            
            # Log save options
            save_individual = self.save_individual_checkbox.isChecked()
            save_combined = self.save_combined_checkbox.isChecked()
            self.ai_log(f"Save individual files: {save_individual}", "DEBUG")
            self.ai_log(f"Save combined summary: {save_combined}", "DEBUG")
            
            # Start worker thread
            self.ai_worker_thread = AIReviewWorker(
                ai_manager=self.ai_review_manager,
                segments=self.current_segments,
                case_facts=case_facts,
                model_name=model_name,
                log_callback=self.ai_log_callback
            )
            
            self.ai_worker_thread.progress_update.connect(self.ai_log_callback)
            self.ai_worker_thread.segment_complete.connect(self.on_segment_complete)
            self.ai_worker_thread.finished.connect(self.on_ai_analysis_finished)
            self.ai_worker_thread.start()
            
        except Exception as e:
            self.ai_log(f"Error starting AI analysis: {str(e)}", "ERROR")
            import traceback
            self.ai_log(f"Full traceback: {traceback.format_exc()}", "DEBUG")
            QMessageBox.critical(self, "Error", f"Failed to start AI analysis: {str(e)}")
            
            # Reset UI state on error
            self.analyze_btn.setEnabled(True)
            self.stop_ai_btn.setEnabled(False)
            self.test_connection_btn.setEnabled(True)
            self.select_transcript_btn.setEnabled(True)
    
    def ai_log_callback(self, message, level="INFO"):
        """Enhanced callback for AI analysis logging with level detection."""
        # Parse log level from message if present in enhanced format
        if "] DEBUG:" in message:
            level = "DEBUG"
            msg = message.split("] DEBUG:", 1)[1].strip()
        elif "] ERROR:" in message:
            level = "ERROR" 
            msg = message.split("] ERROR:", 1)[1].strip()
        elif "] INFO:" in message:
            level = "INFO"
            msg = message.split("] INFO:", 1)[1].strip()
        elif "] WARNING:" in message:
            level = "WARNING"
            msg = message.split("] WARNING:", 1)[1].strip()
        else:
            # Legacy format or direct message
            msg = message
            
        self.ai_log(msg, level)
    
    def stop_ai_analysis(self):
        """Stop the AI analysis process with logging."""
        self.ai_log("User requested analysis cancellation", "WARNING")
        if hasattr(self, 'ai_worker_thread') and self.ai_worker_thread:
            self.ai_worker_thread.cancel()
            self.ai_log("Cancellation request sent to worker thread", "INFO")
    
    def on_segment_complete(self, result, current, total):
        """Handle completion of a segment analysis with detailed logging."""
        try:
            # Update progress
            progress_pct = (current / total) * 100
            if hasattr(self, 'ai_progress') and self.ai_progress:
                self.ai_progress.setValue(int(progress_pct))
            
            segment = result.get('segment', {}) if isinstance(result, dict) else {}
            
            if result.get('success', False):
                # Extract relevance information
                relevance_info = result.get('relevance_score', {})
                if isinstance(relevance_info, dict):
                    is_relevant = relevance_info.get('is_relevant', False)
                    relevance_score = relevance_info.get('relevance_score', 0)
                else:
                    is_relevant = False
                    relevance_score = 0
                
                # Update progress label with relevance info
                relevance_text = "RELEVANT" if is_relevant else "Not relevant"
                filename = getattr(segment, 'filename', 'unknown') if hasattr(segment, 'filename') else 'unknown'
                
                if hasattr(self, 'ai_progress_label') and self.ai_progress_label:
                    self.ai_progress_label.setText(f"Segment {current}/{total}: {filename} - {relevance_text}")
                
                # Log detailed completion info
                processing_time = result.get('processing_time', 0)
                response_length = result.get('response_length', 0)
                response_words = result.get('response_words', 0)
                
                self.ai_log(f"✓ Completed {filename} ({processing_time:.1f}s)", "SUCCESS")
                self.ai_log(f"  Relevance: {relevance_text} (Score: {relevance_score})", "DEBUG")
                self.ai_log(f"  Response: {response_length} chars, {response_words} words", "DEBUG")
                
                # Log AI metrics if available
                ai_metrics = result.get('ai_metrics', {})
                if ai_metrics.get('eval_count'):
                    self.ai_log(f"  AI tokens generated: {ai_metrics['eval_count']}", "DEBUG")
                if ai_metrics.get('eval_duration'):
                    eval_duration = ai_metrics['eval_duration'] / 1e9  # Convert nanoseconds to seconds
                    self.ai_log(f"  AI generation time: {eval_duration:.1f}s", "DEBUG")
            else:
                error = result.get('error', 'Unknown error')
                filename = getattr(segment, 'filename', 'unknown') if hasattr(segment, 'filename') else 'unknown'
                if hasattr(self, 'ai_progress_label') and self.ai_progress_label:
                    self.ai_progress_label.setText(f"Segment {current}/{total}: {filename} - FAILED")
                self.ai_log(f"✗ Failed {filename}: {error}", "ERROR")
        except Exception as e:
            self.ai_log(f"Error in segment completion handler: {str(e)}", "ERROR")
            print(f"[SEGMENT_COMPLETE_ERROR] {e}", file=sys.stderr)
        
        # Add result to display
        self.add_result_to_display(result, current)
    
    def add_result_to_display(self, result, segment_number):
        """Add a single analysis result to the display with enhanced formatting."""
        # Create result widget
        result_frame = QFrame()
        result_frame.setFrameStyle(QFrame.StyledPanel)
        result_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #ddd;
                border-radius: 6px;
                margin: 5px;
                padding: 10px;
                background-color: #f9f9f9;
            }
        """)
        
        result_layout = QVBoxLayout(result_frame)
        
        # Header
        header_layout = QHBoxLayout()
        
        # Status indicator and filename
        segment = result['segment']
        if result['success']:
            # Extract relevance information for better display
            relevance_info = result.get('relevance_score', {})
            if isinstance(relevance_info, dict):
                is_relevant = relevance_info.get('is_relevant', False)
                relevance_score = relevance_info.get('relevance_score', 0)
            else:
                is_relevant = False
                relevance_score = 0
                
            status_color = "#28a745" if is_relevant else "#6c757d"  # Green for relevant, gray for not relevant
            status_icon = "✓"
            relevance_text = f"RELEVANT (Score: {relevance_score})" if is_relevant else "Not Relevant"
        else:
            status_color = "#dc3545"  # Red for errors
            status_icon = "✗"
            relevance_text = f"Error: {result.get('error', 'Unknown error')}"
        
        header_label = QLabel(f"{status_icon} Segment {segment_number}: {segment.filename}")
        header_label.setStyleSheet(f"font-weight: bold; color: {status_color}; font-size: 12px;")
        header_layout.addWidget(header_label)
        
        header_layout.addStretch()
        
        # Processing time and metrics
        if result['success']:
            processing_time = result.get('processing_time', 0)
            response_length = result.get('response_length', 0)
            response_words = result.get('response_words', 0)
            
            metrics_text = f"{processing_time:.1f}s | {response_words} words"
            metrics_label = QLabel(metrics_text)
            metrics_label.setStyleSheet("color: #6c757d; font-size: 10px;")
            header_layout.addWidget(metrics_label)
        
        result_layout.addLayout(header_layout)
        
        # Relevance indicator with more details
        relevance_layout = QHBoxLayout()
        relevance_label = QLabel(relevance_text)
        relevance_label.setStyleSheet(f"color: {status_color}; font-style: italic; margin-bottom: 10px; font-size: 11px;")
        relevance_layout.addWidget(relevance_label)
        
        # Add word count info
        word_count_label = QLabel(f"({segment.word_count} words in transcript)")
        word_count_label.setStyleSheet("color: #6c757d; font-size: 10px;")
        relevance_layout.addWidget(word_count_label)
        
        relevance_layout.addStretch()
        result_layout.addLayout(relevance_layout)
        
        # AI response (if successful)
        if result['success'] and result['ai_response']:
            response_text = QTextEdit()
            response_text.setPlainText(result['ai_response'])
            response_text.setReadOnly(True)
            response_text.setMaximumHeight(150)
            response_text.setStyleSheet("""
                QTextEdit {
                    background-color: white;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 11px;
                    padding: 8px;
                }
            """)
            result_layout.addWidget(response_text)
        
        # Insert before the stretch at the end
        self.ai_results_layout.insertWidget(self.ai_results_layout.count() - 1, result_frame)
        
        # Scroll to the new result
        self.ai_results_area.verticalScrollBar().setValue(
            self.ai_results_area.verticalScrollBar().maximum()
        )
    
    def on_ai_analysis_finished(self, results):
        """Handle completion of all AI analysis with comprehensive logging and statistics."""
        try:
            # Update UI state
            if hasattr(self, 'analyze_btn') and self.analyze_btn:
                self.analyze_btn.setEnabled(True)
            if hasattr(self, 'stop_ai_btn') and self.stop_ai_btn:
                self.stop_ai_btn.setEnabled(False)
            if hasattr(self, 'test_connection_btn') and self.test_connection_btn:
                self.test_connection_btn.setEnabled(True)
            if hasattr(self, 'select_transcript_btn') and self.select_transcript_btn:
                self.select_transcript_btn.setEnabled(True)
            
            # Update progress
            if hasattr(self, 'ai_progress') and self.ai_progress:
                self.ai_progress.setValue(100)
            if hasattr(self, 'ai_progress_label') and self.ai_progress_label:
                self.ai_progress_label.setText("Analysis complete")
            
            # Calculate and log comprehensive statistics
            total_segments = len(results)
            successful_analyses = sum(1 for r in results if r.get('success', False))
            failed_analyses = total_segments - successful_analyses
            
            # Calculate relevance statistics
            relevant_segments = 0
            total_processing_time = 0
            total_response_words = 0
            
            for result in results:
                if result.get('success', False):
                    total_processing_time += result.get('processing_time', 0)
                    total_response_words += result.get('response_words', 0)
                    
                    relevance_info = result.get('relevance_score', {})
                    if isinstance(relevance_info, dict) and relevance_info.get('is_relevant', False):
                        relevant_segments += 1
            
            avg_processing_time = total_processing_time / successful_analyses if successful_analyses > 0 else 0
            avg_response_words = total_response_words / successful_analyses if successful_analyses > 0 else 0
            
            # Log comprehensive completion statistics
            self.ai_log("=== AI ANALYSIS SESSION COMPLETED ===", "SYSTEM")
            self.ai_log(f"✓ Analysis completed successfully", "SUCCESS")
            self.ai_log(f"Total segments: {total_segments}", "INFO")
            self.ai_log(f"Successful analyses: {successful_analyses}", "INFO")
            self.ai_log(f"Failed analyses: {failed_analyses}", "INFO")
            self.ai_log(f"Relevant segments found: {relevant_segments}", "INFO")
            self.ai_log(f"Total processing time: {total_processing_time:.1f}s", "INFO")
            self.ai_log(f"Average processing time: {avg_processing_time:.1f}s per segment", "DEBUG")
            self.ai_log(f"Average response length: {avg_response_words:.0f} words", "DEBUG")
            
            if failed_analyses > 0:
                self.ai_log(f"⚠️  Some segments failed to analyze", "WARNING")
            
            if relevant_segments == 0 and successful_analyses > 0:
                self.ai_log("⚠️  No relevant content found in any segments", "WARNING")
            elif relevant_segments > 0:
                relevance_pct = (relevant_segments / successful_analyses) * 100
                self.ai_log(f"Found relevant content in {relevance_pct:.1f}% of analyzed segments", "INFO")
            
                # Save results if options are enabled
                if results and (self.save_individual_checkbox.isChecked() or self.save_combined_checkbox.isChecked()):
                    self.save_ai_results(results)
        except Exception as e:
            self.ai_log(f"Error in analysis completion handler: {str(e)}", "ERROR")
            print(f"[ANALYSIS_COMPLETE_ERROR] {e}", file=sys.stderr)
            # Still try to re-enable buttons
            try:
                if hasattr(self, 'analyze_btn') and self.analyze_btn:
                    self.analyze_btn.setEnabled(True)
                if hasattr(self, 'stop_ai_btn') and self.stop_ai_btn:
                    self.stop_ai_btn.setEnabled(False)
            except:
                pass
    
    def save_ai_results(self, results):
        """Save AI analysis results with comprehensive logging."""
        self.ai_log("=== SAVING ANALYSIS RESULTS ===", "SYSTEM")
        
        def log_callback(message):
            """Callback to capture detailed save logs."""
            # Parse log level from message if present
            if "] DEBUG:" in message:
                level = "DEBUG"
                msg = message.split("] DEBUG:", 1)[1].strip()
            elif "] ERROR:" in message:
                level = "ERROR" 
                msg = message.split("] ERROR:", 1)[1].strip()
            elif "] INFO:" in message:
                level = "INFO"
                msg = message.split("] INFO:", 1)[1].strip()
            elif "] WARNING:" in message:
                level = "WARNING"
                msg = message.split("] WARNING:", 1)[1].strip()
            else:
                level = "INFO"
                msg = message
            
            self.ai_log(msg, level)
        
        try:
            # Get output directory (use current output folder setting)
            output_directory = self.output_folder_label.text()
            if output_directory == "No folder selected":
                output_directory = "outputs"
            
            # Get audio directory for linking in crime report
            audio_directory = self.input_folder_label.text()
            if audio_directory == "No folder selected":
                audio_directory = None
            
            case_facts = self.case_facts_text.toPlainText().strip()
            save_individual = self.save_individual_checkbox.isChecked()
            save_combined = self.save_combined_checkbox.isChecked()
            
            self.ai_log(f"Output directory: {output_directory}", "DEBUG")
            self.ai_log(f"Audio directory: {audio_directory}", "DEBUG")
            self.ai_log(f"Save individual files: {save_individual}", "DEBUG")
            self.ai_log(f"Save combined summary: {save_combined}", "DEBUG")
            
            # Save using the enhanced save method with crime report
            save_result = self.ai_review_manager.save_results_to_files(
                results=results,
                output_directory=output_directory,
                case_facts=case_facts,
                save_individual=save_individual,
                save_combined=save_combined,
                save_crime_report=True,
                audio_directory=audio_directory,
                progress_callback=log_callback
            )
            
            if save_result['success']:
                files_saved = save_result['files_saved']
                self.ai_log(f"✓ Successfully saved {len(files_saved)} files", "SUCCESS")
                
                for file_path in files_saved:
                    self.ai_log(f"  Saved: {file_path}", "DEBUG")
                    
                if 'save_time' in save_result:
                    self.ai_log(f"Save operation completed in {save_result['save_time']:.3f}s", "DEBUG")
            else:
                error = save_result.get('error', 'Unknown error')
                self.ai_log(f"✗ Failed to save results: {error}", "ERROR")
                
                if 'error_type' in save_result:
                    self.ai_log(f"Error type: {save_result['error_type']}", "DEBUG")
                if 'traceback' in save_result:
                    self.ai_log(f"Full traceback: {save_result['traceback']}", "DEBUG")
                    
        except Exception as e:
            self.ai_log(f"Exception during save operation: {str(e)}", "ERROR")
            import traceback
            self.ai_log(f"Full traceback: {traceback.format_exc()}", "DEBUG")
    
    def clear_ai_results(self):
        """Clear the AI results display."""
        # Remove all result widgets except the stretch
        while self.ai_results_layout.count() > 1:
            child = self.ai_results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def clear_ai_logs(self):
        """Clear the AI logs display."""
        self.ai_log_text.clear()
    
    def ai_log(self, message: str, level: str = "INFO"):
        """Add a message to the AI log display - thread-safe version."""
        from PySide6.QtCore import QMetaObject, Qt
        
        # If we're not on the main thread, use invokeMethod to safely update GUI
        if threading.current_thread() != threading.main_thread():
            QMetaObject.invokeMethod(self, "_ai_log_impl", Qt.QueuedConnection,
                                   message, level)
        else:
            self._ai_log_impl(message, level)
    
    def _ai_log_impl(self, message: str, level: str = "INFO"):
        """Internal implementation of ai_log that runs on the main thread."""
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # Color coding based on log level
            color_map = {
                "INFO": "#ffffff",
                "SUCCESS": "#28a745",
                "WARNING": "#ffc107",
                "ERROR": "#dc3545",
                "DEBUG": "#6c757d",
                "SYSTEM": "#17a2b8"
            }
            
            # Icon mapping for different log levels
            icon_map = {
                "INFO": "ℹ️",
                "SUCCESS": "✅",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "DEBUG": "🔍",
                "SYSTEM": "🤖"
            }
            
            color = color_map.get(level, "#ffffff")
            icon = icon_map.get(level, "•")
            
            # Format message with HTML for colored output
            formatted_message = f'<span style="color: {color};">[{timestamp}] {icon} {level}: {message}</span>'
            
            # Safely check if the widget still exists
            if hasattr(self, 'ai_log_text') and self.ai_log_text:
                # Append as HTML to preserve colors
                cursor = self.ai_log_text.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                cursor.insertHtml(formatted_message + "<br>")
                
                # Auto-scroll to bottom
                self.ai_log_text.setTextCursor(cursor)
        except Exception as e:
            # Fallback to print if GUI logging fails
            print(f"[AI_LOG_ERROR] {e}: {message}", file=sys.stderr)
    
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
            
            # Check AI analysis thread
            if hasattr(self, 'ai_worker_thread') and self.ai_worker_thread and self.ai_worker_thread.isRunning():
                operations_running.append("AI analysis")
            
            if operations_running:
                operations_text = " and ".join(operations_running)
                reply = QMessageBox.question(self, "Confirm Exit", 
                                           f"{operations_text.title()} is in progress. Are you sure you want to exit?",
                                           QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    # Stop transcription
                    if hasattr(self, 'worker_thread') and self.worker_thread:
                        self.worker_thread.cancel()
                        if not self.worker_thread.wait(3000):
                            self.worker_thread.terminate()
                            self.worker_thread.wait(1000)
                    
                    # Stop AI analysis
                    if hasattr(self, 'ai_worker_thread') and self.ai_worker_thread:
                        self.ai_worker_thread.cancel()
                        if not self.ai_worker_thread.wait(3000):
                            self.ai_worker_thread.terminate()
                            self.ai_worker_thread.wait(1000)
                else:
                    event.ignore()
                    return
            
            # Save settings before closing
            if hasattr(self, 'settings'):
                self.settings.sync()
            
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