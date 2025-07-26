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
    QMessageBox, QStatusBar, QMenuBar, QMenu, QCheckBox
)
from PySide6.QtCore import QThread, Signal, QTimer, Qt, QSettings
from PySide6.QtGui import QFont, QIcon, QAction, QPalette, QTextCursor

from transcriber import AudioTranscriber
from utils import get_audio_files, create_html_report, format_time


class TranscriptionWorker(QThread):
    """Worker thread for running transcription in the background."""
    
    progress_update = Signal(str)
    file_progress = Signal(int, int)
    finished = Signal(dict)
    
    def __init__(self, transcriber: AudioTranscriber, input_dir: str, output_dir: str):
        super().__init__()
        self.transcriber = transcriber
        self.input_dir = input_dir
        self.output_dir = output_dir
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
                file_progress_callback
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
                    'failure_count': 0
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
        self.settings = QSettings("AudioTranscriber", "Settings")
        
        # Initialize UI
        self.init_ui()
        self.load_settings()
        
        # Initialize device selection
        self.populate_device_combo()
        self.update_device_info()
        self.update_status()
        
        # Set up timer for periodic updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(1000)  # Update every second
    
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
        
        # Create main content in splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Configuration
        config_widget = self.create_config_panel()
        splitter.addWidget(config_widget)
        
        # Right panel - Progress and logs
        progress_widget = self.create_progress_panel()
        splitter.addWidget(progress_widget)
        
        # Set splitter proportions
        splitter.setSizes([400, 500])
        
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
    
    def create_header(self, layout):
        """Create the application header."""
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.StyledPanel)
        header_layout = QVBoxLayout(header_frame)
        
        title_label = QLabel("Audio Transcriber")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)
        
        subtitle_label = QLabel("Powered by OpenAI Whisper")
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
        
        # Model Selection Group
        model_group = QGroupBox("Whisper Model")
        model_layout = QVBoxLayout(model_group)
        
        model_info_layout = QHBoxLayout()
        model_info_layout.addWidget(QLabel("Model:"))
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("base")
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_info_layout.addWidget(self.model_combo)
        
        model_layout.addLayout(model_info_layout)
        
        # Model description
        self.model_description = QLabel(self.get_model_description("base"))
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
        
        self.open_html_btn = QPushButton("Open HTML Report")
        self.open_html_btn.setEnabled(False)
        self.open_html_btn.clicked.connect(self.open_html_report)
        results_layout.addWidget(self.open_html_btn)
        
        layout.addWidget(results_group)
        
        return progress_widget
    
    def get_model_description(self, model_name: str) -> str:
        """Get description for the selected model."""
        descriptions = {
            "tiny": "Fastest model, lower accuracy (~32x realtime)",
            "base": "Good balance of speed and accuracy (~16x realtime)",
            "small": "Better accuracy, moderate speed (~6x realtime)",
            "medium": "High accuracy, slower processing (~2x realtime)",
            "large": "Best accuracy, slowest processing (~1x realtime)"
        }
        return descriptions.get(model_name, "Unknown model")
    
    def on_model_changed(self):
        """Handle model selection change."""
        model_name = self.model_combo.currentText()
        self.model_description.setText(self.get_model_description(model_name))
        self.transcriber = None  # Reset transcriber to reload model
    
    def on_device_changed(self):
        """Handle device selection change."""
        self.transcriber = None  # Reset transcriber to use new device
        self.update_device_info()
        # Save device preference
        device = self.device_combo.currentData()
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
                self.log(f"Found {len(audio_files)} audio files in selected folder")
            except Exception as e:
                self.log(f"Error scanning folder: {e}")
    
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
            self.transcriber = AudioTranscriber(model_name, device=selected_device)
        
        # Reset progress
        self.overall_progress.setValue(0)
        self.file_progress_label.setText("Initializing...")
        self.log("Starting transcription process...")
        
        # Update UI state
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.select_input_btn.setEnabled(False)
        self.select_output_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        
        # Start worker thread
        self.worker_thread = TranscriptionWorker(self.transcriber, input_folder, output_folder)
        self.worker_thread.progress_update.connect(self.log)
        self.worker_thread.file_progress.connect(self.update_file_progress)
        self.worker_thread.finished.connect(self.transcription_finished)
        self.worker_thread.start()
    
    def stop_transcription(self):
        """Stop the transcription process."""
        if self.worker_thread:
            self.worker_thread.cancel()
            self.log("Stopping transcription...")
    
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
        self.model_combo.setEnabled(True)
        
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
            self.log(f"Transcription completed! {success_count} successful, {failure_count} failed")
            
            # Create HTML report
            output_folder = self.output_folder_label.text()
            html_path = create_html_report(
                result['results'], output_folder, result['total_time'],
                success_count, failure_count
            )
            
            if html_path:
                self.html_report_path = html_path
                self.open_html_btn.setEnabled(True)
                self.log(f"HTML report created: {html_path}")
        else:
            self.log(f"Transcription failed: {result.get('error', 'Unknown error')}")
            QMessageBox.critical(self, "Error", f"Transcription failed:\n{result.get('error', 'Unknown error')}")
        
        # Clean up worker thread
        self.worker_thread = None
    
    def open_html_report(self):
        """Open the HTML report in the default browser."""
        if hasattr(self, 'html_report_path') and self.html_report_path:
            import webbrowser
            webbrowser.open(f"file:///{self.html_report_path}")
    
    def log(self, message: str):
        """Add a message to the log display."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
    
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
        
        # Device preference will be loaded in populate_device_combo()
    
    def closeEvent(self, event):
        """Handle application close event."""
        try:
            # Stop the timer first
            if hasattr(self, 'timer') and self.timer:
                self.timer.stop()
            
            # Stop any running transcription
            if hasattr(self, 'worker_thread') and self.worker_thread and self.worker_thread.isRunning():
                reply = QMessageBox.question(self, "Confirm Exit", 
                                           "Transcription is in progress. Are you sure you want to exit?",
                                           QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.worker_thread.cancel()
                    # Give the thread a reasonable time to finish
                    if not self.worker_thread.wait(3000):  # 3 second timeout
                        # Force terminate if it doesn't finish gracefully
                        self.worker_thread.terminate()
                        self.worker_thread.wait(1000)  # Wait another second for termination
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