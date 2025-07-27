import os
import re
import threading
import traceback
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from ollama_client import OllamaClient


class TranscriptSegment:
    """Represents a single recording segment from the combined transcript."""
    
    def __init__(self, filename: str, content: str, segment_index: int, 
                 detailed_segments: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize transcript segment.
        
        Args:
            filename: Original audio filename
            content: Transcript content for this segment
            segment_index: Index of this segment in the overall transcript
            detailed_segments: List of timestamped segments within this recording
        """
        self.filename = filename
        self.content = content.strip() if content else ""
        self.segment_index = segment_index
        self.word_count = len(self.content.split()) if self.content else 0
        # Ensure detailed_segments is always a list
        self.detailed_segments = detailed_segments if detailed_segments is not None else []
        
    def __str__(self):
        return f"Segment {self.segment_index}: {self.filename} ({self.word_count} words)"


class CrimeRelevantSection:
    """Represents a section of transcript that is relevant to a crime investigation."""
    
    def __init__(self, filename: str, text: str, start_time: float, end_time: float, 
                 relevance_explanation: str, audio_file_path: Optional[str] = None):
        """
        Initialize crime-relevant section.
        
        Args:
            filename: Original audio filename
            text: Relevant text content
            start_time: Start time in seconds
            end_time: End time in seconds
            relevance_explanation: AI explanation of why this is relevant
            audio_file_path: Path to the audio file for direct linking
        """
        self.filename = filename
        self.text = text.strip() if text else ""
        self.start_time = max(0.0, float(start_time) if start_time is not None else 0.0)
        self.end_time = max(self.start_time, float(end_time) if end_time is not None else self.start_time)
        self.relevance_explanation = relevance_explanation or "Crime-relevant content identified"
        self.audio_file_path = audio_file_path
        
    def get_audio_link(self, timestamp_offset: float = 0) -> str:
        """Generate a direct link to the audio file at the specific timestamp."""
        if not self.audio_file_path:
            return f"#{self.filename}@{self._format_timestamp(self.start_time)}"
        
        # Create file URI with timestamp parameter
        file_uri = Path(self.audio_file_path).as_uri()
        start_seconds = int(self.start_time + timestamp_offset)
        return f"{file_uri}#t={start_seconds}"
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        try:
            seconds = max(0.0, float(seconds))
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            if minutes >= 60:
                hours = minutes // 60
                minutes = minutes % 60
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            return f"{minutes:02d}:{secs:02d}"
        except (ValueError, TypeError):
            return "00:00"
    
    def get_timestamp_range(self) -> str:
        """Get formatted timestamp range."""
        start_str = self._format_timestamp(self.start_time)
        end_str = self._format_timestamp(self.end_time)
        return f"{start_str} - {end_str}"


class AIReviewManager:
    """Manages AI review process for transcribed audio segments."""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize AI Review Manager.
        
        Args:
            ollama_base_url: Base URL for Ollama API
        """
        self.ollama_client = OllamaClient(ollama_base_url)
        self.is_cancelled = False
        self.enable_terminal_logging = True  # Enable comprehensive terminal logging
        
        # Initialize terminal logging
        self._log_terminal_info("AI Review Manager initialized")
        self._log_terminal_debug(f"Ollama base URL: {ollama_base_url}")
        
    def _log_terminal(self, level: str, message: str):
        """Log message directly to terminal with timestamp and level."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            formatted_msg = f"[{timestamp}] {level}: {message}"
            
            if level in ["ERROR", "CRITICAL"]:
                print(formatted_msg, file=sys.stderr, flush=True)
            else:
                print(formatted_msg, flush=True)
        except Exception as e:
            # Fallback logging if terminal logging fails
            try:
                print(f"[LOGGING ERROR] {e}: {message}", file=sys.stderr, flush=True)
            except:
                pass  # Silent fallback
    
    def _log_terminal_debug(self, message: str):
        """Log debug message to terminal."""
        if self.enable_terminal_logging:
            self._log_terminal("DEBUG", message)
    
    def _log_terminal_info(self, message: str):
        """Log info message to terminal."""
        if self.enable_terminal_logging:
            self._log_terminal("INFO", message)
    
    def _log_terminal_warning(self, message: str):
        """Log warning message to terminal."""
        if self.enable_terminal_logging:
            self._log_terminal("WARNING", message)
    
    def _log_terminal_error(self, message: str):
        """Log error message to terminal."""
        if self.enable_terminal_logging:
            self._log_terminal("ERROR", message)
    
    def _log_terminal_critical(self, message: str):
        """Log critical error message to terminal."""
        if self.enable_terminal_logging:
            self._log_terminal("CRITICAL", message)
        
    def _log_debug(self, message: str, callback: Optional[Callable] = None):
        """Log debug message with timestamp to both callback and terminal."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log_msg = f"[{timestamp}] DEBUG: {message}"
            
            # Log to terminal
            self._log_terminal_debug(message)
            
            # Log to callback if provided
            if callback:
                callback(log_msg)
        except Exception as e:
            self._log_terminal_error(f"Logging error in _log_debug: {e}")
            
    def _log_error(self, message: str, callback: Optional[Callable] = None):
        """Log error message with timestamp to both callback and terminal."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log_msg = f"[{timestamp}] ERROR: {message}"
            
            # Log to terminal
            self._log_terminal_error(message)
            
            # Log to callback if provided
            if callback:
                callback(log_msg)
        except Exception as e:
            self._log_terminal_critical(f"Critical logging error in _log_error: {e}")
            
    def _log_info(self, message: str, callback: Optional[Callable] = None):
        """Log info message with timestamp to both callback and terminal."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log_msg = f"[{timestamp}] INFO: {message}"
            
            # Log to terminal
            self._log_terminal_info(message)
            
            # Log to callback if provided
            if callback:
                callback(log_msg)
        except Exception as e:
            self._log_terminal_error(f"Logging error in _log_info: {e}")
            
    def _log_warning(self, message: str, callback: Optional[Callable] = None):
        """Log warning message with timestamp to both callback and terminal."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log_msg = f"[{timestamp}] WARNING: {message}"
            
            # Log to terminal
            self._log_terminal_warning(message)
            
            # Log to callback if provided
            if callback:
                callback(log_msg)
        except Exception as e:
            self._log_terminal_error(f"Logging error in _log_warning: {e}")

    def _log_exception(self, message: str, exception: Exception, callback: Optional[Callable] = None):
        """Log exception with full traceback to both callback and terminal."""
        try:
            exc_type = type(exception).__name__
            exc_message = str(exception)
            full_traceback = traceback.format_exc()
            
            error_msg = f"{message} - {exc_type}: {exc_message}"
            traceback_msg = f"Full traceback:\n{full_traceback}"
            
            # Log to terminal
            self._log_terminal_error(error_msg)
            self._log_terminal_debug(traceback_msg)
            
            # Log to callback if provided
            if callback:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                callback(f"[{timestamp}] ERROR: {error_msg}")
                callback(f"[{timestamp}] DEBUG: {traceback_msg}")
                
        except Exception as e:
            self._log_terminal_critical(f"Critical error in exception logging: {e}")
        
    def load_combined_transcript(self, transcript_path: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Load the combined transcript file with comprehensive logging and JSON support.
        
        Args:
            transcript_path: Path to the combined transcript file (text or JSON)
            progress_callback: Optional callback for logging progress
            
        Returns:
            Dict with success status, content, and detailed file information
        """
        self._log_info(f"=== STARTING TRANSCRIPT LOADING ===", progress_callback)
        self._log_info(f"Loading combined transcript from: {transcript_path}", progress_callback)
        
        try:
            # Check if file exists
            self._log_debug(f"Checking if file exists: {transcript_path}", progress_callback)
            if not os.path.exists(transcript_path):
                error_msg = f"Transcript file not found: {transcript_path}"
                self._log_error(error_msg, progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'content': '',
                    'file_path': transcript_path
                }
                
            self._log_debug("File exists, getting file statistics...", progress_callback)

            # Get file info
            file_stat = os.stat(transcript_path)
            file_size = file_stat.st_size
            modified_time = datetime.fromtimestamp(file_stat.st_mtime)
            
            self._log_debug(f"File size: {file_size} bytes ({file_size / 1024:.1f} KB)", progress_callback)
            self._log_debug(f"Last modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}", progress_callback)

            # Determine file type
            file_extension = Path(transcript_path).suffix.lower()
            is_json_file = file_extension == '.json'
            
            self._log_debug(f"File type detected: {'JSON' if is_json_file else 'Text'}", progress_callback)
            self._log_debug(f"File extension: {file_extension}", progress_callback)

            # Read file content
            self._log_debug("Starting file read operation...", progress_callback)
            start_time = datetime.now()
            
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
            except Exception as read_error:
                self._log_exception("Error during file read", read_error, progress_callback)
                raise
                
            read_time = (datetime.now() - start_time).total_seconds()
            
            self._log_debug(f"File read completed in {read_time:.3f}s", progress_callback)
            self._log_debug(f"Raw content length: {len(raw_content)} characters", progress_callback)

            if not raw_content.strip():
                error_msg = "Transcript file is empty"
                self._log_error(error_msg, progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'content': '',
                    'file_size': file_size,
                    'file_path': transcript_path
                }

            # Process content based on file type
            self._log_debug("Processing content based on file type...", progress_callback)
            if is_json_file:
                try:
                    self._log_debug("Attempting to parse JSON content...", progress_callback)
                    # Parse JSON content
                    json_data = json.loads(raw_content)
                    self._log_info("✓ JSON file parsed successfully", progress_callback)
                    self._log_debug(f"JSON data type: {type(json_data)}", progress_callback)
                    
                    # Convert JSON to text format for processing
                    self._log_debug("Converting JSON to text format...", progress_callback)
                    content = self._convert_json_to_text(json_data, progress_callback)
                    self._log_debug(f"Converted content length: {len(content)} characters", progress_callback)
                    
                    # Store original JSON data for enhanced processing
                    self._log_debug("Analyzing JSON structure...", progress_callback)
                    json_metadata = {
                        'is_json': True,
                        'original_json': json_data,
                        'json_structure': self._analyze_json_structure(json_data, progress_callback)
                    }
                    
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON format: {str(e)}"
                    self._log_exception("JSON parsing failed", e, progress_callback)
                    return {
                        'success': False,
                        'error': error_msg,
                        'content': '',
                        'file_size': file_size,
                        'file_path': transcript_path,
                        'error_type': 'JSONDecodeError',
                        'error_details': str(e)
                    }
                except Exception as json_error:
                    self._log_exception("Unexpected error during JSON processing", json_error, progress_callback)
                    raise
            else:
                # Text file processing (existing logic)
                self._log_debug("Processing as text file", progress_callback)
                content = raw_content
                json_metadata = {'is_json': False}

            # Analyze content
            self._log_debug("Analyzing processed content...", progress_callback)
            char_count = len(content)
            word_count = len(content.split())
            line_count = len(content.splitlines())
            
            self._log_info(f"✓ Transcript loaded successfully", progress_callback)
            self._log_debug(f"Content stats: {char_count} chars, {word_count} words, {line_count} lines", progress_callback)
            self._log_info(f"=== TRANSCRIPT LOADING COMPLETED ===", progress_callback)

            result = {
                'success': True,
                'content': content,
                'file_size': file_size,
                'file_path': transcript_path,
                'character_count': char_count,
                'word_count': word_count,
                'line_count': line_count,
                'modified_time': modified_time,
                'read_time': read_time,
                **json_metadata
            }
            
            return result

        except UnicodeDecodeError as e:
            error_msg = f"Unicode decode error: {str(e)}"
            self._log_exception("Unicode decode error during file reading", e, progress_callback)
            self._log_debug(f"Try opening with different encoding. Error at position: {e.start}-{e.end}", progress_callback)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'error_type': 'UnicodeDecodeError',
                'error_details': str(e)
            }
        except PermissionError as e:
            error_msg = f"Permission denied accessing file: {str(e)}"
            self._log_exception("Permission error accessing file", e, progress_callback)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'error_type': 'PermissionError'
            }
        except Exception as e:
            error_msg = f"Error reading transcript file: {str(e)}"
            self._log_exception("Unexpected error during transcript loading", e, progress_callback)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }

    def _convert_json_to_text(self, json_data: Dict[str, Any], progress_callback: Optional[Callable] = None) -> str:
        """
        Convert JSON transcript data to text format for processing.
        
        Args:
            json_data: Parsed JSON data
            progress_callback: Optional callback for logging progress
            
        Returns:
            Formatted text content
        """
        self._log_debug("Converting JSON data to text format", progress_callback)
        
        try:
            # Handle different JSON structures
            if isinstance(json_data, list):
                # Array of transcripts or segments
                return self._process_json_array(json_data, progress_callback)
            elif isinstance(json_data, dict):
                # Object with transcript data
                return self._process_json_object(json_data, progress_callback)
            else:
                # Fallback: convert to string
                self._log_warning("Unexpected JSON structure, converting to string", progress_callback)
                return str(json_data) if json_data is not None else ""
                
        except Exception as e:
            self._log_error(f"Error converting JSON to text: {str(e)}", progress_callback)
            # Fallback to JSON pretty print
            try:
                return json.dumps(json_data, indent=2, ensure_ascii=False) if json_data is not None else ""
            except Exception:
                return str(json_data) if json_data is not None else ""

    def _process_json_array(self, json_array: List[Any], progress_callback: Optional[Callable] = None) -> str:
        """Process JSON array structure."""
        text_parts = []
        
        try:
            if not isinstance(json_array, list):
                return str(json_array) if json_array is not None else ""
                
            for i, item in enumerate(json_array):
                try:
                    if isinstance(item, dict):
                        # Look for common transcript fields
                        text_content = self._extract_text_from_object(item, i)
                        if text_content:
                            text_parts.append(text_content)
                    elif isinstance(item, str):
                        text_parts.append(f"==== Segment {i+1} ====\n{item}")
                    else:
                        text_parts.append(f"==== Item {i+1} ====\n{str(item) if item is not None else ''}")
                except Exception as e:
                    self._log_warning(f"Error processing array item {i}: {str(e)}", progress_callback)
                    continue
            
            self._log_debug(f"Processed {len(json_array)} items from JSON array", progress_callback)
            return "\n\n".join(text_parts)
            
        except Exception as e:
            self._log_error(f"Error processing JSON array: {str(e)}", progress_callback)
            return str(json_array) if json_array is not None else ""

    def _process_json_object(self, json_obj: Dict[str, Any], progress_callback: Optional[Callable] = None) -> str:
        """Process JSON object structure."""
        try:
            if not isinstance(json_obj, dict):
                return str(json_obj) if json_obj is not None else ""
                
            # Look for common transcript structures
            if 'transcripts' in json_obj and isinstance(json_obj['transcripts'], list):
                return self._process_json_array(json_obj['transcripts'], progress_callback)
            elif 'segments' in json_obj and isinstance(json_obj['segments'], list):
                return self._process_json_array(json_obj['segments'], progress_callback)
            elif 'recordings' in json_obj and isinstance(json_obj['recordings'], list):
                return self._process_json_array(json_obj['recordings'], progress_callback)
            
            # Check for direct text content
            text_content = self._extract_text_from_object(json_obj, 0)
            if text_content:
                return text_content
            
            # Fallback: process all key-value pairs
            text_parts = []
            for key, value in json_obj.items():
                try:
                    if isinstance(value, (str, int, float)):
                        text_parts.append(f"==== {key} ====\n{value}")
                    elif isinstance(value, list):
                        text_parts.append(f"==== {key} ====\n{self._process_json_array(value, progress_callback)}")
                    elif isinstance(value, dict):
                        text_parts.append(f"==== {key} ====\n{self._process_json_object(value, progress_callback)}")
                except Exception as e:
                    self._log_warning(f"Error processing object key {key}: {str(e)}", progress_callback)
                    continue
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            self._log_error(f"Error processing JSON object: {str(e)}", progress_callback)
            return str(json_obj) if json_obj is not None else ""

    def _extract_text_from_object(self, obj: Dict[str, Any], index: int) -> str:
        """Extract text content from a JSON object using common field names."""
        try:
            if not isinstance(obj, dict):
                return str(obj) if obj is not None else ""
                
            # Common field names for transcript content
            text_fields = ['text', 'transcript', 'content', 'speech', 'words', 'dialogue']
            filename_fields = ['filename', 'file', 'name', 'source', 'recording']
            
            # Find text content
            text_content = None
            for field in text_fields:
                if field in obj and obj[field] is not None:
                    text_content = str(obj[field])
                    break
            
            if not text_content:
                # Look for any string field that might contain text
                for key, value in obj.items():
                    if isinstance(value, str) and len(value) > 20:  # Assume longer strings are content
                        text_content = value
                        break
            
            # Find filename
            filename = None
            for field in filename_fields:
                if field in obj and obj[field] is not None:
                    filename = str(obj[field])
                    break
            
            if not filename:
                filename = f"segment_{index+1}.unknown"
            
            # Format as transcript segment
            if text_content:
                return f"==== {filename} ====\n{text_content}"
            
            return ""
            
        except Exception as e:
            self._log_warning(f"Error extracting text from object: {str(e)}", None)
            return ""

    def _analyze_json_structure(self, json_data: Dict[str, Any], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Analyze JSON structure for optimization hints."""
        structure_info = {
            'data_type': type(json_data).__name__,
            'top_level_keys': [],
            'has_timestamps': False,
            'has_segments': False,
            'estimated_segments': 0,
            'content_fields': []
        }
        
        try:
            if isinstance(json_data, dict):
                structure_info['top_level_keys'] = list(json_data.keys())
                
                # Check for timestamp information
                timestamp_indicators = ['time', 'timestamp', 'start', 'end', 'duration']
                for key in json_data.keys():
                    if any(indicator in key.lower() for indicator in timestamp_indicators):
                        structure_info['has_timestamps'] = True
                        break
                
                # Check for segment information
                segment_indicators = ['segments', 'transcripts', 'recordings', 'parts']
                for key in json_data.keys():
                    if key.lower() in segment_indicators and isinstance(json_data[key], list):
                        structure_info['has_segments'] = True
                        structure_info['estimated_segments'] = len(json_data[key])
                        break
                
                # Identify content fields
                content_indicators = ['text', 'transcript', 'content', 'speech', 'words']
                for key in json_data.keys():
                    if any(indicator in key.lower() for indicator in content_indicators):
                        structure_info['content_fields'].append(key)
                        
            elif isinstance(json_data, list):
                structure_info['estimated_segments'] = len(json_data)
                structure_info['has_segments'] = True
                
                # Analyze first item for structure
                if json_data and isinstance(json_data[0], dict):
                    first_item = json_data[0]
                    structure_info['top_level_keys'] = list(first_item.keys())
                    
                    timestamp_indicators = ['time', 'timestamp', 'start', 'end', 'duration']
                    for key in first_item.keys():
                        if any(indicator in key.lower() for indicator in timestamp_indicators):
                            structure_info['has_timestamps'] = True
                            break
                    
                    content_indicators = ['text', 'transcript', 'content', 'speech', 'words']
                    for key in first_item.keys():
                        if any(indicator in key.lower() for indicator in content_indicators):
                            structure_info['content_fields'].append(key)
            
            self._log_debug(f"JSON structure analysis: {structure_info}", progress_callback)
            return structure_info
            
        except Exception as e:
            self._log_warning(f"Error analyzing JSON structure: {str(e)}", progress_callback)
            return structure_info

    def segment_transcript(self, transcript_content: str, progress_callback: Optional[Callable] = None, 
                          json_metadata: Optional[Dict[str, Any]] = None) -> List[TranscriptSegment]:
        """
        Segment the combined transcript by recording delimiters with comprehensive logging and JSON optimization.
        
        Args:
            transcript_content: Full transcript content
            progress_callback: Optional callback for logging progress
            json_metadata: Optional JSON metadata for enhanced processing
            
        Returns:
            List of TranscriptSegment objects with detailed processing information
        """
        self._log_info("=== STARTING TRANSCRIPT SEGMENTATION ===", progress_callback)
        self._log_debug(f"Input content length: {len(transcript_content) if transcript_content else 0} characters", progress_callback)
        self._log_debug(f"JSON metadata provided: {json_metadata is not None}", progress_callback)
        
        try:
            # Check if we have JSON metadata for enhanced processing
            is_json = json_metadata and json_metadata.get('is_json', False)
            is_chunked_json = json_metadata and json_metadata.get('is_chunked', False)
            
            if is_chunked_json and json_metadata.get('original_json'):
                self._log_info("Using chunked JSON-optimized segmentation", progress_callback)
                self._log_debug(f"Chunked JSON structure: {json_metadata.get('json_structure', {})}", progress_callback)
                return self._segment_chunked_json_transcript(json_metadata.get('original_json', {}), progress_callback)
            elif is_json:
                self._log_info("Using JSON-optimized segmentation", progress_callback)
                self._log_debug(f"JSON structure: {json_metadata.get('json_structure', {})}", progress_callback)
                return self._segment_json_transcript(transcript_content, json_metadata, progress_callback)
            else:
                self._log_info("Using standard text segmentation", progress_callback)
                return self._segment_text_transcript(transcript_content, progress_callback)
        except Exception as e:
            self._log_exception("Critical error in segment_transcript", e, progress_callback)
            return []

    def _segment_text_transcript(self, transcript_content: str, progress_callback: Optional[Callable] = None) -> List[TranscriptSegment]:
        """
        Segment text-based transcript using the original method.
        
        Args:
            transcript_content: Full transcript content
            progress_callback: Optional callback for logging progress
            
        Returns:
            List of TranscriptSegment objects
        """
        self._log_info("=== STARTING TEXT TRANSCRIPT SEGMENTATION ===", progress_callback)
        start_time = datetime.now()
        segments = []
        
        try:
            if not transcript_content:
                self._log_error("Empty transcript content provided", progress_callback)
                return segments
                
            # Split by recording delimiters (e.g., "==== recording_001.mp3 ====")
            # Pattern matches various delimiter formats
            delimiter_pattern = r'={3,}\s*([^=\n]+?\.(?:mp3|wav|m4a|flac|aac|ogg|wma|mp4|avi|mov|mkv))\s*={3,}'
            
            self._log_debug(f"Using delimiter pattern: {delimiter_pattern}", progress_callback)
            self._log_debug(f"Input content length: {len(transcript_content)} characters", progress_callback)

            # Find all delimiters first for analysis
            self._log_debug("Searching for delimiter matches...", progress_callback)
            try:
                delimiter_matches = list(re.finditer(delimiter_pattern, transcript_content, flags=re.IGNORECASE))
                self._log_info(f"Found {len(delimiter_matches)} delimiter matches", progress_callback)
                
                for i, match in enumerate(delimiter_matches):
                    filename = match.group(1)
                    self._log_debug(f"  Delimiter {i+1}: {filename} at position {match.start()}-{match.end()}", progress_callback)
            except Exception as re_error:
                self._log_exception("Error during delimiter matching", re_error, progress_callback)
                raise

            # Split the content by the delimiters
            self._log_debug("Splitting content by delimiters...", progress_callback)
            try:
                parts = re.split(delimiter_pattern, transcript_content, flags=re.IGNORECASE)
                self._log_debug(f"Split into {len(parts)} parts", progress_callback)
            except Exception as split_error:
                self._log_exception("Error during content splitting", split_error, progress_callback)
                raise

            # Process the parts - every odd index is a filename, every even index is content
            current_filename = None
            segment_index = 0

            self._log_debug("Processing split parts...", progress_callback)
            for i, part in enumerate(parts):
                try:
                    part_length = len(part.strip()) if part else 0
                    self._log_debug(f"Processing part {i}: {part_length} characters", progress_callback)
                    
                    if i == 0:
                        # First part might be content before any delimiter
                        if part and part.strip():
                            content = part.strip()
                            segment = TranscriptSegment(
                                filename="unknown_recording",
                                content=content,
                                segment_index=segment_index
                            )
                            segments.append(segment)
                            self._log_info(f"Created header segment: {segment}", progress_callback)
                            segment_index += 1
                    elif i % 2 == 1:
                        # Odd indices are filenames from the regex capture groups
                        current_filename = part.strip() if part else None
                        self._log_debug(f"Found filename: {current_filename}", progress_callback)
                    else:
                        # Even indices (except 0) are content sections
                        if current_filename and part and part.strip():
                            content = part.strip()
                            try:
                                segment = TranscriptSegment(
                                    filename=current_filename,
                                    content=content,
                                    segment_index=segment_index
                                )
                                segments.append(segment)
                                self._log_info(f"Created segment: {segment}", progress_callback)
                                segment_index += 1
                            except Exception as segment_error:
                                self._log_exception(f"Error creating segment for {current_filename}", segment_error, progress_callback)
                                continue
                        elif current_filename and (not part or not part.strip()):
                            self._log_warning(f"Empty content for recording: {current_filename}", progress_callback)
                except Exception as part_error:
                    self._log_exception(f"Error processing part {i}", part_error, progress_callback)
                    continue

            # If no delimiters found, treat the entire content as one segment
            if not segments and transcript_content and transcript_content.strip():
                self._log_info("No delimiters found, creating single segment", progress_callback)
                try:
                    segment = TranscriptSegment(
                        filename="combined_transcript",
                        content=transcript_content.strip(),
                        segment_index=0
                    )
                    segments.append(segment)
                    self._log_info(f"Created single segment: {segment}", progress_callback)
                except Exception as single_segment_error:
                    self._log_exception("Error creating single segment", single_segment_error, progress_callback)

            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Log summary statistics
            total_words = sum(seg.word_count for seg in segments)
            avg_words = total_words / len(segments) if segments else 0
            
            self._log_info(f"✓ Text segmentation completed in {processing_time:.3f}s", progress_callback)
            self._log_info(f"Created {len(segments)} segments with {total_words} total words", progress_callback)
            self._log_debug(f"Average words per segment: {avg_words:.1f}", progress_callback)
            
            for i, segment in enumerate(segments):
                self._log_debug(f"  Segment {i+1}: {segment.filename} ({segment.word_count} words)", progress_callback)

            self._log_info("=== TEXT TRANSCRIPT SEGMENTATION COMPLETED ===", progress_callback)
            return segments
            
        except Exception as e:
            error_msg = f"Error during text transcript segmentation: {str(e)}"
            self._log_exception("Critical error in _segment_text_transcript", e, progress_callback)
            
            # Return empty list but log the error
            return []

    def _segment_json_transcript(self, transcript_content: str, json_metadata: Dict[str, Any], progress_callback: Optional[Callable] = None) -> List[TranscriptSegment]:
        """
        Segment JSON-based transcript using enhanced processing.
        
        Args:
            transcript_content: Converted text content from JSON
            json_metadata: JSON metadata for optimization
            progress_callback: Optional callback for logging progress
            
        Returns:
            List of TranscriptSegment objects with enhanced JSON processing
        """
        start_time = datetime.now()
        segments = []
        
        try:
            original_json = json_metadata.get('original_json', {})
            json_structure = json_metadata.get('json_structure', {})
            
            self._log_debug(f"JSON structure info: {json_structure}", progress_callback)
            
            # Use original JSON data for more accurate segmentation if available
            if original_json:
                segments = self._create_segments_from_json(original_json, json_structure, progress_callback)
            
            # Fallback to text-based segmentation if JSON processing fails
            if not segments:
                self._log_warning("JSON segmentation failed, falling back to text segmentation", progress_callback)
                return self._segment_text_transcript(transcript_content, progress_callback)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Log summary statistics
            total_words = sum(seg.word_count for seg in segments)
            avg_words = total_words / len(segments) if segments else 0
            
            self._log_info(f"✓ JSON segmentation completed in {processing_time:.3f}s", progress_callback)
            self._log_info(f"Created {len(segments)} segments with {total_words} total words", progress_callback)
            self._log_debug(f"Average words per segment: {avg_words:.1f}", progress_callback)
            
            # Enhanced logging for JSON segments
            for i, segment in enumerate(segments):
                timestamp_info = ""
                if hasattr(segment, 'detailed_segments') and segment.detailed_segments:
                    timestamp_info = f" (timestamped: {len(segment.detailed_segments)} segments)"
                self._log_debug(f"  JSON Segment {i+1}: {segment.filename} ({segment.word_count} words){timestamp_info}", progress_callback)

            return segments
            
        except Exception as e:
            error_msg = f"Error during JSON transcript segmentation: {str(e)}"
            self._log_error(error_msg, progress_callback)
            self._log_debug(f"Full traceback: {traceback.format_exc()}", progress_callback)
            
            # Fallback to text segmentation
            self._log_info("Falling back to text segmentation due to JSON error", progress_callback)
            return self._segment_text_transcript(transcript_content, progress_callback)

    def _create_segments_from_json(self, json_data: Any, structure_info: Dict[str, Any], progress_callback: Optional[Callable] = None) -> List[TranscriptSegment]:
        """
        Create transcript segments directly from JSON data for better accuracy.
        
        Args:
            json_data: Original JSON data
            structure_info: Analyzed structure information
            progress_callback: Optional callback for logging progress
            
        Returns:
            List of TranscriptSegment objects
        """
        segments = []
        
        try:
            if isinstance(json_data, list):
                # Process array of segments
                for i, item in enumerate(json_data):
                    try:
                        segment = self._create_segment_from_json_item(item, i, structure_info, progress_callback)
                        if segment:
                            segments.append(segment)
                    except Exception as e:
                        self._log_warning(f"Error creating segment from array item {i}: {str(e)}", progress_callback)
                        continue
                        
            elif isinstance(json_data, dict):
                # Check for known array fields
                array_fields = ['transcripts', 'segments', 'recordings', 'parts']
                found_array = False
                
                for field in array_fields:
                    if field in json_data and isinstance(json_data[field], list):
                        self._log_debug(f"Found JSON array field: {field}", progress_callback)
                        for i, item in enumerate(json_data[field]):
                            try:
                                segment = self._create_segment_from_json_item(item, i, structure_info, progress_callback)
                                if segment:
                                    segments.append(segment)
                            except Exception as e:
                                self._log_warning(f"Error creating segment from {field}[{i}]: {str(e)}", progress_callback)
                                continue
                        found_array = True
                        break
                
                # If no array found, treat as single segment
                if not found_array:
                    try:
                        segment = self._create_segment_from_json_item(json_data, 0, structure_info, progress_callback)
                        if segment:
                            segments.append(segment)
                    except Exception as e:
                        self._log_warning(f"Error creating single segment from JSON object: {str(e)}", progress_callback)
            
            self._log_info(f"Created {len(segments)} segments from JSON data", progress_callback)
            return segments
            
        except Exception as e:
            self._log_error(f"Error creating segments from JSON: {str(e)}", progress_callback)
            return []

    def _create_segment_from_json_item(self, item: Any, index: int, structure_info: Dict[str, Any], progress_callback: Optional[Callable] = None) -> Optional[TranscriptSegment]:
        """
        Create a single TranscriptSegment from a JSON item.
        
        Args:
            item: JSON item (dict, string, etc.)
            index: Index of this item
            structure_info: Structure analysis information
            progress_callback: Optional callback for logging
            
        Returns:
            TranscriptSegment object or None if creation fails
        """
        try:
            if isinstance(item, str):
                # Simple string segment
                return TranscriptSegment(
                    filename=f"json_segment_{index+1}",
                    content=item,
                    segment_index=index
                )
            
            elif isinstance(item, dict):
                # Extract fields from dictionary
                filename = self._extract_filename_from_json(item, index)
                content = self._extract_content_from_json(item)
                
                if not content:
                    self._log_warning(f"No content found in JSON item {index}", progress_callback)
                    return None
                
                # Extract timestamped segments if available
                detailed_segments = self._extract_timestamped_segments(item, progress_callback)
                
                return TranscriptSegment(
                    filename=filename,
                    content=content,
                    segment_index=index,
                    detailed_segments=detailed_segments
                )
            
            else:
                # Convert other types to string
                return TranscriptSegment(
                    filename=f"json_item_{index+1}",
                    content=str(item) if item is not None else "",
                    segment_index=index
                )
                
        except Exception as e:
            self._log_error(f"Error creating segment from JSON item {index}: {str(e)}", progress_callback)
            return None

    def _extract_filename_from_json(self, item: Dict[str, Any], index: int) -> str:
        """Extract filename from JSON item."""
        try:
            if not isinstance(item, dict):
                return f"json_item_{index+1}.json"
                
            filename_fields = ['filename', 'file', 'name', 'source', 'recording', 'audio_file', 'path']
            
            for field in filename_fields:
                if field in item and item[field] is not None:
                    return str(item[field])
            
            # Fallback filename
            return f"json_recording_{index+1}.json"
            
        except Exception:
            return f"json_item_{index+1}.json"

    def _extract_content_from_json(self, item: Dict[str, Any]) -> str:
        """Extract text content from JSON item."""
        try:
            if not isinstance(item, dict):
                return str(item) if item is not None else ""
                
            content_fields = ['text', 'transcript', 'content', 'speech', 'words', 'dialogue']
            
            for field in content_fields:
                if field in item and item[field] is not None:
                    content = item[field]
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        # Join list of words/segments
                        return ' '.join(str(x) for x in content if x is not None)
                    else:
                        return str(content)
            
            # Look for any substantial string field
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 20:
                    return value
            
            return ""
            
        except Exception:
            return ""

    def _extract_timestamped_segments(self, item: Dict[str, Any], progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Extract timestamped segments from JSON item if available."""
        timestamped_segments = []
        
        try:
            if not isinstance(item, dict):
                return timestamped_segments
                
            # Look for common timestamp segment fields
            segment_fields = ['segments', 'words', 'timestamps', 'chunks']
            
            for field in segment_fields:
                if field in item and isinstance(item[field], list):
                    for segment in item[field]:
                        if isinstance(segment, dict):
                            try:
                                # Extract timestamp and text
                                timestamp_segment = {}
                                
                                # Look for start time
                                start_fields = ['start', 'start_time', 'begin', 'from']
                                for start_field in start_fields:
                                    if start_field in segment and segment[start_field] is not None:
                                        try:
                                            timestamp_segment['start'] = float(segment[start_field])
                                            break
                                        except (ValueError, TypeError):
                                            continue
                                
                                # Look for end time
                                end_fields = ['end', 'end_time', 'to', 'until']
                                for end_field in end_fields:
                                    if end_field in segment and segment[end_field] is not None:
                                        try:
                                            timestamp_segment['end'] = float(segment[end_field])
                                            break
                                        except (ValueError, TypeError):
                                            continue
                                
                                # Look for text content
                                text_fields = ['text', 'word', 'content', 'transcript']
                                for text_field in text_fields:
                                    if text_field in segment and segment[text_field] is not None:
                                        timestamp_segment['text'] = str(segment[text_field])
                                        break
                                
                                # Add segment if it has required fields
                                if 'text' in timestamp_segment and ('start' in timestamp_segment or 'end' in timestamp_segment):
                                    timestamped_segments.append(timestamp_segment)
                                    
                            except Exception as e:
                                self._log_warning(f"Error processing timestamp segment: {str(e)}", progress_callback)
                                continue
                    
                    if timestamped_segments:
                        self._log_debug(f"Extracted {len(timestamped_segments)} timestamped segments", progress_callback)
                        break
            
            return timestamped_segments
            
        except Exception as e:
            self._log_warning(f"Error extracting timestamped segments: {str(e)}", progress_callback)
            return []

    def build_analysis_prompt(self, case_facts: str, segment: TranscriptSegment, progress_callback: Optional[Callable] = None) -> str:
        """
        Build the enhanced analysis prompt for crime-relevant content identification with JSON optimization.
        
        Args:
            case_facts: User-provided case facts
            segment: Transcript segment to analyze
            progress_callback: Optional callback for logging progress
            
        Returns:
            Formatted prompt string with detailed construction logging and JSON optimization
        """
        self._log_debug(f"Building crime analysis prompt for {segment.filename}", progress_callback)
        
        try:
            # Validate inputs
            case_facts = case_facts or ""
            if not case_facts.strip():
                self._log_warning("Case facts are empty", progress_callback)
                
            if not segment or not segment.content.strip():
                self._log_warning(f"Segment content is empty for {getattr(segment, 'filename', 'unknown')}", progress_callback)

            # Detect if this is a JSON-originated segment for optimized prompting
            is_json_segment = (hasattr(segment, 'filename') and 
                             (segment.filename.startswith('json_') or '.json' in segment.filename.lower()))
            
            # Safely check for timestamps
            has_timestamps = (hasattr(segment, 'detailed_segments') and 
                            isinstance(segment.detailed_segments, list) and 
                            len(segment.detailed_segments) > 0)
            
            if is_json_segment:
                self._log_debug(f"JSON segment detected, using optimized prompt", progress_callback)
            
            if has_timestamps:
                self._log_debug(f"Timestamped segments available: {len(segment.detailed_segments)}", progress_callback)

            # Build base prompt with JSON-aware instructions
            prompt = f"""CASE FACTS:
{case_facts.strip()}

TRANSCRIPT FROM {getattr(segment, 'filename', 'unknown')}:
{getattr(segment, 'content', '')}"""

            # Add timestamp information if available
            if has_timestamps:
                prompt += f"""

AVAILABLE TIMESTAMP SEGMENTS ({len(segment.detailed_segments)} segments):
"""
                # Safely process timestamp segments
                for i, ts_segment in enumerate(segment.detailed_segments[:5]):  # Show first 5 segments
                    try:
                        start_time = ts_segment.get('start', 'N/A') if isinstance(ts_segment, dict) else 'N/A'
                        end_time = ts_segment.get('end', 'N/A') if isinstance(ts_segment, dict) else 'N/A'
                        text = ts_segment.get('text', '') if isinstance(ts_segment, dict) else str(ts_segment)
                        
                        # Safely truncate text
                        if isinstance(text, str) and len(text) > 100:
                            text = text[:100] + "..."
                        
                        prompt += f"{i+1}. [{start_time}s - {end_time}s]: {text}\n"
                    except Exception as e:
                        self._log_warning(f"Error processing timestamp segment {i}: {str(e)}", progress_callback)
                        continue
                
                if len(segment.detailed_segments) > 5:
                    prompt += f"... and {len(segment.detailed_segments) - 5} more timestamped segments\n"

            # Enhanced analysis instructions for JSON data
            if is_json_segment:
                prompt += f"""

ENHANCED JSON ANALYSIS INSTRUCTIONS:
This transcript comes from structured JSON data. Pay special attention to:
1. Structured information that may be more precise than regular speech transcripts
2. Metadata fields that might contain relevant case information
3. Timestamped segments that can provide exact timing for evidence
4. Multiple speakers or sources if indicated in the data structure
5. Technical details that might be preserved better in JSON format

ANALYSIS FOCUS:"""
            else:
                prompt += f"""

ANALYSIS INSTRUCTIONS:"""

            prompt += f"""
1. Carefully analyze this transcript for ANY content that may be relevant to the crime or investigation described in the case facts
2. Look for:
   - Direct evidence related to the crime
   - Witness statements or observations
   - Suspect behavior or statements
   - Timeline information
   - Physical evidence descriptions
   - Names, locations, or other identifying details
   - Contradictions or inconsistencies
   - Suspicious activities or behavior"""

            # Add timestamp-specific instructions if available
            if has_timestamps:
                prompt += f"""
   - Precise timing information for critical events
   - Sequence of events with exact timestamps
   - Duration of suspicious activities or conversations"""

            prompt += f"""

3. For EACH relevant section you find:
   - Quote the EXACT text from the transcript
   - Explain WHY it's relevant to the case
   - Estimate the approximate timestamp within this recording where this content appears"""

            # Enhanced timestamp instructions for JSON
            if has_timestamps:
                prompt += f"""
   - Use the provided timestamp segments for more accurate timing
   - Reference specific timestamp ranges when available"""

            prompt += f"""

RESPONSE FORMAT:
If relevant content is found, respond with:
RELEVANT SECTIONS:
1. Text: "[exact quote from transcript]"
   Relevance: [explanation of why this is relevant]
   Estimated Time: [approximate time in recording when this occurs]"""

            if has_timestamps:
                prompt += f"""
   Timestamp Reference: [reference to timestamp segment if applicable]"""

            prompt += f"""

2. [additional sections if found]

If no relevant content is found, respond with:
NO RELEVANT CONTENT FOUND

Be thorough and consider all possible connections to the case facts."""

            # Add JSON-specific guidance
            if is_json_segment:
                prompt += f"""

JSON DATA GUIDANCE:
- This data may contain more structured and precise information than typical audio transcripts
- Look for technical details, exact quotes, or structured data that might be case-relevant
- Consider that timestamps (if present) are likely more accurate than estimated times
- Pay attention to any metadata or structured fields that might contain evidence"""

            prompt_length = len(prompt)
            prompt_words = len(prompt.split())
            
            analysis_type = "JSON-optimized" if is_json_segment else "standard"
            timestamp_info = f" with {len(segment.detailed_segments) if has_timestamps else 0} timestamps"
            
            self._log_debug(f"{analysis_type} prompt constructed{timestamp_info}: {prompt_length} chars, {prompt_words} words", progress_callback)
            self._log_debug(f"Case facts length: {len(case_facts)} chars", progress_callback)
            self._log_debug(f"Transcript content length: {len(getattr(segment, 'content', ''))} chars", progress_callback)

            return prompt
            
        except Exception as e:
            error_msg = f"Error building prompt for {getattr(segment, 'filename', 'unknown')}: {str(e)}"
            self._log_error(error_msg, progress_callback)
            # Return a basic prompt as fallback
            filename = getattr(segment, 'filename', 'unknown') if segment else 'unknown'
            return f"Analyze this transcript segment from {filename} for crime-relevant content."

    def analyze_segment(self, segment: TranscriptSegment, case_facts: str,
                       model_name: str = "mistral",
                       progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Analyze a single transcript segment with AI using comprehensive logging.
        
        Args:
            segment: Transcript segment to analyze
            case_facts: Case facts for context
            model_name: AI model to use
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with detailed analysis results and processing metrics
        """
        self._log_info(f"=== STARTING SEGMENT ANALYSIS ===", progress_callback)
        segment_start_time = datetime.now()
        
        # Validate inputs
        try:
            if not segment:
                error_msg = "No segment provided for analysis"
                self._log_error(error_msg, progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'segment': None,
                    'ai_response': '',
                    'processing_time': 0
                }
                
            self._log_info(f"Analyzing segment: {segment.filename}", progress_callback)
            self._log_debug(f"Segment details: {segment.word_count} words, index {segment.segment_index}", progress_callback)
            self._log_debug(f"Using model: {model_name}", progress_callback)
            self._log_debug(f"Case facts length: {len(case_facts) if case_facts else 0} characters", progress_callback)
            
            if self.is_cancelled:
                self._log_warning("Analysis cancelled by user", progress_callback)
                return {
                    'success': False,
                    'error': 'Analysis cancelled',
                    'segment': segment,
                    'ai_response': '',
                    'processing_time': 0,
                    'cancelled': True
                }

            # Build the prompt
            self._log_debug("Building analysis prompt...", progress_callback)
            prompt_start_time = datetime.now()
            try:
                prompt = self.build_analysis_prompt(case_facts, segment, progress_callback)
                prompt_build_time = (datetime.now() - prompt_start_time).total_seconds()
                
                self._log_debug(f"Prompt built in {prompt_build_time:.3f}s", progress_callback)
                
                # Safe prompt length logging
                try:
                    prompt_len = len(prompt) if prompt else 0
                    self._log_debug(f"Prompt length: {prompt_len} characters", progress_callback)
                except Exception as len_error:
                    self._log_error(f"Error getting prompt length: {str(len_error)}", progress_callback)
                    self._log_debug(f"Prompt type: {type(prompt)}, prompt value: {str(prompt)[:100] if prompt else 'None'}", progress_callback)
            except Exception as prompt_error:
                self._log_exception("Error building analysis prompt", prompt_error, progress_callback)
                return {
                    'success': False,
                    'error': f'Prompt building failed: {str(prompt_error)}',
                    'segment': segment,
                    'ai_response': '',
                    'processing_time': (datetime.now() - segment_start_time).total_seconds()
                }

            # Send to AI model
            self._log_info("Sending request to AI model...", progress_callback)
            ai_start_time = datetime.now()
            try:
                ai_result = self.ollama_client.generate_response(
                    model=model_name,
                    prompt=prompt,
                    progress_callback=progress_callback
                )
                ai_processing_time = (datetime.now() - ai_start_time).total_seconds()
                self._log_debug(f"AI request completed in {ai_processing_time:.1f}s", progress_callback)
            except Exception as ai_error:
                self._log_exception("Error during AI request", ai_error, progress_callback)
                ai_processing_time = (datetime.now() - ai_start_time).total_seconds()
                return {
                    'success': False,
                    'error': f'AI request failed: {str(ai_error)}',
                    'segment': segment,
                    'ai_response': '',
                    'processing_time': (datetime.now() - segment_start_time).total_seconds(),
                    'ai_processing_time': ai_processing_time
                }
            
            total_segment_time = (datetime.now() - segment_start_time).total_seconds()
            
            if ai_result and ai_result.get('success'):
                response_length = len(ai_result.get('response', ''))
                response_words = len(ai_result.get('response', '').split())
                
                self._log_info(f"✓ Segment analysis completed successfully for {segment.filename}", progress_callback)
                self._log_debug(f"AI response: {response_length} chars, {response_words} words", progress_callback)
                self._log_debug(f"Total segment processing: {total_segment_time:.1f}s", progress_callback)
                
                # Analyze response for relevance indicators
                self._log_debug("Analyzing response relevance...", progress_callback)
                try:
                    relevance_score = self._analyze_response_relevance(ai_result.get('response', ''), progress_callback)
                except Exception as relevance_error:
                    self._log_exception("Error analyzing response relevance", relevance_error, progress_callback)
                    relevance_score = {'error': str(relevance_error)}
                
                # Extract crime-relevant sections with timestamps
                self._log_debug("Extracting crime-relevant sections...", progress_callback)
                try:
                    crime_sections = self.extract_crime_relevant_sections(
                        ai_result.get('response', ''), segment, progress_callback
                    )
                except Exception as extraction_error:
                    self._log_exception("Error extracting crime-relevant sections", extraction_error, progress_callback)
                    crime_sections = []
                
                self._log_info(f"=== SEGMENT ANALYSIS COMPLETED SUCCESSFULLY ===", progress_callback)
                
                return {
                    'success': True,
                    'segment': segment,
                    'ai_response': ai_result.get('response', ''),
                    'processing_time': total_segment_time,
                    'ai_processing_time': ai_processing_time,
                    'prompt_build_time': prompt_build_time,
                    'model_used': ai_result.get('model_used', model_name),
                    'response_length': response_length,
                    'response_words': response_words,
                    'relevance_score': relevance_score,
                    'crime_relevant_sections': crime_sections,
                    'has_crime_content': len(crime_sections) > 0,
                    'ai_metrics': {
                        'eval_count': ai_result.get('eval_count'),
                        'eval_duration': ai_result.get('eval_duration'),
                        'response_time': ai_result.get('processing_time')
                    }
                }
            else:
                error_msg = ai_result.get('error', 'Unknown AI error') if ai_result else 'No AI response received'
                self._log_error(f"AI analysis failed for {segment.filename}: {error_msg}", progress_callback)
                
                return {
                    'success': False,
                    'error': error_msg,
                    'segment': segment,
                    'ai_response': '',
                    'processing_time': total_segment_time,
                    'ai_processing_time': ai_processing_time,
                    'model_used': model_name,
                    'ai_error_details': ai_result
                }

        except Exception as e:
            error_msg = f"Error analyzing segment {getattr(segment, 'filename', 'unknown')}: {str(e)}"
            self._log_exception("Critical error in analyze_segment", e, progress_callback)
            
            total_time = (datetime.now() - segment_start_time).total_seconds()
            
            return {
                'success': False,
                'error': error_msg,
                'segment': segment,
                'ai_response': '',
                'processing_time': total_time,
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }

    def _analyze_response_relevance(self, ai_response: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Analyze AI response to determine relevance with detailed logging.
        
        Args:
            ai_response: AI model response
            progress_callback: Optional callback for logging progress
            
        Returns:
            Dict with relevance analysis results
        """
        try:
            response_lower = ai_response.lower()
            
            positive_indicators = [
                "yes", "relevant", "contains", "mentions", "discusses",
                "refers to", "relates to", "quote:", "important", "significant",
                "shows", "indicates", "evidence", "supports"
            ]
            negative_indicators = [
                "no", "not relevant", "does not contain", "no mention",
                "not related", "nothing relevant", "no relevant", "unrelated",
                "irrelevant", "no evidence"
            ]

            # Count indicators
            positive_count = sum(1 for indicator in positive_indicators if indicator in response_lower)
            negative_count = sum(1 for indicator in negative_indicators if indicator in response_lower)
            
            # Look for quoted content
            quote_count = response_lower.count('"') // 2  # Pairs of quotes
            
            # Calculate relevance score
            relevance_score = positive_count - negative_count + (quote_count * 2)
            is_relevant = relevance_score > 0
            
            self._log_debug(f"Relevance analysis: +{positive_count} positive, -{negative_count} negative, {quote_count} quotes", progress_callback)
            self._log_debug(f"Relevance score: {relevance_score} ({'relevant' if is_relevant else 'not relevant'})", progress_callback)
            
            return {
                'is_relevant': is_relevant,
                'relevance_score': relevance_score,
                'positive_indicators': positive_count,
                'negative_indicators': negative_count,
                'quote_count': quote_count,
                'response_length': len(ai_response)
            }
            
        except Exception as e:
            self._log_warning(f"Error analyzing response relevance: {str(e)}", progress_callback)
            return {
                'is_relevant': None,
                'relevance_score': 0,
                'error': str(e)
            }

    def extract_crime_relevant_sections(self, ai_response: str, segment: TranscriptSegment, 
                                      progress_callback: Optional[Callable] = None) -> List[CrimeRelevantSection]:
        """
        Extract crime-relevant sections from AI response with timestamp estimation and JSON optimization.
        
        Args:
            ai_response: AI model response containing relevant sections
            segment: Original transcript segment
            progress_callback: Optional callback for logging progress
            
        Returns:
            List of CrimeRelevantSection objects with enhanced timestamp accuracy for JSON data
        """
        relevant_sections = []
        
        try:
            self._log_debug(f"Extracting crime-relevant sections from {segment.filename}", progress_callback)
            
            # Check if this is a JSON segment with enhanced capabilities
            is_json_segment = segment.filename.startswith('json_') or '.json' in segment.filename.lower()
            has_timestamps = hasattr(segment, 'detailed_segments') and segment.detailed_segments
            
            if is_json_segment:
                self._log_debug("JSON segment detected, using enhanced extraction", progress_callback)
            if has_timestamps:
                self._log_debug(f"Timestamp data available: {len(segment.detailed_segments)} segments", progress_callback)
            
            # Check if response indicates no relevant content
            if "NO RELEVANT CONTENT FOUND" in ai_response.upper():
                self._log_debug("AI response indicates no relevant content found", progress_callback)
                return relevant_sections
            
            # Look for structured relevant sections
            lines = ai_response.split('\n')
            current_section = {}
            in_relevant_sections = False
            
            for line in lines:
                line = line.strip()
                
                if "RELEVANT SECTIONS:" in line.upper():
                    in_relevant_sections = True
                    continue
                
                if not in_relevant_sections:
                    continue
                
                # Parse structured format
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    # Save previous section if exists
                    if current_section.get('text'):
                        crime_section = self._create_enhanced_crime_section(current_section, segment, has_timestamps)
                        if crime_section:
                            relevant_sections.append(crime_section)
                    
                    # Start new section
                    current_section = {}
                    if 'Text:' in line:
                        text_part = line.split('Text:', 1)[1].strip()
                        current_section['text'] = text_part.strip('"[]')
                
                elif line.startswith('Text:'):
                    current_section['text'] = line.split('Text:', 1)[1].strip().strip('"[]')
                
                elif line.startswith('Relevance:'):
                    current_section['relevance'] = line.split('Relevance:', 1)[1].strip()
                
                elif line.startswith('Estimated Time:'):
                    time_str = line.split('Estimated Time:', 1)[1].strip()
                    current_section['estimated_time'] = self._parse_timestamp(time_str)
                
                elif line.startswith('Timestamp Reference:') and has_timestamps:
                    # Enhanced for JSON: extract timestamp reference
                    ref_str = line.split('Timestamp Reference:', 1)[1].strip()
                    current_section['timestamp_reference'] = ref_str
            
            # Add last section if exists
            if current_section.get('text'):
                crime_section = self._create_enhanced_crime_section(current_section, segment, has_timestamps)
                if crime_section:
                    relevant_sections.append(crime_section)
            
            # Fallback: look for quoted text if structured format not found
            if not relevant_sections and '"' in ai_response:
                self._log_debug("Structured format not found, trying fallback extraction", progress_callback)
                relevant_sections = self._extract_quoted_sections_enhanced(ai_response, segment, has_timestamps)
            
            # Post-process for JSON segments with timestamp accuracy
            if is_json_segment and has_timestamps and relevant_sections:
                relevant_sections = self._enhance_timestamps_from_json(relevant_sections, segment, progress_callback)
            
            self._log_info(f"Extracted {len(relevant_sections)} crime-relevant sections from {segment.filename}", progress_callback)
            
            for i, section in enumerate(relevant_sections):
                timestamp_info = f"{section.get_timestamp_range()}"
                if hasattr(section, 'accuracy_level'):
                    timestamp_info += f" ({section.accuracy_level} accuracy)"
                self._log_debug(f"  Section {i+1}: {timestamp_info} - {section.text[:50]}...", progress_callback)
            
            return relevant_sections
            
        except Exception as e:
            error_msg = f"Error extracting crime-relevant sections: {str(e)}"
            self._log_error(error_msg, progress_callback)
            return relevant_sections

    def _create_enhanced_crime_section(self, section_data: Dict[str, Any], segment: TranscriptSegment, has_timestamps: bool) -> Optional[CrimeRelevantSection]:
        """Create a CrimeRelevantSection with enhanced JSON and timestamp support."""
        try:
            if not isinstance(section_data, dict) or not segment:
                return None
                
            text = section_data.get('text', '')
            relevance = section_data.get('relevance', 'Crime-relevant content identified')
            estimated_time = section_data.get('estimated_time', 0.0)
            timestamp_reference = section_data.get('timestamp_reference', '')
            
            # Enhanced timestamp resolution for JSON segments
            if has_timestamps and hasattr(segment, 'detailed_segments') and isinstance(segment.detailed_segments, list):
                # Try to find more accurate timestamp from reference
                if timestamp_reference:
                    accurate_time = self._resolve_timestamp_reference(timestamp_reference, segment.detailed_segments)
                    if accurate_time is not None:
                        estimated_time = accurate_time
                
                # Try to find timestamp by matching text content
                if estimated_time == 0.0:
                    text_match_time = self._find_timestamp_by_text_match(text, segment.detailed_segments)
                    if text_match_time is not None:
                        estimated_time = text_match_time
            
            # Estimate end time (assume 30 seconds or based on text length)
            if isinstance(text, str):
                text_words = len(text.split())
                estimated_duration = max(5.0, min(60.0, text_words * 0.5))  # 0.5 seconds per word, 5-60 sec range
            else:
                estimated_duration = 30.0
                
            end_time = estimated_time + estimated_duration
            
            crime_section = CrimeRelevantSection(
                filename=getattr(segment, 'filename', 'unknown'),
                text=text,
                start_time=estimated_time,
                end_time=end_time,
                relevance_explanation=relevance
            )
            
            # Add accuracy indicator for JSON segments
            if has_timestamps:
                crime_section.accuracy_level = "high" if timestamp_reference else "medium"
            else:
                crime_section.accuracy_level = "estimated"
            
            return crime_section
            
        except Exception as e:
            return None

    def _resolve_timestamp_reference(self, reference: str, detailed_segments: List[Dict[str, Any]]) -> Optional[float]:
        """Resolve timestamp reference to actual time from JSON segments."""
        try:
            if not reference or not isinstance(detailed_segments, list):
                return None
                
            # Look for segment number references
            import re
            segment_match = re.search(r'segment\s*(\d+)', reference.lower())
            if segment_match:
                try:
                    segment_idx = int(segment_match.group(1)) - 1  # Convert to 0-based index
                    if 0 <= segment_idx < len(detailed_segments):
                        segment = detailed_segments[segment_idx]
                        if isinstance(segment, dict) and 'start' in segment:
                            return float(segment['start'])
                except (ValueError, TypeError, IndexError):
                    pass
            
            # Look for time references
            time_match = re.search(r'(\d+\.?\d*)\s*s', reference)
            if time_match:
                try:
                    return float(time_match.group(1))
                except (ValueError, TypeError):
                    pass
            
            return None
            
        except Exception:
            return None

    def _find_timestamp_by_text_match(self, target_text: str, detailed_segments: List[Dict[str, Any]]) -> Optional[float]:
        """Find timestamp by matching text content in detailed segments."""
        try:
            if not target_text or not isinstance(detailed_segments, list):
                return None
                
            target_lower = target_text.lower()
            
            # Look for exact matches first
            for segment in detailed_segments:
                if not isinstance(segment, dict):
                    continue
                    
                segment_text = segment.get('text', '')
                if isinstance(segment_text, str) and segment_text and target_lower in segment_text.lower():
                    start_time = segment.get('start')
                    if start_time is not None:
                        try:
                            return float(start_time)
                        except (ValueError, TypeError):
                            continue
            
            # Look for partial matches (more than 50% overlap)
            target_words = set(target_lower.split()) if target_lower else set()
            if not target_words:
                return None
                
            best_match_time = None
            best_match_score = 0
            
            for segment in detailed_segments:
                if not isinstance(segment, dict):
                    continue
                    
                segment_text = segment.get('text', '')
                if isinstance(segment_text, str) and segment_text:
                    segment_words = set(segment_text.lower().split())
                    if segment_words:
                        overlap = len(target_words.intersection(segment_words))
                        score = overlap / len(target_words)
                        if score > 0.5 and score > best_match_score:
                            start_time = segment.get('start')
                            if start_time is not None:
                                try:
                                    best_match_score = score
                                    best_match_time = float(start_time)
                                except (ValueError, TypeError):
                                    continue
            
            return best_match_time
            
        except Exception:
            return None

    def _extract_quoted_sections_enhanced(self, ai_response: str, segment: TranscriptSegment, has_timestamps: bool) -> List[CrimeRelevantSection]:
        """Enhanced fallback method to extract quoted text with JSON timestamp support."""
        sections = []
        
        # Find all quoted text
        import re
        quoted_pattern = r'"([^"]+)"'
        quotes = re.findall(quoted_pattern, ai_response)
        
        for i, quote in enumerate(quotes):
            if len(quote.strip()) > 10:  # Only consider substantial quotes
                # Enhanced timestamp estimation for JSON segments
                if has_timestamps and segment.detailed_segments:
                    estimated_time = self._find_timestamp_by_text_match(quote, segment.detailed_segments)
                    if estimated_time is None:
                        estimated_time = self._estimate_timestamp_from_text(quote, segment.content)
                else:
                    estimated_time = self._estimate_timestamp_from_text(quote, segment.content)
                
                crime_section = CrimeRelevantSection(
                    filename=segment.filename,
                    text=quote,
                    start_time=estimated_time,
                    end_time=estimated_time + 30.0,  # Default 30 second duration
                    relevance_explanation="Quoted content identified as relevant"
                )
                
                # Add accuracy indicator
                if has_timestamps:
                    crime_section.accuracy_level = "medium"
                else:
                    crime_section.accuracy_level = "estimated"
                
                sections.append(crime_section)
        
        return sections

    def _enhance_timestamps_from_json(self, sections: List[CrimeRelevantSection], segment: TranscriptSegment, progress_callback: Optional[Callable] = None) -> List[CrimeRelevantSection]:
        """Enhance timestamp accuracy for JSON segments using detailed timestamp data."""
        try:
            if not hasattr(segment, 'detailed_segments') or not segment.detailed_segments:
                return sections
            
            enhanced_sections = []
            for section in sections:
                # Try to find more accurate timestamps
                better_start_time = self._find_timestamp_by_text_match(section.text, segment.detailed_segments)
                
                if better_start_time is not None:
                    # Calculate better end time based on text length and segment boundaries
                    better_end_time = self._calculate_end_time_from_segments(
                        section.text, better_start_time, segment.detailed_segments
                    )
                    
                    enhanced_section = CrimeRelevantSection(
                        filename=section.filename,
                        text=section.text,
                        start_time=better_start_time,
                        end_time=better_end_time,
                        relevance_explanation=section.relevance_explanation,
                        audio_file_path=section.audio_file_path
                    )
                    enhanced_section.accuracy_level = "high"
                    enhanced_sections.append(enhanced_section)
                    
                    self._log_debug(f"Enhanced timestamp accuracy for section: {better_start_time:.1f}s", progress_callback)
                else:
                    enhanced_sections.append(section)
            
            return enhanced_sections
            
        except Exception as e:
            self._log_warning(f"Error enhancing timestamps: {str(e)}", progress_callback)
            return sections

    def _calculate_end_time_from_segments(self, text: str, start_time: float, detailed_segments: List[Dict[str, Any]]) -> float:
        """Calculate more accurate end time using segment boundaries."""
        try:
            text_words = len(text.split())
            
            # Find the segment containing the start time
            start_segment = None
            for segment in detailed_segments:
                seg_start = segment.get('start', 0.0)
                seg_end = segment.get('end', seg_start + 1.0)
                if seg_start <= start_time <= seg_end:
                    start_segment = segment
                    break
            
            if start_segment:
                # Use segment boundary as baseline
                seg_end = start_segment.get('end', start_time + 30.0)
                
                # Estimate based on text length but constrain to reasonable bounds
                word_duration = max(5.0, min(60.0, text_words * 0.5))
                estimated_end = start_time + word_duration
                
                # Use the minimum of segment boundary and estimated duration
                return min(estimated_end, seg_end + 5.0)  # Allow 5s buffer
            
            # Fallback to text-based estimation
            return start_time + max(5.0, min(60.0, text_words * 0.5))
            
        except Exception:
            return start_time + 30.0  # Default fallback

    def _parse_timestamp(self, time_str: str) -> float:
        """Parse timestamp string to seconds."""
        try:
            if not isinstance(time_str, str):
                return 0.0
                
            time_str = time_str.lower().strip()
            
            # Remove common words
            for word in ['approximately', 'around', 'about', 'at', 'time']:
                time_str = time_str.replace(word, '').strip()
            
            # Look for time patterns
            import re
            
            # Pattern: MM:SS or HH:MM:SS
            time_pattern = r'(\d{1,2}):(\d{2})(?::(\d{2}))?'
            match = re.search(time_pattern, time_str)
            if match:
                try:
                    minutes = int(match.group(1))
                    seconds = int(match.group(2))
                    hours = int(match.group(3)) if match.group(3) else 0
                    
                    if hours > 0:  # HH:MM:SS format
                        return hours * 3600 + minutes * 60 + seconds
                    else:  # MM:SS format
                        return minutes * 60 + seconds
                except (ValueError, TypeError):
                    pass
            
            # Pattern: "X minutes Y seconds"
            minute_pattern = r'(\d+)\s*(?:minute|min)'
            second_pattern = r'(\d+)\s*(?:second|sec)'
            
            minutes = 0
            seconds = 0
            
            min_match = re.search(minute_pattern, time_str)
            if min_match:
                try:
                    minutes = int(min_match.group(1))
                except (ValueError, TypeError):
                    pass
            
            sec_match = re.search(second_pattern, time_str)
            if sec_match:
                try:
                    seconds = int(sec_match.group(1))
                except (ValueError, TypeError):
                    pass
            
            if minutes > 0 or seconds > 0:
                return minutes * 60 + seconds
            
            # Pattern: just numbers (assume seconds)
            number_pattern = r'(\d+)'
            match = re.search(number_pattern, time_str)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, TypeError):
                    pass
            
            return 0.0
            
        except Exception:
            return 0.0

    def _estimate_timestamp_from_text(self, target_text: str, full_content: str) -> float:
        """Estimate timestamp based on text position in content."""
        try:
            if not isinstance(target_text, str) or not isinstance(full_content, str):
                return 0.0
                
            # Find approximate position of text in content
            position = full_content.lower().find(target_text.lower())
            if position == -1:
                return 0.0
            
            # Estimate time based on character position
            # Assume average speaking rate of 150 words per minute
            chars_before = position
            words_before = len(full_content[:position].split())
            estimated_time = (words_before / 150.0) * 60.0  # Convert to seconds
            
            return max(0.0, estimated_time)
            
        except Exception:
            return 0.0

    def analyze_all_segments(self, segments: List[TranscriptSegment], case_facts: str,
                           model_name: str = "mistral",
                           progress_callback: Optional[Callable] = None,
                           segment_complete_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Analyze all transcript segments sequentially with comprehensive logging.
        
        Args:
            segments: List of transcript segments
            case_facts: Case facts for context
            model_name: AI model to use
            progress_callback: Optional callback for overall progress
            segment_complete_callback: Optional callback when each segment completes
            
        Returns:
            List of detailed analysis results with performance metrics
        """
        self._log_info(f"=== STARTING BATCH ANALYSIS OF ALL SEGMENTS ===", progress_callback)
        analysis_start_time = datetime.now()
        results = []
        total_segments = len(segments)
        
        # Validate inputs
        try:
            if not segments:
                self._log_error("No segments provided for analysis", progress_callback)
                return results
                
            if not isinstance(segments, list):
                self._log_error(f"Invalid segments type: {type(segments)}", progress_callback)
                return results
                
            self._log_info(f"Analyzing {total_segments} recording segments...", progress_callback)
            self._log_debug(f"Model: {model_name}", progress_callback)
            self._log_debug(f"Case facts length: {len(case_facts) if case_facts else 0} characters", progress_callback)
            
            # Log segment summary
            try:
                total_words = sum(seg.word_count for seg in segments if hasattr(seg, 'word_count'))
                self._log_info(f"Total words to analyze: {total_words}", progress_callback)
                
                # Log individual segment info
                for i, segment in enumerate(segments):
                    try:
                        if hasattr(segment, 'filename') and hasattr(segment, 'word_count'):
                            self._log_debug(f"  Segment {i+1}: {segment.filename} ({segment.word_count} words)", progress_callback)
                        else:
                            self._log_warning(f"  Segment {i+1}: Invalid segment object", progress_callback)
                    except Exception as seg_log_error:
                        self._log_exception(f"Error logging segment {i+1} info", seg_log_error, progress_callback)
            except Exception as summary_error:
                self._log_exception("Error generating segment summary", summary_error, progress_callback)
            
            successful_analyses = 0
            failed_analyses = 0
            total_processing_time = 0
            
            for i, segment in enumerate(segments):
                try:
                    if self.is_cancelled:
                        self._log_warning(f"Analysis cancelled at segment {i+1}/{total_segments}", progress_callback)
                        break

                    segment_number = i + 1
                    self._log_info(f"=== PROCESSING SEGMENT {segment_number}/{total_segments} ===", progress_callback)
                    
                    # Validate segment before processing
                    if not segment:
                        self._log_error(f"Segment {segment_number} is None", progress_callback)
                        failed_analyses += 1
                        continue
                        
                    if not hasattr(segment, 'filename'):
                        self._log_error(f"Segment {segment_number} missing filename attribute", progress_callback)
                        failed_analyses += 1
                        continue
                        
                    self._log_info(f"Processing segment {segment_number}/{total_segments}: {segment.filename}", progress_callback)
                    
                    # Analyze this segment
                    segment_start_time = datetime.now()
                    try:
                        result = self.analyze_segment(
                            segment=segment,
                            case_facts=case_facts,
                            model_name=model_name,
                            progress_callback=progress_callback
                        )
                        segment_time = (datetime.now() - segment_start_time).total_seconds()
                        self._log_debug(f"Segment {segment_number} analysis completed in {segment_time:.1f}s", progress_callback)
                    except Exception as analyze_error:
                        self._log_exception(f"Error analyzing segment {segment_number}", analyze_error, progress_callback)
                        segment_time = (datetime.now() - segment_start_time).total_seconds()
                        result = {
                            'success': False,
                            'error': f'Analysis exception: {str(analyze_error)}',
                            'segment': segment,
                            'ai_response': '',
                            'processing_time': segment_time,
                            'error_type': type(analyze_error).__name__
                        }

                    results.append(result)
                    
                    # Update statistics
                    if result and result.get('success'):
                        successful_analyses += 1
                        total_processing_time += result.get('processing_time', 0)
                        
                        # Log success with relevance info
                        try:
                            relevance_info = result.get('relevance_score', {})
                            if isinstance(relevance_info, dict) and relevance_info.get('is_relevant'):
                                self._log_info(f"✓ {segment.filename}: RELEVANT content found", progress_callback)
                            else:
                                self._log_info(f"✓ {segment.filename}: No relevant content", progress_callback)
                        except Exception as relevance_log_error:
                            self._log_warning(f"Error logging relevance for {segment.filename}: {str(relevance_log_error)}", progress_callback)
                    else:
                        failed_analyses += 1
                        error = result.get('error', 'Unknown error') if result else 'No result returned'
                        self._log_error(f"✗ {segment.filename}: {error}", progress_callback)

                    # Notify completion of this segment
                    if segment_complete_callback:
                        try:
                            segment_complete_callback(result, segment_number, total_segments)
                        except Exception as callback_error:
                            self._log_exception("Error in segment complete callback", callback_error, progress_callback)

                    # Log progress summary
                    progress_pct = (segment_number / total_segments) * 100
                    self._log_debug(f"Progress: {segment_number}/{total_segments} ({progress_pct:.1f}%)", progress_callback)
                    
                except Exception as segment_error:
                    self._log_exception(f"Critical error processing segment {i+1}", segment_error, progress_callback)
                    failed_analyses += 1
                    continue

            # Final analysis summary
            total_analysis_time = (datetime.now() - analysis_start_time).total_seconds()
            avg_time_per_segment = total_processing_time / successful_analyses if successful_analyses > 0 else 0
            
            self._log_info(f"=== BATCH ANALYSIS COMPLETED ===", progress_callback)
            self._log_info(f"Analysis completed in {total_analysis_time:.1f}s", progress_callback)
            self._log_info(f"Results: {successful_analyses} successful, {failed_analyses} failed", progress_callback)
            if successful_analyses > 0:
                self._log_debug(f"Average processing time per segment: {avg_time_per_segment:.1f}s", progress_callback)
            
            # Count relevant segments
            try:
                relevant_segments = sum(1 for r in results 
                                      if r and r.get('success') and 
                                      isinstance(r.get('relevance_score'), dict) and 
                                      r.get('relevance_score', {}).get('is_relevant'))
                
                self._log_info(f"Found {relevant_segments} segments with relevant content", progress_callback)
            except Exception as count_error:
                self._log_exception("Error counting relevant segments", count_error, progress_callback)

            return results
            
        except Exception as e:
            self._log_exception("Critical error in analyze_all_segments", e, progress_callback)
            return results

    def cancel_analysis(self, progress_callback: Optional[Callable] = None):
        """Cancel ongoing analysis with logging."""
        self._log_info("=== ANALYSIS CANCELLATION REQUESTED ===", progress_callback)
        self.is_cancelled = True

    def reset_cancellation(self, progress_callback: Optional[Callable] = None):
        """Reset cancellation flag for new analysis with logging."""
        self._log_debug("Resetting cancellation flag for new analysis", progress_callback)
        self.is_cancelled = False

    def save_results_to_files(self, results: List[Dict[str, Any]], output_directory: str,
                           case_facts: str, save_individual: bool = True,
                           save_combined: bool = True, save_crime_report: bool = True,
                           audio_directory: Optional[str] = None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Save analysis results to files with comprehensive logging and crime-focused reports.
        
        Args:
            results: List of analysis results
            output_directory: Directory to save files
            case_facts: Original case facts for reference
            save_individual: Whether to save individual .ai.txt files
            save_combined: Whether to save combined summary file
            save_crime_report: Whether to save crime-focused report with audio links
            audio_directory: Directory containing original audio files for linking
            progress_callback: Optional callback for logging progress
            
        Returns:
            Dict with save status, file paths, and detailed save metrics
        """
        save_start_time = datetime.now()
        
        self._log_info(f"Saving analysis results to: {output_directory}", progress_callback)
        self._log_debug(f"Save individual files: {save_individual}", progress_callback)
        self._log_debug(f"Save combined file: {save_combined}", progress_callback)
        self._log_debug(f"Save crime report: {save_crime_report}", progress_callback)
        self._log_debug(f"Audio directory: {audio_directory}", progress_callback)
        self._log_debug(f"Results to save: {len(results)}", progress_callback)
        
        try:
            output_path = Path(output_directory)
            
            # Create directory if needed
            if not output_path.exists():
                self._log_info(f"Creating output directory: {output_path}", progress_callback)
                output_path.mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            successful_saves = 0
            failed_saves = 0

            # Save individual files
            if save_individual:
                self._log_info("Saving individual result files...", progress_callback)
                
                for i, result in enumerate(results):
                    try:
                        if result['success']:
                            segment = result['segment']
                            # Create safe filename
                            base_name = Path(segment.filename).stem
                            safe_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
                            filename = f"{safe_name}.ai.txt"
                            filepath = output_path / filename

                            content = f"""AI Analysis Results
================
Recording: {segment.filename}
Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model Used: {result.get('model_used', 'Unknown')}
Processing Time: {result.get('processing_time', 0):.1f} seconds
Segment Index: {segment.segment_index}
Word Count: {segment.word_count}
Relevance Score: {result.get('relevance_score', {}).get('relevance_score', 'N/A')}

CASE FACTS:
{case_facts}

TRANSCRIPT SEGMENT:
{segment.content}

AI ANALYSIS:
{result['ai_response']}

PROCESSING METRICS:
- Total Processing Time: {result.get('processing_time', 0):.3f}s
- AI Processing Time: {result.get('ai_processing_time', 0):.3f}s
- Prompt Build Time: {result.get('prompt_build_time', 0):.3f}s
- Response Length: {result.get('response_length', 0)} characters
- Response Words: {result.get('response_words', 0)} words
"""

                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(content)
                            saved_files.append(str(filepath))
                            successful_saves += 1
                            
                            self._log_debug(f"Saved individual file: {filename}", progress_callback)
                        else:
                            failed_saves += 1
                            self._log_warning(f"Skipping failed result for segment {i}", progress_callback)
                            
                    except Exception as e:
                        failed_saves += 1
                        self._log_error(f"Error saving individual file for segment {i}: {str(e)}", progress_callback)

            # Save combined summary
            if save_combined:
                self._log_info("Saving combined summary file...", progress_callback)
                
                try:
                    summary_filename = f"ai_review_summary_{timestamp}.txt"
                    summary_path = output_path / summary_filename

                    # Calculate summary statistics
                    successful_results = [r for r in results if r['success']]
                    failed_results = [r for r in results if not r['success']]
                    relevant_results = [r for r in successful_results 
                                      if isinstance(r.get('relevance_score'), dict) and 
                                      r.get('relevance_score', {}).get('is_relevant')]
                    
                    total_processing_time = sum(r.get('processing_time', 0) for r in successful_results)
                    avg_processing_time = total_processing_time / len(successful_results) if successful_results else 0

                    summary_content = f"""AI Review Summary
===============
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Segments Analyzed: {len(results)}
Successful Analyses: {len(successful_results)}
Failed Analyses: {len(failed_results)}
Relevant Segments Found: {len(relevant_results)}
Total Processing Time: {total_processing_time:.1f} seconds
Average Processing Time: {avg_processing_time:.1f} seconds per segment

CASE FACTS:
{case_facts}

EXECUTIVE SUMMARY:
- {len(relevant_results)} out of {len(successful_results)} segments contained relevant content
- Average response length: {sum(r.get('response_length', 0) for r in successful_results) / len(successful_results) if successful_results else 0:.0f} characters
- Total words analyzed: {sum(r['segment'].word_count for r in results)}

ANALYSIS RESULTS:
"""

                    for i, result in enumerate(results, 1):
                        summary_content += f"\n{'='*50}\nSEGMENT {i}: {result['segment'].filename}\n{'='*50}\n"

                        if result['success']:
                            relevance = result.get('relevance_score', {})
                            is_relevant = relevance.get('is_relevant', False) if isinstance(relevance, dict) else False
                            relevance_score = relevance.get('relevance_score', 0) if isinstance(relevance, dict) else 0
                            
                            summary_content += f"Status: ✓ Success\n"
                            summary_content += f"Relevance: {'RELEVANT' if is_relevant else 'Not Relevant'} (Score: {relevance_score})\n"
                            summary_content += f"Processing Time: {result.get('processing_time', 0):.1f} seconds\n"
                            summary_content += f"Model: {result.get('model_used', 'Unknown')}\n"
                            summary_content += f"Word Count: {result['segment'].word_count}\n\n"
                            summary_content += f"TRANSCRIPT:\n{result['segment'].content}\n\n"
                            summary_content += f"AI ANALYSIS:\n{result['ai_response']}\n"
                        else:
                            summary_content += f"Status: ✗ Failed\n"
                            summary_content += f"Error: {result.get('error', 'Unknown error')}\n"
                            summary_content += f"Word Count: {result['segment'].word_count}\n"
                            summary_content += f"TRANSCRIPT:\n{result['segment'].content}\n"

                    with open(summary_path, 'w', encoding='utf-8') as f:
                        f.write(summary_content)
                    saved_files.append(str(summary_path))
                    successful_saves += 1
                    
                    self._log_info(f"Saved combined summary: {summary_filename}", progress_callback)
                    
                except Exception as e:
                    failed_saves += 1
                    self._log_error(f"Error saving combined summary: {str(e)}", progress_callback)

            # Save crime-focused report with audio links
            if save_crime_report:
                self._log_info("Saving crime-focused investigation report...", progress_callback)
                
                try:
                    crime_report_filename = f"crime_investigation_report_{timestamp}.html"
                    crime_report_path = output_path / crime_report_filename

                    # Collect all crime-relevant sections
                    all_crime_sections = []
                    for result in results:
                        if result['success'] and result.get('crime_relevant_sections'):
                            sections = result['crime_relevant_sections']
                            # Set audio file paths for linking
                            for section in sections:
                                if audio_directory:
                                    audio_file_path = Path(audio_directory) / section.filename
                                    if audio_file_path.exists():
                                        section.audio_file_path = str(audio_file_path)
                            all_crime_sections.extend(sections)

                    crime_report_content = self._generate_crime_html_report(
                        all_crime_sections, case_facts, timestamp, results
                    )

                    with open(crime_report_path, 'w', encoding='utf-8') as f:
                        f.write(crime_report_content)
                    saved_files.append(str(crime_report_path))
                    successful_saves += 1
                    
                    self._log_info(f"Saved crime investigation report: {crime_report_filename}", progress_callback)
                    self._log_info(f"Found {len(all_crime_sections)} crime-relevant sections across all recordings", progress_callback)
                    
                except Exception as e:
                    failed_saves += 1
                    self._log_error(f"Error saving crime investigation report: {str(e)}", progress_callback)

            save_time = (datetime.now() - save_start_time).total_seconds()
            
            self._log_info(f"✓ File saving completed in {save_time:.3f}s", progress_callback)
            self._log_info(f"Successfully saved {successful_saves} files", progress_callback)
            if failed_saves > 0:
                self._log_warning(f"Failed to save {failed_saves} files", progress_callback)

            return {
                'success': True,
                'files_saved': saved_files,
                'output_directory': str(output_path),
                'successful_saves': successful_saves,
                'failed_saves': failed_saves,
                'save_time': save_time,
                'timestamp': timestamp
            }

        except Exception as e:
            error_msg = f"Error saving results: {str(e)}"
            self._log_error(error_msg, progress_callback)
            self._log_debug(f"Full traceback: {traceback.format_exc()}", progress_callback)
            return {
                'success': False,
                'error': error_msg,
                'files_saved': [],
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }

    def test_ollama_connection(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Test connection to Ollama instance with logging.
        
        Args:
            progress_callback: Optional callback for logging progress
            
        Returns:
            Dict with connection status and available models
        """
        self._log_info("Testing Ollama connection...", progress_callback)
        return self.ollama_client.test_connection(progress_callback)

    def get_available_models(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Get available AI models from Ollama with logging.
        
        Args:
            progress_callback: Optional callback for logging progress
            
        Returns:
            Dict with model information
        """
        self._log_info("Retrieving available models...", progress_callback)
        return self.ollama_client.get_available_models(progress_callback) 

    def _generate_crime_html_report(self, crime_sections: List[CrimeRelevantSection], case_facts: str, 
                                   timestamp: str, analysis_results: List[Dict[str, Any]]) -> str:
        """
        Generate enhanced HTML crime investigation report with JSON optimization.
        
        Args:
            crime_sections: List of crime-relevant sections
            case_facts: Original case facts
            timestamp: Report timestamp
            analysis_results: Full analysis results for context
            
        Returns:
            HTML content string with enhanced formatting for JSON data
        """
        # Count JSON vs text segments
        json_segments = sum(1 for result in analysis_results 
                          if result.get('success') and 
                          (result['segment'].filename.startswith('json_') or '.json' in result['segment'].filename.lower()))
        text_segments = len([r for r in analysis_results if r.get('success')]) - json_segments
        
        # Calculate enhanced statistics
        high_accuracy_sections = sum(1 for section in crime_sections 
                                   if hasattr(section, 'accuracy_level') and section.accuracy_level == 'high')
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crime Investigation Report - Enhanced Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #d32f2f, #b71c1c);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background-color: #fafafa;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #2196F3;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 2em;
        }}
        .stat-card p {{
            margin: 0;
            color: #666;
            font-weight: 500;
        }}
        .json-indicator {{
            border-left-color: #4CAF50;
        }}
        .high-accuracy {{
            border-left-color: #FF9800;
        }}
        .content {{
            padding: 30px;
        }}
        .case-facts {{
            background-color: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        .case-facts h2 {{
            margin: 0 0 15px 0;
            color: #1976D2;
        }}
        .evidence-section {{
            margin: 30px 0;
            padding: 25px;
            background-color: #fff3e0;
            border-left: 4px solid #FF9800;
            border-radius: 0 8px 8px 0;
            position: relative;
        }}
        .evidence-section h3 {{
            margin: 0 0 15px 0;
            color: #F57C00;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .accuracy-badge {{
            font-size: 0.8em;
            padding: 4px 8px;
            border-radius: 12px;
            color: white;
            font-weight: bold;
        }}
        .accuracy-high {{ background-color: #4CAF50; }}
        .accuracy-medium {{ background-color: #FF9800; }}
        .accuracy-estimated {{ background-color: #9E9E9E; }}
        .timestamp {{
            background-color: #f5f5f5;
            padding: 8px 12px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
            color: #555;
            margin: 10px 0;
        }}
        .evidence-text {{
            background-color: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #ddd;
            font-style: italic;
            margin: 15px 0;
        }}
        .relevance {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 6px;
            border-left: 3px solid #4CAF50;
            margin: 15px 0;
        }}
        .audio-link {{
            display: inline-block;
            background-color: #2196F3;
            color: white;
            padding: 8px 16px;
            text-decoration: none;
            border-radius: 4px;
            font-size: 0.9em;
            margin: 10px 0;
            transition: background-color 0.3s;
        }}
        .audio-link:hover {{
            background-color: #1976D2;
        }}
        .no-evidence {{
            text-align: center;
            padding: 50px;
            color: #666;
            font-style: italic;
        }}
        .footer {{
            background-color: #333;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        .enhancement-note {{
            background-color: #e8f5e8;
            border: 1px solid #4CAF50;
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
        }}
        .enhancement-note h4 {{
            margin: 0 0 10px 0;
            color: #2E7D32;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Crime Investigation Report</h1>
            <p>Enhanced AI Analysis with JSON Support</p>
            <p>Generated: {timestamp.replace('_', ' ')}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>{len(crime_sections)}</h3>
                <p>Relevant Evidence Sections</p>
            </div>
            <div class="stat-card json-indicator">
                <h3>{json_segments}</h3>
                <p>JSON Segments Analyzed</p>
            </div>
            <div class="stat-card">
                <h3>{text_segments}</h3>
                <p>Text Segments Analyzed</p>
            </div>
            <div class="stat-card high-accuracy">
                <h3>{high_accuracy_sections}</h3>
                <p>High-Accuracy Timestamps</p>
            </div>
        </div>
        
        <div class="content">
            <div class="case-facts">
                <h2>📋 Case Facts</h2>
                <p>{case_facts}</p>
            </div>
            
            {self._generate_enhancement_notice(json_segments, high_accuracy_sections)}
            
            <h2>🚨 Crime-Relevant Evidence Found</h2>
"""

        if crime_sections:
            for i, section in enumerate(crime_sections, 1):
                accuracy_class = f"accuracy-{getattr(section, 'accuracy_level', 'estimated')}"
                accuracy_text = getattr(section, 'accuracy_level', 'estimated').title()
                
                html_content += f"""
            <div class="evidence-section">
                <h3>
                    Evidence #{i} - {section.filename}
                    <span class="accuracy-badge {accuracy_class}">{accuracy_text} Accuracy</span>
                </h3>
                
                <div class="timestamp">
                    ⏰ Timestamp: {section.get_timestamp_range()}
                </div>
                
                <div class="evidence-text">
                    "{section.text}"
                </div>
                
                <div class="relevance">
                    <strong>Relevance to Case:</strong> {section.relevance_explanation}
                </div>
                
                {self._generate_audio_link_html(section)}
            </div>
"""
        else:
            html_content += """
            <div class="no-evidence">
                <h3>No Crime-Relevant Evidence Found</h3>
                <p>The AI analysis did not identify any content directly relevant to the specified case facts.</p>
            </div>
"""

        html_content += f"""
        </div>
        
        <div class="footer">
            <p>Generated by Enhanced AI Review System | Timestamp Accuracy: {high_accuracy_sections}/{len(crime_sections)} High Precision</p>
            <p>JSON Support Enabled | Advanced Timestamp Analysis | Crime Investigation AI v2.0</p>
        </div>
    </div>
</body>
</html>"""

        return html_content

    def _generate_enhancement_notice(self, json_segments: int, high_accuracy_sections: int) -> str:
        """Generate notice about JSON enhancements."""
        if json_segments > 0:
            return f"""
            <div class="enhancement-note">
                <h4>🚀 Enhanced JSON Analysis</h4>
                <p>This report includes analysis of {json_segments} JSON-formatted segments with enhanced structured data processing. 
                JSON data provides more precise timestamps and metadata, resulting in {high_accuracy_sections} high-accuracy evidence sections.</p>
                <ul>
                    <li>Structured data parsing for improved accuracy</li>
                    <li>Enhanced timestamp resolution from JSON metadata</li>
                    <li>Optimized AI prompts for structured content analysis</li>
                    <li>Advanced text-to-timestamp matching algorithms</li>
                </ul>
            </div>"""
        return ""

    def _generate_audio_link_html(self, section: CrimeRelevantSection) -> str:
        """Generate HTML for audio links."""
        if section.audio_file_path:
            return f'''
                <a href="{section.get_audio_link()}" class="audio-link" target="_blank">
                    🎵 Play Audio at {section._format_timestamp(section.start_time)}
                </a>'''
        return ""

    def load_chunked_json_transcript(self, transcript_path: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Load the chunked JSON transcript file optimized for AI review.
        
        Args:
            transcript_path: Path to the chunked JSON transcript file (recordings_chunked.json)
            progress_callback: Optional callback for logging progress
            
        Returns:
            Dict with success status, content, and detailed file information
        """
        self._log_info(f"=== STARTING CHUNKED JSON TRANSCRIPT LOADING ===", progress_callback)
        self._log_info(f"Loading chunked JSON transcript from: {transcript_path}", progress_callback)
        
        try:
            # Check if file exists
            self._log_debug(f"Checking if file exists: {transcript_path}", progress_callback)
            if not os.path.exists(transcript_path):
                error_msg = f"Chunked JSON transcript file not found: {transcript_path}"
                self._log_error(error_msg, progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'content': '',
                    'file_path': transcript_path
                }
                
            self._log_debug("File exists, getting file statistics...", progress_callback)

            # Get file info
            file_stat = os.stat(transcript_path)
            file_size = file_stat.st_size
            modified_time = datetime.fromtimestamp(file_stat.st_mtime)
            
            self._log_debug(f"File size: {file_size} bytes ({file_size / 1024:.1f} KB)", progress_callback)
            self._log_debug(f"Last modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}", progress_callback)

            # Read file content
            self._log_debug("Starting file read operation...", progress_callback)
            start_time = datetime.now()
            
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
            except Exception as read_error:
                self._log_exception("Error during file read", read_error, progress_callback)
                raise
                
            read_time = (datetime.now() - start_time).total_seconds()
            
            self._log_debug(f"File read completed in {read_time:.3f}s", progress_callback)
            self._log_debug(f"Raw content length: {len(raw_content)} characters", progress_callback)

            if not raw_content.strip():
                error_msg = "Chunked JSON transcript file is empty"
                self._log_error(error_msg, progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'content': '',
                    'file_size': file_size,
                    'file_path': transcript_path
                }

            # Parse JSON content
            try:
                self._log_debug("Parsing chunked JSON content...", progress_callback)
                json_data = json.loads(raw_content)
                self._log_info("✓ Chunked JSON file parsed successfully", progress_callback)
                self._log_debug(f"JSON data type: {type(json_data)}", progress_callback)
                
                # Validate structure
                if not isinstance(json_data, dict):
                    raise ValueError("Invalid JSON format: root should be an object")
                
                if "recordings" not in json_data or not isinstance(json_data["recordings"], list):
                    raise ValueError("Invalid JSON format: missing 'recordings' array")
                
                # Extract summary information
                summary = json_data.get("summary", {})
                model_used = summary.get("model_used", "Unknown")
                device = summary.get("device", "Unknown")
                generated = summary.get("generated", "Unknown")
                recordings_count = summary.get("recordings_transcribed", 0)
                
                self._log_info(f"Transcript summary: {recordings_count} recordings, model: {model_used}", progress_callback)
                self._log_debug(f"Generated: {generated}, Device: {device}", progress_callback)
                
                # Count total chunks
                total_chunks = 0
                for recording in json_data["recordings"]:
                    chunks = recording.get("chunks", [])
                    total_chunks += len(chunks)
                
                self._log_info(f"Found {len(json_data['recordings'])} recordings with {total_chunks} total chunks", progress_callback)
                
                # Store original JSON data for enhanced processing
                json_metadata = {
                    'is_json': True,
                    'is_chunked': True,
                    'original_json': json_data,
                    'json_structure': {
                        'recordings_count': len(json_data['recordings']),
                        'total_chunks': total_chunks,
                        'model_used': model_used,
                        'device': device,
                        'generated': generated
                    }
                }
                
                # Convert to text format for compatibility with existing code
                content = self._convert_chunked_json_to_text(json_data, progress_callback)
                
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON format: {str(e)}"
                self._log_exception("JSON parsing failed", e, progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'content': '',
                    'file_size': file_size,
                    'file_path': transcript_path,
                    'error_type': 'JSONDecodeError',
                    'error_details': str(e)
                }
            except Exception as json_error:
                self._log_exception("Unexpected error during JSON processing", json_error, progress_callback)
                raise

            # Analyze content
            self._log_debug("Analyzing processed content...", progress_callback)
            char_count = len(content)
            word_count = len(content.split())
            line_count = len(content.splitlines())
            
            self._log_info(f"✓ Chunked JSON transcript loaded successfully", progress_callback)
            self._log_debug(f"Content stats: {char_count} chars, {word_count} words, {line_count} lines", progress_callback)
            self._log_info(f"=== CHUNKED JSON TRANSCRIPT LOADING COMPLETED ===", progress_callback)

            result = {
                'success': True,
                'content': content,
                'file_size': file_size,
                'file_path': transcript_path,
                'character_count': char_count,
                'word_count': word_count,
                'line_count': line_count,
                'modified_time': modified_time,
                'read_time': read_time,
                **json_metadata
            }
            
            return result

        except UnicodeDecodeError as e:
            error_msg = f"Unicode decode error: {str(e)}"
            self._log_exception("Unicode decode error during file reading", e, progress_callback)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'error_type': 'UnicodeDecodeError',
                'error_details': str(e)
            }
        except PermissionError as e:
            error_msg = f"Permission denied accessing file: {str(e)}"
            self._log_exception("Permission error accessing file", e, progress_callback)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'error_type': 'PermissionError'
            }
        except Exception as e:
            error_msg = f"Error reading chunked JSON transcript file: {str(e)}"
            self._log_exception("Unexpected error during transcript loading", e, progress_callback)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }

    def _convert_chunked_json_to_text(self, json_data: Dict[str, Any], progress_callback: Optional[Callable] = None) -> str:
        """
        Convert chunked JSON transcript data to text format for processing.
        
        Args:
            json_data: Parsed chunked JSON data
            progress_callback: Optional callback for logging progress
            
        Returns:
            Formatted text content
        """
        self._log_debug("Converting chunked JSON data to text format", progress_callback)
        
        try:
            text_parts = []
            
            # Process each recording
            for recording_idx, recording in enumerate(json_data.get("recordings", [])):
                filename = recording.get("filename", f"recording_{recording_idx+1}.unknown")
                duration = recording.get("duration_seconds", 0)
                language = recording.get("language", "unknown")
                
                # Add recording header
                text_parts.append(f"==== {filename} ====")
                text_parts.append(f"Duration: {duration} seconds")
                text_parts.append(f"Language: {language}")
                text_parts.append("")
                
                # Process each chunk
                chunks = recording.get("chunks", [])
                for chunk in chunks:
                    chunk_id = chunk.get("chunk_id", 0)
                    start = chunk.get("start", 0)
                    end = chunk.get("end", 0)
                    text = chunk.get("text", "").strip()
                    
                    if text:
                        # Format as timestamped chunk
                        text_parts.append(f"[Chunk {chunk_id}: {start:.1f}s - {end:.1f}s]")
                        text_parts.append(text)
                        text_parts.append("")
                
                # Add separator between recordings
                text_parts.append("-" * 50)
                text_parts.append("")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            self._log_error(f"Error converting chunked JSON to text: {str(e)}", progress_callback)
            # Fallback to JSON pretty print
            try:
                return json.dumps(json_data, indent=2, ensure_ascii=False) if json_data is not None else ""
            except Exception:
                return str(json_data) if json_data is not None else ""

    def _segment_chunked_json_transcript(self, json_data: Dict[str, Any], progress_callback: Optional[Callable] = None) -> List[TranscriptSegment]:
        """
        Create transcript segments directly from chunked JSON data.
        
        Args:
            json_data: Original chunked JSON data
            progress_callback: Optional callback for logging progress
            
        Returns:
            List of TranscriptSegment objects
        """
        segments = []
        
        try:
            recordings = json_data.get("recordings", [])
            self._log_info(f"Processing {len(recordings)} recordings from chunked JSON", progress_callback)
            
            segment_index = 0
            for recording_idx, recording in enumerate(recordings):
                filename = recording.get("filename", f"recording_{recording_idx+1}.unknown")
                chunks = recording.get("chunks", [])
                
                self._log_debug(f"Processing recording: {filename} with {len(chunks)} chunks", progress_callback)
                
                for chunk in chunks:
                    chunk_id = chunk.get("chunk_id", 0)
                    start = chunk.get("start", 0)
                    end = chunk.get("end", 0)
                    text = chunk.get("text", "").strip()
                    
                    if not text:
                        continue
                    
                    # Create detailed segments for timestamp info
                    detailed_segments = [{
                        "start": start,
                        "end": end,
                        "text": text
                    }]
                    
                    # Create segment with chunk info in filename
                    segment = TranscriptSegment(
                        filename=f"{filename} (Chunk {chunk_id})",
                        content=text,
                        segment_index=segment_index,
                        detailed_segments=detailed_segments
                    )
                    
                    segments.append(segment)
                    segment_index += 1
            
            self._log_info(f"Created {len(segments)} segments from chunked JSON data", progress_callback)
            return segments
            
        except Exception as e:
            self._log_error(f"Error creating segments from chunked JSON: {str(e)}", progress_callback)
            return []