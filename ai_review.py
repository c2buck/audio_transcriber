import os
import re
import threading
import traceback
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
        self.content = content.strip()
        self.segment_index = segment_index
        self.word_count = len(self.content.split()) if self.content else 0
        self.detailed_segments = detailed_segments or []
        
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
        self.text = text.strip()
        self.start_time = start_time
        self.end_time = end_time
        self.relevance_explanation = relevance_explanation
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
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes >= 60:
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
    
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
        
    def _log_debug(self, message: str, callback: Optional[Callable] = None):
        """Log debug message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_msg = f"[{timestamp}] DEBUG: {message}"
        if callback:
            callback(log_msg)
            
    def _log_error(self, message: str, callback: Optional[Callable] = None):
        """Log error message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_msg = f"[{timestamp}] ERROR: {message}"
        if callback:
            callback(log_msg)
            
    def _log_info(self, message: str, callback: Optional[Callable] = None):
        """Log info message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_msg = f"[{timestamp}] INFO: {message}"
        if callback:
            callback(log_msg)
            
    def _log_warning(self, message: str, callback: Optional[Callable] = None):
        """Log warning message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_msg = f"[{timestamp}] WARNING: {message}"
        if callback:
            callback(log_msg)
        
    def load_combined_transcript(self, transcript_path: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Load the combined transcript file with comprehensive logging.
        
        Args:
            transcript_path: Path to the combined transcript file
            progress_callback: Optional callback for logging progress
            
        Returns:
            Dict with success status, content, and detailed file information
        """
        self._log_info(f"Loading combined transcript from: {transcript_path}", progress_callback)
        
        try:
            # Check if file exists
            if not os.path.exists(transcript_path):
                error_msg = f"Transcript file not found: {transcript_path}"
                self._log_error(error_msg, progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'content': '',
                    'file_path': transcript_path
                }

            # Get file info
            file_stat = os.stat(transcript_path)
            file_size = file_stat.st_size
            modified_time = datetime.fromtimestamp(file_stat.st_mtime)
            
            self._log_debug(f"File size: {file_size} bytes ({file_size / 1024:.1f} KB)", progress_callback)
            self._log_debug(f"Last modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}", progress_callback)

            # Read file content
            start_time = datetime.now()
            with open(transcript_path, 'r', encoding='utf-8') as f:
                content = f.read()
            read_time = (datetime.now() - start_time).total_seconds()
            
            self._log_debug(f"File read completed in {read_time:.3f}s", progress_callback)

            if not content.strip():
                error_msg = "Transcript file is empty"
                self._log_error(error_msg, progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'content': '',
                    'file_size': file_size,
                    'file_path': transcript_path
                }

            # Analyze content
            char_count = len(content)
            word_count = len(content.split())
            line_count = len(content.splitlines())
            
            self._log_info(f"✓ Transcript loaded successfully", progress_callback)
            self._log_debug(f"Content stats: {char_count} chars, {word_count} words, {line_count} lines", progress_callback)

            return {
                'success': True,
                'content': content,
                'file_size': file_size,
                'file_path': transcript_path,
                'character_count': char_count,
                'word_count': word_count,
                'line_count': line_count,
                'modified_time': modified_time,
                'read_time': read_time
            }

        except UnicodeDecodeError as e:
            error_msg = f"Unicode decode error: {str(e)}"
            self._log_error(error_msg, progress_callback)
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
            self._log_error(error_msg, progress_callback)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'error_type': 'PermissionError'
            }
        except Exception as e:
            error_msg = f"Error reading transcript file: {str(e)}"
            self._log_error(error_msg, progress_callback)
            self._log_debug(f"Full traceback: {traceback.format_exc()}", progress_callback)
            return {
                'success': False,
                'error': error_msg,
                'content': '',
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }

    def segment_transcript(self, transcript_content: str, progress_callback: Optional[Callable] = None) -> List[TranscriptSegment]:
        """
        Segment the combined transcript by recording delimiters with comprehensive logging.
        
        Args:
            transcript_content: Full transcript content
            progress_callback: Optional callback for logging progress
            
        Returns:
            List of TranscriptSegment objects with detailed processing information
        """
        self._log_info("Starting transcript segmentation...", progress_callback)
        
        start_time = datetime.now()
        segments = []
        
        try:
            # Split by recording delimiters (e.g., "==== recording_001.mp3 ====")
            # Pattern matches various delimiter formats
            delimiter_pattern = r'={3,}\s*([^=\n]+?\.(?:mp3|wav|m4a|flac|aac|ogg|wma|mp4|avi|mov|mkv))\s*={3,}'
            
            self._log_debug(f"Using delimiter pattern: {delimiter_pattern}", progress_callback)
            self._log_debug(f"Input content length: {len(transcript_content)} characters", progress_callback)

            # Find all delimiters first for analysis
            delimiter_matches = list(re.finditer(delimiter_pattern, transcript_content, flags=re.IGNORECASE))
            self._log_info(f"Found {len(delimiter_matches)} delimiter matches", progress_callback)
            
            for i, match in enumerate(delimiter_matches):
                filename = match.group(1)
                self._log_debug(f"  Delimiter {i+1}: {filename} at position {match.start()}-{match.end()}", progress_callback)

            # Split the content by the delimiters
            parts = re.split(delimiter_pattern, transcript_content, flags=re.IGNORECASE)
            self._log_debug(f"Split into {len(parts)} parts", progress_callback)

            # Process the parts - every odd index is a filename, every even index is content
            current_filename = None
            segment_index = 0

            for i, part in enumerate(parts):
                part_length = len(part.strip())
                self._log_debug(f"Processing part {i}: {part_length} characters", progress_callback)
                
                if i == 0:
                    # First part might be content before any delimiter
                    if part.strip():
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
                    current_filename = part.strip()
                    self._log_debug(f"Found filename: {current_filename}", progress_callback)
                else:
                    # Even indices (except 0) are content sections
                    if current_filename and part.strip():
                        content = part.strip()
                        segment = TranscriptSegment(
                            filename=current_filename,
                            content=content,
                            segment_index=segment_index
                        )
                        segments.append(segment)
                        self._log_info(f"Created segment: {segment}", progress_callback)
                        segment_index += 1
                    elif current_filename and not part.strip():
                        self._log_warning(f"Empty content for recording: {current_filename}", progress_callback)

            # If no delimiters found, treat the entire content as one segment
            if not segments and transcript_content.strip():
                segment = TranscriptSegment(
                    filename="combined_transcript",
                    content=transcript_content.strip(),
                    segment_index=0
                )
                segments.append(segment)
                self._log_info(f"No delimiters found, created single segment: {segment}", progress_callback)

            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Log summary statistics
            total_words = sum(seg.word_count for seg in segments)
            avg_words = total_words / len(segments) if segments else 0
            
            self._log_info(f"✓ Segmentation completed in {processing_time:.3f}s", progress_callback)
            self._log_info(f"Created {len(segments)} segments with {total_words} total words", progress_callback)
            self._log_debug(f"Average words per segment: {avg_words:.1f}", progress_callback)
            
            for i, segment in enumerate(segments):
                self._log_debug(f"  Segment {i+1}: {segment.filename} ({segment.word_count} words)", progress_callback)

            return segments
            
        except Exception as e:
            error_msg = f"Error during transcript segmentation: {str(e)}"
            self._log_error(error_msg, progress_callback)
            self._log_debug(f"Full traceback: {traceback.format_exc()}", progress_callback)
            
            # Return empty list but log the error
            return []

    def build_analysis_prompt(self, case_facts: str, segment: TranscriptSegment, progress_callback: Optional[Callable] = None) -> str:
        """
        Build the enhanced analysis prompt for crime-relevant content identification.
        
        Args:
            case_facts: User-provided case facts
            segment: Transcript segment to analyze
            progress_callback: Optional callback for logging progress
            
        Returns:
            Formatted prompt string with detailed construction logging
        """
        self._log_debug(f"Building crime analysis prompt for {segment.filename}", progress_callback)
        
        try:
            # Validate inputs
            if not case_facts.strip():
                self._log_warning("Case facts are empty", progress_callback)
                
            if not segment.content.strip():
                self._log_warning(f"Segment content is empty for {segment.filename}", progress_callback)

            prompt = f"""CASE FACTS:
{case_facts.strip()}

TRANSCRIPT FROM {segment.filename}:
{segment.content}

ANALYSIS INSTRUCTIONS:
1. Carefully analyze this transcript for ANY content that may be relevant to the crime or investigation described in the case facts
2. Look for:
   - Direct evidence related to the crime
   - Witness statements or observations
   - Suspect behavior or statements
   - Timeline information
   - Physical evidence descriptions
   - Names, locations, or other identifying details
   - Contradictions or inconsistencies
   - Suspicious activities or behavior

3. For EACH relevant section you find:
   - Quote the EXACT text from the transcript
   - Explain WHY it's relevant to the case
   - Estimate the approximate timestamp within this recording where this content appears

RESPONSE FORMAT:
If relevant content is found, respond with:
RELEVANT SECTIONS:
1. Text: "[exact quote from transcript]"
   Relevance: [explanation of why this is relevant]
   Estimated Time: [approximate time in recording when this occurs]

2. [additional sections if found]

If no relevant content is found, respond with:
NO RELEVANT CONTENT FOUND

Be thorough and consider all possible connections to the case facts."""

            prompt_length = len(prompt)
            prompt_words = len(prompt.split())
            
            self._log_debug(f"Enhanced prompt constructed: {prompt_length} chars, {prompt_words} words", progress_callback)
            self._log_debug(f"Case facts length: {len(case_facts)} chars", progress_callback)
            self._log_debug(f"Transcript content length: {len(segment.content)} chars", progress_callback)

            return prompt
            
        except Exception as e:
            error_msg = f"Error building prompt for {segment.filename}: {str(e)}"
            self._log_error(error_msg, progress_callback)
            # Return a basic prompt as fallback
            return f"Analyze this transcript segment from {segment.filename} for crime-relevant content."

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
        segment_start_time = datetime.now()
        
        self._log_info(f"Starting analysis of segment: {segment.filename}", progress_callback)
        self._log_debug(f"Segment details: {segment.word_count} words, index {segment.segment_index}", progress_callback)
        self._log_debug(f"Using model: {model_name}", progress_callback)
        
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

        try:
            # Build the prompt
            prompt_start_time = datetime.now()
            prompt = self.build_analysis_prompt(case_facts, segment, progress_callback)
            prompt_build_time = (datetime.now() - prompt_start_time).total_seconds()
            
            self._log_debug(f"Prompt built in {prompt_build_time:.3f}s", progress_callback)

            # Send to AI model
            ai_start_time = datetime.now()
            ai_result = self.ollama_client.generate_response(
                model=model_name,
                prompt=prompt,
                progress_callback=progress_callback
            )
            ai_processing_time = (datetime.now() - ai_start_time).total_seconds()
            
            total_segment_time = (datetime.now() - segment_start_time).total_seconds()
            
            if ai_result['success']:
                response_length = len(ai_result.get('response', ''))
                response_words = len(ai_result.get('response', '').split())
                
                self._log_info(f"✓ Segment analysis completed successfully for {segment.filename}", progress_callback)
                self._log_debug(f"AI response: {response_length} chars, {response_words} words", progress_callback)
                self._log_debug(f"Total segment processing: {total_segment_time:.1f}s", progress_callback)
                
                # Analyze response for relevance indicators
                relevance_score = self._analyze_response_relevance(ai_result.get('response', ''), progress_callback)
                
                # Extract crime-relevant sections with timestamps
                crime_sections = self.extract_crime_relevant_sections(
                    ai_result.get('response', ''), segment, progress_callback
                )
                
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
                error_msg = ai_result.get('error', 'Unknown AI error')
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
            error_msg = f"Error analyzing segment {segment.filename}: {str(e)}"
            self._log_error(error_msg, progress_callback)
            self._log_debug(f"Full traceback: {traceback.format_exc()}", progress_callback)
            
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
        Extract crime-relevant sections from AI response with timestamp estimation.
        
        Args:
            ai_response: AI model response containing relevant sections
            segment: Original transcript segment
            progress_callback: Optional callback for logging progress
            
        Returns:
            List of CrimeRelevantSection objects with timestamps
        """
        relevant_sections = []
        
        try:
            self._log_debug(f"Extracting crime-relevant sections from {segment.filename}", progress_callback)
            
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
                        relevant_sections.append(self._create_crime_section(current_section, segment))
                    
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
            
            # Add last section if exists
            if current_section.get('text'):
                relevant_sections.append(self._create_crime_section(current_section, segment))
            
            # Fallback: look for quoted text if structured format not found
            if not relevant_sections and '"' in ai_response:
                self._log_debug("Structured format not found, trying fallback extraction", progress_callback)
                relevant_sections = self._extract_quoted_sections(ai_response, segment)
            
            self._log_info(f"Extracted {len(relevant_sections)} crime-relevant sections from {segment.filename}", progress_callback)
            
            for i, section in enumerate(relevant_sections):
                self._log_debug(f"  Section {i+1}: {section.get_timestamp_range()} - {section.text[:50]}...", progress_callback)
            
            return relevant_sections
            
        except Exception as e:
            error_msg = f"Error extracting crime-relevant sections: {str(e)}"
            self._log_error(error_msg, progress_callback)
            return relevant_sections

    def _create_crime_section(self, section_data: Dict[str, Any], segment: TranscriptSegment) -> CrimeRelevantSection:
        """Create a CrimeRelevantSection from parsed data."""
        text = section_data.get('text', '')
        relevance = section_data.get('relevance', 'Crime-relevant content identified')
        estimated_time = section_data.get('estimated_time', 0.0)
        
        # Estimate end time (assume 30 seconds or based on text length)
        text_words = len(text.split())
        estimated_duration = max(5.0, min(60.0, text_words * 0.5))  # 0.5 seconds per word, 5-60 sec range
        end_time = estimated_time + estimated_duration
        
        return CrimeRelevantSection(
            filename=segment.filename,
            text=text,
            start_time=estimated_time,
            end_time=end_time,
            relevance_explanation=relevance
        )

    def _extract_quoted_sections(self, ai_response: str, segment: TranscriptSegment) -> List[CrimeRelevantSection]:
        """Fallback method to extract quoted text as relevant sections."""
        sections = []
        
        # Find all quoted text
        import re
        quoted_pattern = r'"([^"]+)"'
        quotes = re.findall(quoted_pattern, ai_response)
        
        for i, quote in enumerate(quotes):
            if len(quote.strip()) > 10:  # Only consider substantial quotes
                # Estimate timestamp based on position in transcript
                estimated_time = self._estimate_timestamp_from_text(quote, segment.content)
                
                sections.append(CrimeRelevantSection(
                    filename=segment.filename,
                    text=quote,
                    start_time=estimated_time,
                    end_time=estimated_time + 30.0,  # Default 30 second duration
                    relevance_explanation="Quoted content identified as relevant"
                ))
        
        return sections

    def _parse_timestamp(self, time_str: str) -> float:
        """Parse timestamp string to seconds."""
        try:
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
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                hours = int(match.group(3)) if match.group(3) else 0
                
                if hours > 0:  # HH:MM:SS format
                    return hours * 3600 + minutes * 60 + seconds
                else:  # MM:SS format
                    return minutes * 60 + seconds
            
            # Pattern: "X minutes Y seconds"
            minute_pattern = r'(\d+)\s*(?:minute|min)'
            second_pattern = r'(\d+)\s*(?:second|sec)'
            
            minutes = 0
            seconds = 0
            
            min_match = re.search(minute_pattern, time_str)
            if min_match:
                minutes = int(min_match.group(1))
            
            sec_match = re.search(second_pattern, time_str)
            if sec_match:
                seconds = int(sec_match.group(1))
            
            if minutes > 0 or seconds > 0:
                return minutes * 60 + seconds
            
            # Pattern: just numbers (assume seconds)
            number_pattern = r'(\d+)'
            match = re.search(number_pattern, time_str)
            if match:
                return float(match.group(1))
            
            return 0.0
            
        except Exception:
            return 0.0

    def _estimate_timestamp_from_text(self, target_text: str, full_content: str) -> float:
        """Estimate timestamp based on text position in content."""
        try:
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
        analysis_start_time = datetime.now()
        results = []
        total_segments = len(segments)
        
        self._log_info(f"Starting analysis of {total_segments} recording segments...", progress_callback)
        self._log_debug(f"Model: {model_name}", progress_callback)
        self._log_debug(f"Case facts length: {len(case_facts)} characters", progress_callback)
        
        # Log segment summary
        total_words = sum(seg.word_count for seg in segments)
        self._log_info(f"Total words to analyze: {total_words}", progress_callback)
        
        successful_analyses = 0
        failed_analyses = 0
        total_processing_time = 0
        
        for i, segment in enumerate(segments):
            if self.is_cancelled:
                self._log_warning(f"Analysis cancelled at segment {i+1}/{total_segments}", progress_callback)
                break

            segment_number = i + 1
            self._log_info(f"Processing segment {segment_number}/{total_segments}: {segment.filename}", progress_callback)
            
            # Analyze this segment
            result = self.analyze_segment(
                segment=segment,
                case_facts=case_facts,
                model_name=model_name,
                progress_callback=progress_callback
            )

            results.append(result)
            
            # Update statistics
            if result['success']:
                successful_analyses += 1
                total_processing_time += result.get('processing_time', 0)
                
                # Log success with relevance info
                relevance_info = result.get('relevance_score', {})
                if isinstance(relevance_info, dict) and relevance_info.get('is_relevant'):
                    self._log_info(f"✓ {segment.filename}: RELEVANT content found", progress_callback)
                else:
                    self._log_info(f"✓ {segment.filename}: No relevant content", progress_callback)
            else:
                failed_analyses += 1
                error = result.get('error', 'Unknown error')
                self._log_error(f"✗ {segment.filename}: {error}", progress_callback)

            # Notify completion of this segment
            if segment_complete_callback:
                try:
                    segment_complete_callback(result, segment_number, total_segments)
                except Exception as e:
                    self._log_warning(f"Error in segment complete callback: {str(e)}", progress_callback)

            # Log progress summary
            progress_pct = (segment_number / total_segments) * 100
            self._log_debug(f"Progress: {segment_number}/{total_segments} ({progress_pct:.1f}%)", progress_callback)

        # Final analysis summary
        total_analysis_time = (datetime.now() - analysis_start_time).total_seconds()
        avg_time_per_segment = total_processing_time / successful_analyses if successful_analyses > 0 else 0
        
        self._log_info(f"Analysis completed in {total_analysis_time:.1f}s", progress_callback)
        self._log_info(f"Results: {successful_analyses} successful, {failed_analyses} failed", progress_callback)
        if successful_analyses > 0:
            self._log_debug(f"Average processing time per segment: {avg_time_per_segment:.1f}s", progress_callback)
        
        # Count relevant segments
        relevant_segments = sum(1 for r in results 
                              if r.get('success') and 
                              isinstance(r.get('relevance_score'), dict) and 
                              r.get('relevance_score', {}).get('is_relevant'))
        
        self._log_info(f"Found {relevant_segments} segments with relevant content", progress_callback)

        return results

    def cancel_analysis(self, progress_callback: Optional[Callable] = None):
        """Cancel ongoing analysis with logging."""
        self._log_info("Analysis cancellation requested", progress_callback)
        self.is_cancelled = True

    def reset_cancellation(self, progress_callback: Optional[Callable] = None):
        """Reset cancellation flag for new analysis with logging."""
        self._log_debug("Resetting cancellation flag", progress_callback)
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