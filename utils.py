import os
import re
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable


def format_time(seconds: float) -> str:
    """Format seconds into a readable time string (HH:MM:SS)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing invalid characters."""
    # Remove or replace invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores and dots
    safe_name = safe_name.strip('_.')
    
    # Ensure it's not empty
    if not safe_name:
        safe_name = "unnamed"
    
    return safe_name


def get_audio_files(directory: str, sort_by_date: bool = True, debug_log: bool = False) -> List[str]:
    """
    Get all supported audio files from a directory.
    
    Args:
        directory: Directory to scan for audio files
        sort_by_date: If True, sort by dates extracted from filenames (chronologically)
                     If False, sort alphabetically
        debug_log: If True, print debug information about date extraction
    """
    import re
    from datetime import datetime
    
    supported_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.mp4', '.avi', '.mov', '.mkv'}
    audio_files = []
    
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in supported_extensions:
                    audio_files.append(os.path.join(root, file))
    except Exception as e:
        print(f"Error scanning directory: {e}")
    
    if sort_by_date:
        # Define common date and time patterns to extract from filenames
        # Priority is given to patterns that include both date and time
        date_time_patterns = [
            # Date and time patterns with 12-hour format (highest priority)
            # YYYY-MM-DD hh:mm:ss AM/PM
            r'(\d{4}[-_]\d{2}[-_]\d{2}[-_ T]\d{1,2}[-_:]\d{2}[-_:]\d{2}\s*[AaPp][Mm])',
            # YYYY-MM-DD hh:mm AM/PM
            r'(\d{4}[-_]\d{2}[-_]\d{2}[-_ T]\d{1,2}[-_:]\d{2}\s*[AaPp][Mm])',
            # DD-MM-YYYY hh:mm:ss AM/PM
            r'(\d{2}[-_]\d{2}[-_]\d{4}[-_ T]\d{1,2}[-_:]\d{2}[-_:]\d{2}\s*[AaPp][Mm])',
            # DD-MM-YYYY hh:mm AM/PM
            r'(\d{2}[-_]\d{2}[-_]\d{4}[-_ T]\d{1,2}[-_:]\d{2}\s*[AaPp][Mm])',
            
            # Date and time patterns with 24-hour format (high priority)
            # YYYY-MM-DD HH:MM:SS or YYYY_MM_DD_HH_MM_SS
            r'(\d{4}[-_]\d{2}[-_]\d{2}[-_ T]\d{2}[-_:]\d{2}[-_:]\d{2})',
            # YYYY-MM-DD HH:MM or YYYY_MM_DD_HH_MM
            r'(\d{4}[-_]\d{2}[-_]\d{2}[-_ T]\d{2}[-_:]\d{2})',
            # DD-MM-YYYY HH:MM:SS or DD_MM_YYYY_HH_MM_SS
            r'(\d{2}[-_]\d{2}[-_]\d{4}[-_ T]\d{2}[-_:]\d{2}[-_:]\d{2})',
            # DD-MM-YYYY HH:MM or DD_MM_YYYY_HH_MM
            r'(\d{2}[-_]\d{2}[-_]\d{4}[-_ T]\d{2}[-_:]\d{2})',
            
            # Compact formats with date and time
            # YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS
            r'(\d{8}[-_]\d{6})',
            # YYYYMMDD_HHMM or YYYYMMDD-HHMM
            r'(\d{8}[-_]\d{4})',
            # DDMMYYYY_HHMMSS or DDMMYYYY-HHMMSS
            r'(\d{8}[-_]\d{6})',
            # DDMMYYYY_HHMM or DDMMYYYY-HHMM
            r'(\d{8}[-_]\d{4})',
            
            # Standalone time patterns (12-hour format)
            r'[-_](\d{1,2}[-_:]\d{2}[-_:]\d{2}\s*[AaPp][Mm])[-_]',  # hh:mm:ss AM/PM
            r'[-_](\d{1,2}[-_:]\d{2}\s*[AaPp][Mm])[-_]',  # hh:mm AM/PM
            r'[-_](\d{1,2}[AaPp][Mm])[-_]',  # hh AM/PM (simple format)
            
            # Standalone time patterns (24-hour format)
            r'[-_](\d{2}[-_:]\d{2}[-_:]\d{2})[-_]',  # HH:MM:SS
            r'[-_](\d{2}[-_:]\d{2})[-_]',  # HH:MM
            r'[-_](\d{6})[-_]',  # HHMMSS
            r'[-_](\d{4})[-_]',  # HHMM
            
            # Date only patterns (lower priority)
            # YYYY-MM-DD or YYYY_MM_DD
            r'(\d{4}[-_]\d{2}[-_]\d{2})',
            # DD-MM-YYYY or DD_MM_YYYY
            r'(\d{2}[-_]\d{2}[-_]\d{4})',
            # MM-DD-YYYY or MM_DD_YYYY
            r'(\d{2}[-_]\d{2}[-_]\d{4})',
            # YYYYMMDD
            r'(\d{8})',
            # DDMMYYYY
            r'(\d{8})',
            # Month names: 01Jan2023, Jan01_2023, etc.
            r'(\d{2}[A-Za-z]{3}\d{4})',
            r'([A-Za-z]{3}\d{2}[-_]\d{4})',
        ]
        
        # Dictionary to store extracted dates for debugging
        date_info = {}
        
        def extract_date_from_filename(filepath):
            filename = Path(filepath).name
            original_date = None
            pattern_used = None
            extracted_time = False
            
            # Try all patterns until we find a match
            for pattern in date_time_patterns:
                match = re.search(pattern, filename)
                if match:
                    date_str = match.group(1)
                    pattern_used = pattern
                    
                    # Try different date/time formats based on the pattern matched
                    try:
                        # 12-hour time formats with AM/PM (highest priority)
                        if re.search(r'[AaPp][Mm]', date_str):
                            time_format = "12-hour"
                            # Extract AM/PM indicator and clean up the string
                            am_pm = re.search(r'([AaPp][Mm])', date_str).group(1).upper()
                            
                            # YYYY-MM-DD hh:mm:ss AM/PM
                            if re.match(r'\d{4}[-_]\d{2}[-_]\d{2}[-_ T]\d{1,2}[-_:]\d{2}[-_:]\d{2}\s*[AaPp][Mm]', date_str):
                                # Remove AM/PM for parsing
                                cleaned = re.sub(r'\s*[AaPp][Mm]', '', date_str)
                                cleaned = cleaned.replace('_', '-').replace(' ', '-').replace('T', '-')
                                
                                # Parse date and time components
                                dt_parts = re.match(r'(\d{4})-(\d{2})-(\d{2})-(\d{1,2})-(\d{2})-(\d{2})', cleaned)
                                if dt_parts:
                                    year, month, day, hour, minute, second = map(int, dt_parts.groups())
                                    
                                    # Convert 12-hour to 24-hour format
                                    if am_pm == 'PM' and hour < 12:
                                        hour += 12
                                    elif am_pm == 'AM' and hour == 12:
                                        hour = 0
                                    
                                    original_date = datetime(year, month, day, hour, minute, second)
                                    extracted_time = True
                                    break
                            
                            # YYYY-MM-DD hh:mm AM/PM
                            elif re.match(r'\d{4}[-_]\d{2}[-_]\d{2}[-_ T]\d{1,2}[-_:]\d{2}\s*[AaPp][Mm]', date_str):
                                # Remove AM/PM for parsing
                                cleaned = re.sub(r'\s*[AaPp][Mm]', '', date_str)
                                cleaned = cleaned.replace('_', '-').replace(' ', '-').replace('T', '-')
                                
                                # Parse date and time components
                                dt_parts = re.match(r'(\d{4})-(\d{2})-(\d{2})-(\d{1,2})-(\d{2})', cleaned)
                                if dt_parts:
                                    year, month, day, hour, minute = map(int, dt_parts.groups())
                                    
                                    # Convert 12-hour to 24-hour format
                                    if am_pm == 'PM' and hour < 12:
                                        hour += 12
                                    elif am_pm == 'AM' and hour == 12:
                                        hour = 0
                                    
                                    original_date = datetime(year, month, day, hour, minute)
                                    extracted_time = True
                                    break
                            
                            # DD-MM-YYYY hh:mm:ss AM/PM or MM-DD-YYYY hh:mm:ss AM/PM
                            elif re.match(r'\d{2}[-_]\d{2}[-_]\d{4}[-_ T]\d{1,2}[-_:]\d{2}[-_:]\d{2}\s*[AaPp][Mm]', date_str):
                                # Remove AM/PM for parsing
                                cleaned = re.sub(r'\s*[AaPp][Mm]', '', date_str)
                                cleaned = cleaned.replace('_', '-').replace(' ', '-').replace('T', '-')
                                
                                # Try DD-MM-YYYY format first
                                try:
                                    dt_parts = re.match(r'(\d{2})-(\d{2})-(\d{4})-(\d{1,2})-(\d{2})-(\d{2})', cleaned)
                                    if dt_parts:
                                        day, month, year, hour, minute, second = map(int, dt_parts.groups())
                                        
                                        # Convert 12-hour to 24-hour format
                                        if am_pm == 'PM' and hour < 12:
                                            hour += 12
                                        elif am_pm == 'AM' and hour == 12:
                                            hour = 0
                                        
                                        original_date = datetime(year, month, day, hour, minute, second)
                                        extracted_time = True
                                        break
                                except ValueError:
                                    # Try MM-DD-YYYY format
                                    dt_parts = re.match(r'(\d{2})-(\d{2})-(\d{4})-(\d{1,2})-(\d{2})-(\d{2})', cleaned)
                                    if dt_parts:
                                        month, day, year, hour, minute, second = map(int, dt_parts.groups())
                                        
                                        # Convert 12-hour to 24-hour format
                                        if am_pm == 'PM' and hour < 12:
                                            hour += 12
                                        elif am_pm == 'AM' and hour == 12:
                                            hour = 0
                                        
                                        original_date = datetime(year, month, day, hour, minute, second)
                                        extracted_time = True
                                        break
                            
                            # DD-MM-YYYY hh:mm AM/PM or MM-DD-YYYY hh:mm AM/PM
                            elif re.match(r'\d{2}[-_]\d{2}[-_]\d{4}[-_ T]\d{1,2}[-_:]\d{2}\s*[AaPp][Mm]', date_str):
                                # Remove AM/PM for parsing
                                cleaned = re.sub(r'\s*[AaPp][Mm]', '', date_str)
                                cleaned = cleaned.replace('_', '-').replace(' ', '-').replace('T', '-')
                                
                                # Try DD-MM-YYYY format first
                                try:
                                    dt_parts = re.match(r'(\d{2})-(\d{2})-(\d{4})-(\d{1,2})-(\d{2})', cleaned)
                                    if dt_parts:
                                        day, month, year, hour, minute = map(int, dt_parts.groups())
                                        
                                        # Convert 12-hour to 24-hour format
                                        if am_pm == 'PM' and hour < 12:
                                            hour += 12
                                        elif am_pm == 'AM' and hour == 12:
                                            hour = 0
                                        
                                        original_date = datetime(year, month, day, hour, minute)
                                        extracted_time = True
                                        break
                                except ValueError:
                                    # Try MM-DD-YYYY format
                                    dt_parts = re.match(r'(\d{2})-(\d{2})-(\d{4})-(\d{1,2})-(\d{2})', cleaned)
                                    if dt_parts:
                                        month, day, year, hour, minute = map(int, dt_parts.groups())
                                        
                                        # Convert 12-hour to 24-hour format
                                        if am_pm == 'PM' and hour < 12:
                                            hour += 12
                                        elif am_pm == 'AM' and hour == 12:
                                            hour = 0
                                        
                                        original_date = datetime(year, month, day, hour, minute)
                                        extracted_time = True
                                        break
                            
                            # Standalone time formats with AM/PM
                            elif re.match(r'\d{1,2}[-_:]\d{2}[-_:]\d{2}\s*[AaPp][Mm]', date_str):
                                # HH:MM:SS AM/PM
                                cleaned = re.sub(r'\s*[AaPp][Mm]', '', date_str).replace('_', ':')
                                time_parts = re.match(r'(\d{1,2}):(\d{2}):(\d{2})', cleaned)
                                if time_parts:
                                    hour, minute, second = map(int, time_parts.groups())
                                    
                                    # Convert 12-hour to 24-hour format
                                    if am_pm == 'PM' and hour < 12:
                                        hour += 12
                                    elif am_pm == 'AM' and hour == 12:
                                        hour = 0
                                    
                                    # Get date from file modification time
                                    date_obj = datetime.fromtimestamp(os.path.getmtime(filepath)).date()
                                    original_date = datetime.combine(date_obj, time(hour, minute, second))
                                    extracted_time = True
                                    break
                            
                            elif re.match(r'\d{1,2}[-_:]\d{2}\s*[AaPp][Mm]', date_str):
                                # HH:MM AM/PM
                                cleaned = re.sub(r'\s*[AaPp][Mm]', '', date_str).replace('_', ':')
                                time_parts = re.match(r'(\d{1,2}):(\d{2})', cleaned)
                                if time_parts:
                                    hour, minute = map(int, time_parts.groups())
                                    
                                    # Convert 12-hour to 24-hour format
                                    if am_pm == 'PM' and hour < 12:
                                        hour += 12
                                    elif am_pm == 'AM' and hour == 12:
                                        hour = 0
                                    
                                    # Get date from file modification time
                                    date_obj = datetime.fromtimestamp(os.path.getmtime(filepath)).date()
                                    original_date = datetime.combine(date_obj, time(hour, minute))
                                    extracted_time = True
                                    break
                            
                            elif re.match(r'\d{1,2}[AaPp][Mm]', date_str):
                                # HAM/PM (e.g., 9AM, 10PM)
                                hour_match = re.match(r'(\d{1,2})[AaPp][Mm]', date_str)
                                if hour_match:
                                    hour = int(hour_match.group(1))
                                    
                                    # Convert 12-hour to 24-hour format
                                    if am_pm == 'PM' and hour < 12:
                                        hour += 12
                                    elif am_pm == 'AM' and hour == 12:
                                        hour = 0
                                    
                                    # Get date from file modification time
                                    date_obj = datetime.fromtimestamp(os.path.getmtime(filepath)).date()
                                    original_date = datetime.combine(date_obj, time(hour, 0))
                                    extracted_time = True
                                    break
                        
                        # 24-hour time formats (high priority)
                        elif re.match(r'\d{4}[-_]\d{2}[-_]\d{2}[-_ T]\d{2}[-_:]\d{2}[-_:]\d{2}', date_str):
                            # YYYY-MM-DD HH:MM:SS
                            cleaned = date_str.replace('_', '-').replace(' ', '-').replace('T', '-')
                            original_date = datetime.strptime(cleaned, '%Y-%m-%d-%H-%M-%S')
                            extracted_time = True
                            time_format = "24-hour"
                            break
                            
                        elif re.match(r'\d{4}[-_]\d{2}[-_]\d{2}[-_ T]\d{2}[-_:]\d{2}', date_str):
                            # YYYY-MM-DD HH:MM
                            cleaned = date_str.replace('_', '-').replace(' ', '-').replace('T', '-')
                            original_date = datetime.strptime(cleaned, '%Y-%m-%d-%H-%M')
                            extracted_time = True
                            time_format = "24-hour"
                            break
                            
                        elif re.match(r'\d{2}[-_]\d{2}[-_]\d{4}[-_ T]\d{2}[-_:]\d{2}[-_:]\d{2}', date_str):
                            # DD-MM-YYYY HH:MM:SS or MM-DD-YYYY HH:MM:SS
                            cleaned = date_str.replace('_', '-').replace(' ', '-').replace('T', '-')
                            try:
                                original_date = datetime.strptime(cleaned, '%d-%m-%Y-%H-%M-%S')
                                extracted_time = True
                                time_format = "24-hour"
                                break
                            except ValueError:
                                original_date = datetime.strptime(cleaned, '%m-%d-%Y-%H-%M-%S')
                                extracted_time = True
                                time_format = "24-hour"
                                break
                                
                        elif re.match(r'\d{2}[-_]\d{2}[-_]\d{4}[-_ T]\d{2}[-_:]\d{2}', date_str):
                            # DD-MM-YYYY HH:MM or MM-DD-YYYY HH:MM
                            cleaned = date_str.replace('_', '-').replace(' ', '-').replace('T', '-')
                            try:
                                original_date = datetime.strptime(cleaned, '%d-%m-%Y-%H-%M')
                                extracted_time = True
                                time_format = "24-hour"
                                break
                            except ValueError:
                                original_date = datetime.strptime(cleaned, '%m-%d-%Y-%H-%M')
                                extracted_time = True
                                time_format = "24-hour"
                                break
                                
                        elif re.match(r'\d{8}[-_]\d{6}', date_str):
                            # YYYYMMDD_HHMMSS or DDMMYYYY_HHMMSS
                            parts = re.split(r'[-_]', date_str)
                            date_part = parts[0]
                            time_part = parts[1]
                            
                            try:
                                # Try YYYYMMDD format
                                date_obj = datetime.strptime(date_part, '%Y%m%d')
                                time_obj = datetime.strptime(time_part, '%H%M%S')
                                original_date = datetime.combine(date_obj.date(), time_obj.time())
                                extracted_time = True
                                break
                            except ValueError:
                                try:
                                    # Try DDMMYYYY format
                                    date_obj = datetime.strptime(date_part, '%d%m%Y')
                                    time_obj = datetime.strptime(time_part, '%H%M%S')
                                    original_date = datetime.combine(date_obj.date(), time_obj.time())
                                    extracted_time = True
                                    break
                                except ValueError:
                                    continue
                                    
                        elif re.match(r'\d{8}[-_]\d{4}', date_str):
                            # YYYYMMDD_HHMM or DDMMYYYY_HHMM
                            parts = re.split(r'[-_]', date_str)
                            date_part = parts[0]
                            time_part = parts[1]
                            
                            try:
                                # Try YYYYMMDD format
                                date_obj = datetime.strptime(date_part, '%Y%m%d')
                                time_obj = datetime.strptime(time_part, '%H%M')
                                original_date = datetime.combine(date_obj.date(), time_obj.time())
                                extracted_time = True
                                break
                            except ValueError:
                                try:
                                    # Try DDMMYYYY format
                                    date_obj = datetime.strptime(date_part, '%d%m%Y')
                                    time_obj = datetime.strptime(time_part, '%H%M')
                                    original_date = datetime.combine(date_obj.date(), time_obj.time())
                                    extracted_time = True
                                    break
                                except ValueError:
                                    continue
                        
                        # Time only patterns - need to combine with file date
                        elif re.match(r'\d{2}[-_:]\d{2}[-_:]\d{2}', date_str):
                            # HH:MM:SS
                            cleaned = date_str.replace('_', ':')
                            time_obj = datetime.strptime(cleaned, '%H:%M:%S').time()
                            # Get date from file modification time
                            date_obj = datetime.fromtimestamp(os.path.getmtime(filepath)).date()
                            original_date = datetime.combine(date_obj, time_obj)
                            extracted_time = True
                            break
                            
                        elif re.match(r'\d{2}[-_:]\d{2}', date_str):
                            # HH:MM
                            cleaned = date_str.replace('_', ':')
                            time_obj = datetime.strptime(cleaned, '%H:%M').time()
                            # Get date from file modification time
                            date_obj = datetime.fromtimestamp(os.path.getmtime(filepath)).date()
                            original_date = datetime.combine(date_obj, time_obj)
                            extracted_time = True
                            break
                            
                        elif re.match(r'\d{6}', date_str):
                            # HHMMSS
                            time_obj = datetime.strptime(date_str, '%H%M%S').time()
                            # Get date from file modification time
                            date_obj = datetime.fromtimestamp(os.path.getmtime(filepath)).date()
                            original_date = datetime.combine(date_obj, time_obj)
                            extracted_time = True
                            break
                            
                        elif re.match(r'\d{4}', date_str) and len(date_str) == 4:
                            # HHMM
                            time_obj = datetime.strptime(date_str, '%H%M').time()
                            # Get date from file modification time
                            date_obj = datetime.fromtimestamp(os.path.getmtime(filepath)).date()
                            original_date = datetime.combine(date_obj, time_obj)
                            extracted_time = True
                            break
                        
                        # Date only patterns (lower priority)
                        elif re.match(r'\d{4}[-_]\d{2}[-_]\d{2}', date_str):
                            # YYYY-MM-DD
                            cleaned = date_str.replace('_', '-')
                            original_date = datetime.strptime(cleaned, '%Y-%m-%d')
                            break
                            
                        elif re.match(r'\d{2}[-_]\d{2}[-_]\d{4}', date_str):
                            # Could be DD-MM-YYYY or MM-DD-YYYY, try both
                            cleaned = date_str.replace('_', '-')
                            try:
                                original_date = datetime.strptime(cleaned, '%d-%m-%Y')
                                break
                            except ValueError:
                                original_date = datetime.strptime(cleaned, '%m-%d-%Y')
                                break
                                
                        elif re.match(r'\d{8}', date_str) and len(date_str) == 8:
                            # Could be YYYYMMDD or DDMMYYYY
                            try:
                                original_date = datetime.strptime(date_str, '%Y%m%d')
                                break
                            except ValueError:
                                original_date = datetime.strptime(date_str, '%d%m%Y')
                                break
                                
                        elif re.match(r'\d{2}[A-Za-z]{3}\d{4}', date_str):
                            # 01Jan2023
                            original_date = datetime.strptime(date_str, '%d%b%Y')
                            break
                            
                        elif re.match(r'[A-Za-z]{3}\d{2}[-_]\d{4}', date_str):
                            # Jan01_2023
                            cleaned = date_str.replace('_', '-')
                            original_date = datetime.strptime(cleaned, '%b%d-%Y')
                            break
                    except ValueError:
                        # If this format fails, continue to the next pattern
                        continue
            
            # If no valid date found, use file modification time as fallback
            if original_date is None:
                original_date = datetime.fromtimestamp(os.path.getmtime(filepath))
                pattern_used = "file_modification_time"
            
            # Store date info for debugging
            if debug_log:
                time_format_info = "None"
                if extracted_time:
                    if 'time_format' in locals():
                        time_format_info = time_format
                    else:
                        time_format_info = "24-hour"  # Default for older time formats
                
                date_info[filename] = {
                    "extracted_date": original_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "pattern_used": pattern_used if pattern_used else "None",
                    "has_time": "Yes" if extracted_time else "No",
                    "time_format": time_format_info,
                    "path": filepath
                }
            
            return original_date
        
        # Sort files by extracted date
        audio_files.sort(key=extract_date_from_filename)
        
        # Print debug information if requested
        if debug_log:
            print("\n===== CHRONOLOGICAL SORTING INFORMATION =====")
            print(f"Found {len(audio_files)} audio files to sort")
            
            # Count files with time information
            files_with_time = sum(1 for filename in date_info if date_info[filename].get("has_time") == "Yes")
            print(f"Files with time information: {files_with_time}/{len(audio_files)}")
            
            # Count files with different time formats
            files_with_12h = sum(1 for filename in date_info if date_info[filename].get("time_format") == "12-hour")
            files_with_24h = sum(1 for filename in date_info if date_info[filename].get("time_format") == "24-hour")
            files_with_date_only = sum(1 for filename in date_info if date_info[filename].get("has_time") == "No")
            
            print(f"Files with 12-hour time: {files_with_12h}")
            print(f"Files with 24-hour time: {files_with_24h}")
            print(f"Files with date only: {files_with_date_only}")
            
            print("\nFiles will be processed in this order:")
            for i, file in enumerate(audio_files, 1):
                filename = Path(file).name
                info = date_info.get(filename, {})
                date_str = info.get("extracted_date", "Unknown")
                pattern = info.get("pattern_used", "Unknown")
                has_time = info.get("has_time", "No")
                time_format = info.get("time_format", "None")
                
                # Format the output to highlight files with time information
                if time_format == "12-hour":
                    time_indicator = "🕒"  # 12-hour format
                elif time_format == "24-hour":
                    time_indicator = "⏰"  # 24-hour format
                else:
                    time_indicator = "📅"  # Date only
                
                print(f"{i}. {time_indicator} {filename} → {date_str} (Format: {time_format}, Pattern: {pattern})")
            
            print("\nℹ️ Legend: 🕒 = 12-hour time format, ⏰ = 24-hour time format, 📅 = Date only")
            print("=============================================\n")
    else:
        # Sort alphabetically
        audio_files.sort()
    
    return audio_files


def create_chunked_json_transcript(transcription_results: List[Dict[str, Any]], output_dir: str, 
                                  total_time: float, success_count: int, failure_count: int, 
                                  model_info: Dict[str, Any], chunk_size_seconds: float = 60.0) -> str:
    """
    Create a chunked JSON transcript file optimized for AI model review.
    
    This function creates a JSON file with transcriptions chunked into time-based segments,
    which is optimized for processing by local AI models like Ollama.
    
    Args:
        transcription_results: List of transcription result dictionaries
        output_dir: Directory to save the JSON file
        total_time: Total processing time in seconds
        success_count: Number of successful transcriptions
        failure_count: Number of failed transcriptions
        model_info: Dictionary with model and device information
        chunk_size_seconds: Target size for each chunk in seconds (default: 60s)
        
    Returns:
        Path to the created JSON file or None if failed
    """
    # Prepare summary information
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract model and device information
    model_name = model_info.get('model_name', 'unknown')
    backend_name = model_info.get('backend_display_name', 'unknown')
    device_name = model_info.get('device_name', 'unknown')
    
    # Create a unified model description
    model_description = f"{backend_name} {model_name}"
    device_description = device_name
    
    # Build the JSON structure
    json_data = {
        "summary": {
            "model_used": model_description,
            "device": device_description,
            "generated": current_time,
            "recordings_transcribed": success_count,
            "total_files": success_count + failure_count,
            "failed_files": failure_count,
            "total_processing_time_seconds": round(total_time, 2)
        },
        "recordings": []
    }
    
    # Process each successful transcription
    for result in transcription_results:
        if not result.get('success', False):
            continue  # Skip failed transcriptions
        
        # Extract filename without path
        filename = Path(result['file_path']).name
        
        # Create recording entry
        recording_data = {
            "filename": filename,
            "duration_seconds": round(result.get('duration', 0.0), 1),
            "language": result.get('language', 'unknown'),
            "processing_time_seconds": round(result.get('processing_time', 0.0), 1),
            "words_count": result.get('words_count', 0),
            "chunks": []
        }
        
        # Process segments into chunks
        raw_segments = result.get('segments', [])
        
        if raw_segments:
            current_chunk = {
                "chunk_id": 1,
                "start": 0.0,
                "end": 0.0,
                "text": ""
            }
            
            chunk_texts = []
            
            for segment in raw_segments:
                segment_start = segment.get('start', 0.0)
                segment_end = segment.get('end', 0.0)
                segment_text = segment.get('text', '').strip()
                
                if not segment_text:
                    continue
                
                # Check if this segment should start a new chunk
                if current_chunk["text"] and (segment_start - current_chunk["start"] >= chunk_size_seconds):
                    # Finalize current chunk
                    current_chunk["end"] = segment_start  # End at the start of the next segment
                    current_chunk["text"] = " ".join(chunk_texts)
                    recording_data["chunks"].append(current_chunk.copy())
                    
                    # Start new chunk
                    current_chunk["chunk_id"] += 1
                    current_chunk["start"] = segment_start
                    current_chunk["end"] = segment_end
                    chunk_texts = [segment_text]
                else:
                    # Add to current chunk
                    if not current_chunk["text"]:
                        current_chunk["start"] = segment_start
                    current_chunk["end"] = segment_end
                    chunk_texts.append(segment_text)
            
            # Add the last chunk if it has content
            if chunk_texts:
                current_chunk["text"] = " ".join(chunk_texts)
                recording_data["chunks"].append(current_chunk)
        else:
            # If no segments are available, create a single chunk with the full transcription
            full_text = result.get('transcription', '').strip()
            if full_text:
                recording_data["chunks"].append({
                    "chunk_id": 1,
                    "start": 0.0,
                    "end": result.get('duration', 0.0),
                    "text": full_text
                })
        
        # Add recording data to the JSON structure
        json_data["recordings"].append(recording_data)
    
    # Save the JSON file
    json_path = os.path.join(output_dir, "recordings_chunked.json")
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        return json_path
    except Exception as e:
        print(f"Error creating chunked JSON transcript: {e}")
        return None


def create_json_transcript(transcription_results: List[Dict[str, Any]], output_dir: str, 
                          total_time: float, success_count: int, failure_count: int, 
                          model_info: Dict[str, Any]) -> str:
    """Create a structured JSON transcript file optimized for AI model review."""
    
    # Prepare summary information
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract model and device information
    model_name = model_info.get('model_name', 'unknown')
    backend_name = model_info.get('backend_display_name', 'unknown')
    device_name = model_info.get('device_name', 'unknown')
    
    # Create a unified model description
    model_description = f"{backend_name} {model_name}"
    device_description = device_name
    
    # Build the JSON structure
    json_data = {
        "transcription_summary": {
            "generated": current_time,
            "model": model_description,
            "device": device_description,
            "total_files": success_count + failure_count,
            "successful_files": success_count,
            "failed_files": failure_count,
            "total_processing_time_seconds": round(total_time, 2)
        },
        "recordings": []
    }
    
    # Process each successful transcription
    for result in transcription_results:
        if not result.get('success', False):
            continue  # Skip failed transcriptions
        
        # Extract filename without path
        filename = Path(result['file_path']).name
        
        # Prepare segments data
        segments = []
        raw_segments = result.get('segments', [])
        
        for segment in raw_segments:
            # Extract text and clean it up
            text = segment.get('text', '').strip()
            if not text:
                continue
                
            segments.append({
                "start": round(segment.get('start', 0.0), 1),
                "end": round(segment.get('end', 0.0), 1), 
                "text": text
            })
        
        # Create recording entry
        recording_data = {
            "filename": filename,
            "duration_seconds": round(result.get('duration', 0.0), 1),
            "language": result.get('language', 'unknown'),
            "processing_time_seconds": round(result.get('processing_time', 0.0), 1),
            "words_count": result.get('words_count', 0),
            "segments": segments
        }
        
        json_data["recordings"].append(recording_data)
    
    # Save the JSON file
    json_path = os.path.join(output_dir, "combined_transcript.json")
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        return json_path
    except Exception as e:
        print(f"Error creating JSON transcript: {e}")
        return None


def escape_js_string(s: str) -> str:
    """
    Escape a string for safe use in JavaScript string literals.
    
    Args:
        s: String to escape
        
    Returns:
        Escaped string safe for JavaScript
    """
    if not s:
        return s
    
    # Replace problematic characters that can break JavaScript strings
    s = s.replace('\\', '\\\\')  # Escape backslashes first
    s = s.replace("'", "\\'")    # Escape single quotes
    s = s.replace('"', '\\"')    # Escape double quotes
    s = s.replace('\n', '\\n')   # Escape newlines
    s = s.replace('\r', '\\r')   # Escape carriage returns
    s = s.replace('\t', '\\t')   # Escape tabs
    return s


def escape_filename_for_html(filename: str) -> str:
    """
    Escape a filename for safe use in HTML src attributes.
    This handles URL fragments and other problematic characters.
    
    Args:
        filename: Filename to escape
        
    Returns:
        URL-encoded filename safe for HTML src attributes
    """
    if not filename:
        return filename
    
    import urllib.parse
    # URL encode the filename to handle special characters like #
    # But preserve the path separator if it exists
    if '/' in filename:
        parts = filename.split('/')
        encoded_parts = [urllib.parse.quote(part, safe='') for part in parts]
        return '/'.join(encoded_parts)
    else:
        return urllib.parse.quote(filename, safe='')


def _get_category_color(weight: float) -> tuple:
    """
    Get color for a category based on its weight (severity).
    Returns (bg_color, text_color, border_color) tuple.
    """
    if weight >= 40:  # Very high severity - threats, physical/sexual abuse, child harm
        return ('#fee', '#8b0000', '#dc3545')  # Light red bg, dark red text, red border
    elif weight >= 25:  # High severity - evidence tampering, version change
        return ('#fff3cd', '#856404', '#ffc107')  # Light yellow bg, dark yellow text, yellow border
    elif weight >= 15:  # Medium-high severity - withdrawal, coercive control, stalking
        return ('#ffe6cc', '#cc6600', '#fd7e14')  # Light orange bg, dark orange text, orange border
    elif weight >= 8:  # Medium severity - isolation, emotional abuse
        return ('#e7f3ff', '#004085', '#17a2b8')  # Light blue bg, dark blue text, cyan border
    else:  # Lower severity
        return ('#f0f0f0', '#555', '#6c757d')  # Light gray bg, gray text, gray border


def _highlight_keywords(text: str, dv_analysis_for_file: Optional[Dict[str, Any]], 
                       category_weights: Optional[Dict[str, float]]) -> str:
    """
    Highlight keywords in text based on DV analysis matches.
    
    Args:
        text: The text to highlight
        dv_analysis_for_file: DV analysis results for this specific file (from analyze_batch)
        category_weights: Dictionary mapping category names to weights
        
    Returns:
        HTML string with highlighted keywords
    """
    if not text:
        return ''
    
    if not dv_analysis_for_file or not category_weights:
        return escape_html(text)
    
    matches = dv_analysis_for_file.get('matches', {})
    if not matches:
        return escape_html(text)
    
    # Escape HTML first
    escaped_text = escape_html(text)
    
    # Collect all unique words to highlight, sorted by weight (highest first)
    # This ensures that if a word appears in multiple categories, we use the highest weight
    word_highlights = {}  # word (lowercase) -> highlight info with highest weight
    
    for category, category_matches in matches.items():
        weight = category_weights.get(category, 1.0)
        bg_color, text_color, border_color = _get_category_color(weight)
        
        if isinstance(category_matches, list):
            for match in category_matches:
                if isinstance(match, dict):
                    word = match.get('word', '')
                else:
                    word = str(match)
                
                if not word:
                    continue
                
                word_lower = word.lower()
                # Only update if this word hasn't been seen, or if this category has higher weight
                if word_lower not in word_highlights or weight > word_highlights[word_lower]['weight']:
                    word_highlights[word_lower] = {
                        'word': word,  # Keep original case
                        'category': category,
                        'weight': weight,
                        'bg_color': bg_color,
                        'text_color': text_color,
                        'border_color': border_color
                    }
        elif isinstance(category_matches, (str, list)):
            # Handle case where category_matches might be a simple list of strings
            words = category_matches if isinstance(category_matches, list) else [category_matches]
            for word in words:
                if not word:
                    continue
                word_lower = word.lower()
                if word_lower not in word_highlights or weight > word_highlights[word_lower]['weight']:
                    word_highlights[word_lower] = {
                        'word': word,
                        'category': category,
                        'weight': weight,
                        'bg_color': bg_color,
                        'text_color': text_color,
                        'border_color': border_color
                    }
    
    # Sort by weight (descending) and then alphabetically to process in consistent order
    sorted_highlights = sorted(word_highlights.items(), key=lambda x: (-x[1]['weight'], x[0]))
    
    # Replace matches with highlighted versions (process in reverse order of length to avoid partial matches)
    # Sort by word length (longest first) to avoid issues with overlapping words
    sorted_by_length = sorted(sorted_highlights, key=lambda x: len(x[1]['word']), reverse=True)
    
    for word_lower, highlight_info in sorted_by_length:
        word = highlight_info['word']
        escaped_word = re.escape(word)
        
        # Create highlight span
        highlight_span = (
            f'<span style="background-color: {highlight_info["bg_color"]}; '
            f'color: {highlight_info["text_color"]}; '
            f'border: 1px solid {highlight_info["border_color"]}; '
            f'padding: 2px 4px; '
            f'border-radius: 3px; '
            f'font-weight: bold; '
            f'title="Category: {highlight_info["category"]} (Weight: {highlight_info["weight"]:.1f})">'
            f'{word}</span>'
        )
        
        # Replace word with highlighted version (case-insensitive, word boundaries)
        # Word boundaries prevent matching inside existing HTML tags
        escaped_text = re.sub(
            r'\b' + escaped_word + r'\b',
            highlight_span,
            escaped_text,
            flags=re.IGNORECASE
        )
    
    return escaped_text


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ''
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))


def create_html_report(transcriptions: List[Dict[str, Any]], output_dir: str, 
                      total_time: float, success_count: int, failure_count: int,
                      dv_analysis: Optional[Dict[str, Any]] = None,
                      progress_callback: Optional[Callable[[str, int], None]] = None) -> str:
    """
    Create an HTML report with all transcriptions, enhanced hyperlinks, and timestamped segments.
    
    Args:
        transcriptions: List of transcription results
        output_dir: Directory to save the HTML report
        total_time: Total processing time
        success_count: Number of successful transcriptions
        failure_count: Number of failed transcriptions
        dv_analysis: Optional DV analysis results for keyword highlighting and TOC
        progress_callback: Optional callback function(message, percentage) for progress updates
    """
    # Note: Audio files are not copied here - they will be copied only when creating the zip package
    
    def _progress(message: str, percentage: int):
        """Helper to call progress callback with message and percentage."""
        if progress_callback:
            progress_callback(message, percentage)
    
    _progress("Compiling transcription data...", 0)
    
    # Get category weights from DV analyzer if available
    category_weights = None
    if dv_analysis:
        try:
            from dv_review import DVWordListAnalyzer
            analyzer = DVWordListAnalyzer()
            category_weights = analyzer.category_weights
        except ImportError:
            pass
    
    # Create a map of filename to DV analysis for quick lookup
    dv_analysis_map = {}
    if dv_analysis and 'analyses' in dv_analysis:
        for analysis in dv_analysis['analyses']:
            dv_analysis_map[analysis['filename']] = analysis
    
    _progress("Generating HTML report structure...", 5)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Transcription Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            padding-top: 100px;
            background-color: #f5f5f5;
        }}
        .search-container {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: white;
            padding: 15px 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
            border-bottom: 2px solid #007bff;
        }}
        .search-wrapper {{
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .search-input-group {{
            flex: 1;
            min-width: 250px;
            display: flex;
            align-items: center;
            gap: 10px;
            background: #f8f9fa;
            padding: 8px 12px;
            border-radius: 6px;
            border: 2px solid #dee2e6;
        }}
        .search-input-group:focus-within {{
            border-color: #007bff;
            background: white;
        }}
        .search-input {{
            flex: 1;
            border: none;
            outline: none;
            background: transparent;
            font-size: 14px;
            padding: 4px 0;
        }}
        .search-input::placeholder {{
            color: #6c757d;
        }}
        .search-clear {{
            background: none;
            border: none;
            cursor: pointer;
            font-size: 18px;
            color: #6c757d;
            padding: 0;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .search-clear:hover {{
            color: #dc3545;
        }}
        .search-options {{
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .search-option {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 13px;
            color: #495057;
        }}
        .search-option input[type="checkbox"] {{
            cursor: pointer;
        }}
        .search-navigation {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .search-nav-btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            transition: background-color 0.2s;
        }}
        .search-nav-btn:hover:not(:disabled) {{
            background: #0056b3;
        }}
        .search-nav-btn:disabled {{
            background: #6c757d;
            cursor: not-allowed;
            opacity: 0.6;
        }}
        .search-results-info {{
            font-size: 13px;
            color: #495057;
            min-width: 100px;
            text-align: center;
        }}
        .search-highlight {{
            background-color: #ffeb3b;
            padding: 2px 0;
            border-radius: 2px;
            font-weight: 500;
        }}
        .search-highlight.active {{
            background-color: #ff9800;
            color: white;
            box-shadow: 0 0 0 2px rgba(255, 152, 0, 0.5);
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e9ecef;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .transcription {{
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            background: #fff;
        }}
        .transcription.success {{
            border-left: 4px solid #28a745;
        }}
        .transcription.error {{
            border-left: 4px solid #dc3545;
        }}
        .file-info {{
            margin-bottom: 15px;
        }}
        .filename {{
            font-weight: bold;
            font-size: 1.1em;
            color: #2c3e50;
            margin-bottom: 8px;
        }}
        .file-actions {{
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
        }}
        .audio-link {{
            color: #007bff;
            text-decoration: none;
            font-size: 0.9em;
            padding: 5px 10px;
            border: 1px solid #007bff;
            border-radius: 4px;
            transition: all 0.3s;
        }}
        .audio-link:hover {{
            background-color: #007bff;
            color: white;
            text-decoration: none;
        }}
        .folder-link {{
            color: #6c757d;
            text-decoration: none;
            font-size: 0.9em;
            padding: 5px 10px;
            border: 1px solid #6c757d;
            border-radius: 4px;
            transition: all 0.3s;
        }}
        .folder-link:hover {{
            background-color: #6c757d;
            color: white;
            text-decoration: none;
        }}
        .transcription-content {{
            background: #f8f9fa;
            border-radius: 4px;
            overflow: hidden;
        }}
        .transcription-content[aria-hidden="true"] {{
            position: absolute;
            left: -9999px;
            visibility: hidden;
            user-select: none;
            pointer-events: none;
        }}
        .full-text[aria-hidden="true"] {{
            position: absolute;
            left: -9999px;
            visibility: hidden;
            user-select: none;
            pointer-events: none;
        }}
        .segment {{
            display: flex;
            border-bottom: 1px solid #e9ecef;
            min-height: 50px;
        }}
        .segment:last-child {{
            border-bottom: none;
        }}
        .timestamp-col {{
            background: #e9ecef;
            padding: 10px 15px;
            min-width: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            font-weight: bold;
            color: #495057;
        }}
        .timestamp-text {{
            color: #495057;
            padding: 4px 8px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.7);
            display: block;
            width: 100%;
            text-align: center;
            border: 1px solid #dee2e6;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .timestamp-text:hover {{
            background: #007bff;
            color: white;
            border-color: #007bff;
            transform: scale(1.05);
        }}
        .segment.active {{
            background: linear-gradient(90deg, #fff3cd 0%, #ffffff 100%);
            border-left: 4px solid #ffc107;
        }}
        .segment.active .timestamp-text {{
            background: #ffc107;
            color: #212529;
            border-color: #ffc107;
            font-weight: bold;
        }}
        .audio-player {{
            width: 100%;
            margin: 15px 0;
            border-radius: 8px;
            background: #f8f9fa;
            padding: 15px;
            border: 1px solid #dee2e6;
        }}
        .audio-controls {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}
        .control-btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
            transition: background-color 0.2s;
        }}
        .control-btn:hover {{
            background: #0056b3;
        }}
        .control-btn:disabled {{
            background: #6c757d;
            cursor: not-allowed;
        }}
        .playback-speed {{
            background: #6c757d;
            color: white;
            border: none;
            padding: 6px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8em;
        }}
        .time-display {{
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #495057;
            margin-left: auto;
        }}
        .audio-element {{
            width: 100%;
            margin-top: 10px;
        }}
        .text-col {{
            padding: 10px 15px;
            flex: 1;
            display: flex;
            align-items: center;
            background: white;
        }}
        .segment-text {{
            word-wrap: break-word;
            line-height: 1.4;
        }}
        .full-text {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Courier New', monospace;
            line-height: 1.5;
        }}
        .error-message {{
            color: #dc3545;
            font-style: italic;
        }}
        .duration {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .timestamp {{
            color: #6c757d;
            font-size: 0.8em;
            text-align: right;
            margin-top: 20px;
        }}
        .segment-toggle {{
            margin: 15px 0;
            text-align: center;
        }}
        .toggle-btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 500;
            transition: background-color 0.2s ease;
            box-shadow: 0 2px 4px rgba(0,123,255,0.2);
        }}
        .toggle-btn:hover {{
            background: #0056b3;
            box-shadow: 0 3px 6px rgba(0,123,255,0.3);
        }}
        .toggle-btn:active {{
            transform: translateY(1px);
        }}
     </style>
         <script>
        // Global variables to track audio players and active segments
        const audioPlayers = new Map();
        const activeSegments = new Map();
        
        function openFileLocation(filePath) {{
            // Handle both relative and absolute paths
            // For relative paths, just open the file directly
            if (!filePath.includes('/') && !filePath.includes('\\\\') && !filePath.startsWith('file:///')) {{
                // This is a relative path (just filename)
                window.open(filePath, '_blank');
                return;
            }}
            
            // For absolute paths (backward compatibility)
            // Try to open the file directly (may auto-play depending on system settings)
            window.open(filePath, '_blank');
            
            // For Windows, we can try to open in explorer
            if (navigator.platform.indexOf('Win') !== -1) {{
                // This will work in some browsers/contexts
                try {{
                    const explorerPath = 'file:///' + filePath.replace(/\\//g, '\\\\\\\\');
                    window.open(explorerPath, '_blank');
                }} catch(e) {{
                    console.log('Could not open in explorer:', e);
                }}
            }}
        }}
        
        function initializeAudioPlayer(playerId, audioSrc) {{
            const audioElement = document.getElementById('audio-' + playerId);
            const timeDisplay = document.getElementById('time-' + playerId);
            
            if (!audioElement) return;
            
            // Store player reference
            audioPlayers.set(playerId, audioElement);
            
            // Add event listeners for time updates
            audioElement.addEventListener('timeupdate', function() {{
                updateTimeDisplay(playerId);
                highlightCurrentSegment(playerId);
            }});
            
            audioElement.addEventListener('loadedmetadata', function() {{
                updateTimeDisplay(playerId);
            }});
            
            // Add error handling for audio loading
            audioElement.addEventListener('error', function(e) {{
                console.error('Audio error:', audioElement.error?.message || 'Unknown error');
                
                const errorMsg = 'Audio playback error';
                if (timeDisplay) {{
                    timeDisplay.textContent = errorMsg;
                    timeDisplay.style.color = '#dc3545';
                }}
            }});
            
            // Reset error styling when audio can play
            audioElement.addEventListener('canplay', function() {{
                if (timeDisplay && timeDisplay.style.color === '#dc3545') {{
                    timeDisplay.style.color = '#495057';  // Reset error color
                }}
            }});
            
            // Initialize time display
            updateTimeDisplay(playerId);
        }}
        
        function updateTimeDisplay(playerId) {{
            const audioElement = audioPlayers.get(playerId);
            const timeDisplay = document.getElementById('time-' + playerId);
            
            if (!audioElement || !timeDisplay) return;
            
            const current = audioElement.currentTime || 0;
            const duration = audioElement.duration || 0;
            
            timeDisplay.textContent = formatTime(current) + ' / ' + formatTime(duration);
        }}
        
        function formatTime(seconds) {{
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            
            if (hours > 0) {{
                return hours.toString().padStart(2, '0') + ':' + 
                       minutes.toString().padStart(2, '0') + ':' + 
                       secs.toString().padStart(2, '0');
            }} else {{
                return minutes.toString().padStart(2, '0') + ':' + 
                       secs.toString().padStart(2, '0');
            }}
        }}
        
        function playPauseAudio(playerId) {{
            const audioElement = audioPlayers.get(playerId);
            if (!audioElement) return;
            
            if (audioElement.paused) {{
                audioElement.play().catch(e => {{
                    console.error('Error playing audio for player ' + playerId + ':', e);
                    const timeDisplay = document.getElementById('time-' + playerId);
                    if (timeDisplay) {{
                        timeDisplay.textContent = 'Playback error - check console';
                        timeDisplay.style.color = '#dc3545';
                    }}
                    
                    // Try to reload the audio if it's a loading issue
                    if (e.name === 'NotSupportedError' || e.name === 'NotAllowedError') {{
                        console.log('Attempting to reload audio source...');
                        audioElement.load();
                    }}
                }});
            }} else {{
                audioElement.pause();
            }}
        }}
        
        function seekToTime(playerId, seconds) {{
            const audioElement = audioPlayers.get(playerId);
            if (!audioElement) return;
            
            // Check if audio is ready for seeking
            if (audioElement.readyState < 2) {{
                console.log('Audio not ready for seeking, waiting...');
                audioElement.addEventListener('canplay', function() {{
                    audioElement.currentTime = seconds;
                    if (audioElement.paused) {{
                        audioElement.play().catch(e => {{
                            console.log('Auto-play prevented by browser:', e);
                        }});
                    }}
                }}, {{ once: true }});
                return;
            }}
            
            audioElement.currentTime = seconds;
            
            // Auto-play when seeking to a timestamp
            if (audioElement.paused) {{
                audioElement.play().catch(e => {{
                    console.log('Auto-play prevented by browser:', e);
                }});
            }}
        }}
        
        function skipTime(playerId, seconds) {{
            const audioElement = audioPlayers.get(playerId);
            if (!audioElement) return;
            
            audioElement.currentTime = Math.max(0, audioElement.currentTime + seconds);
        }}
        
        function changePlaybackSpeed(playerId, speed) {{
            const audioElement = audioPlayers.get(playerId);
            if (!audioElement) return;
            
            audioElement.playbackRate = speed;
            
            // Update button text
            const speedBtn = document.getElementById('speed-' + playerId);
            if (speedBtn) {{
                speedBtn.textContent = speed + 'x';
            }}
        }}
        
        function cyclePlaybackSpeed(playerId) {{
            const speeds = [0.5, 0.75, 1, 1.25, 1.5, 2];
            const audioElement = audioPlayers.get(playerId);
            if (!audioElement) return;
            
            const currentSpeed = audioElement.playbackRate;
            let nextSpeedIndex = speeds.indexOf(currentSpeed) + 1;
            if (nextSpeedIndex >= speeds.length) {{
                nextSpeedIndex = 0;
            }}
            
            const newSpeed = speeds[nextSpeedIndex];
            changePlaybackSpeed(playerId, newSpeed);
        }}
        
        function highlightCurrentSegment(playerId) {{
            const audioElement = audioPlayers.get(playerId);
            if (!audioElement) return;
            
            const currentTime = audioElement.currentTime;
            const segments = document.querySelectorAll('#segments-' + playerId + ' .segment');
            
            // Clear previous active segment
            const previousActive = activeSegments.get(playerId);
            if (previousActive) {{
                previousActive.classList.remove('active');
            }}
            
            // Find and highlight current segment
            let activeSegment = null;
            segments.forEach(segment => {{
                const timestampElement = segment.querySelector('.timestamp-text');
                if (!timestampElement) return;
                
                // Extract start time from data attribute or parse from text
                const startTime = parseFloat(timestampElement.dataset.seconds) || 0;
                const nextSegment = segment.nextElementSibling;
                const endTime = nextSegment ? 
                    (parseFloat(nextSegment.querySelector('.timestamp-text')?.dataset.seconds) || Infinity) : 
                    Infinity;
                
                if (currentTime >= startTime && currentTime < endTime) {{
                    segment.classList.add('active');
                    activeSegment = segment;
                    
                    // Auto-scroll to keep active segment visible
                    scrollToActiveSegment(segment);
                }}
            }});
            
            activeSegments.set(playerId, activeSegment);
        }}
        
        function scrollToActiveSegment(segment) {{
            if (!segment) return;
            
            const container = segment.closest('.transcription-content');
            if (!container) return;
            
            const containerRect = container.getBoundingClientRect();
            const segmentRect = segment.getBoundingClientRect();
            
            // Check if segment is not fully visible
            if (segmentRect.top < containerRect.top || segmentRect.bottom > containerRect.bottom) {{
                segment.scrollIntoView({{
                    behavior: 'smooth',
                    block: 'center',
                    inline: 'nearest'
                }});
            }}
        }}
        
        function toggleTranscriptionView(transcriptionId) {{
            const segmentView = document.getElementById('segments-' + transcriptionId);
            const fullView = document.getElementById('full-' + transcriptionId);
            const toggleBtn = document.getElementById('toggle-' + transcriptionId);
            
            // Check current state and toggle
            if (segmentView.style.display === 'none' || segmentView.style.display === '') {{
                // Show segments view
                segmentView.style.display = 'block';
                segmentView.setAttribute('aria-hidden', 'false');
                fullView.style.display = 'none';
                fullView.setAttribute('aria-hidden', 'true');
                toggleBtn.textContent = 'Show Full Text';
                
                // Initialize segment highlighting when segments view is shown
                highlightCurrentSegment(transcriptionId);
            }} else {{
                // Show full text view
                segmentView.style.display = 'none';
                segmentView.setAttribute('aria-hidden', 'true');
                fullView.style.display = 'block';
                fullView.setAttribute('aria-hidden', 'false');
                toggleBtn.textContent = 'Show Timestamped Segments';
            }}
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            // Only activate if not typing in an input field
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            // Find the currently visible audio player (if any)
            let activePlayerId = null;
            for (const [playerId, element] of audioPlayers) {{
                const playerContainer = element.closest('.transcription');
                if (playerContainer && isElementVisible(playerContainer)) {{
                    activePlayerId = playerId;
                    break;
                }}
            }}
            
            if (!activePlayerId) return;
            
            const audioElement = audioPlayers.get(activePlayerId);
            if (!audioElement) return;
            
            switch(e.key) {{
                case ' ':
                    e.preventDefault();
                    if (audioElement.paused) {{
                        audioElement.play();
                    }} else {{
                        audioElement.pause();
                    }}
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    skipTime(activePlayerId, e.shiftKey ? -30 : -5);
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    skipTime(activePlayerId, e.shiftKey ? 30 : 5);
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    audioElement.volume = Math.min(1, audioElement.volume + 0.1);
                    break;
                case 'ArrowDown':
                    e.preventDefault();
                    audioElement.volume = Math.max(0, audioElement.volume - 0.1);
                    break;
            }}
        }});
        
        function isElementVisible(element) {{
            const rect = element.getBoundingClientRect();
            return rect.top < window.innerHeight && rect.bottom > 0;
        }}
        
        // Audio player management
        
        // Search functionality
        let searchMatches = [];
        let currentMatchIndex = -1;
        let searchQuery = '';
        let caseSensitive = false;
        let wholeWord = false;
        
        // Debounce utility for search input
        function debounce(func, wait) {{
            let timeout;
            return function executedFunction(...args) {{
                const later = () => {{
                    clearTimeout(timeout);
                    func(...args);
                }};
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            }};
        }}
        
        function performSearch() {{
            const searchInput = document.getElementById('searchInput');
            const query = searchInput.value.trim();
            const caseCheckbox = document.getElementById('caseSensitive');
            const wholeWordCheckbox = document.getElementById('wholeWord');
            
            searchQuery = query;
            caseSensitive = caseCheckbox.checked;
            wholeWord = wholeWordCheckbox.checked;
            
            // Clear previous highlights
            clearHighlights();
            searchMatches = [];
            currentMatchIndex = -1;
            
            if (!query) {{
                updateSearchResultsInfo();
                return;
            }}
            
            // Get all searchable elements (already indexed) - only search visible elements
            // Filter out elements that are hidden (aria-hidden="true" or inside hidden containers)
            const allElements = Array.from(document.querySelectorAll('.segment-text, .full-text, .filename'));
            const searchableElements = allElements.filter(el => {{
                // Check if element or its parent is hidden
                const parent = el.closest('.transcription-content, .full-text');
                if (parent && parent.getAttribute('aria-hidden') === 'true') {{
                    return false;
                }}
                // Check if element itself is hidden
                if (el.getAttribute('aria-hidden') === 'true') {{
                    return false;
                }}
                // Check if parent container is hidden via display:none
                if (parent && (parent.style.display === 'none' || window.getComputedStyle(parent).display === 'none')) {{
                    return false;
                }}
                return true;
            }});
            const totalElements = searchableElements.length;
            let currentIndex = 0;
            let matchCount = 0;
            const CHUNK_SIZE = 25; // Process 25 elements per frame
            const pendingUpdates = new Map(); // element -> highlightedHTML
            
            // Show search status
            updateSearchResultsInfo('Searching...');
            
            function processChunk() {{
                const endIndex = Math.min(currentIndex + CHUNK_SIZE, totalElements);
                
                for (let i = currentIndex; i < endIndex; i++) {{
                    const element = searchableElements[i];
                    // Use pre-indexed text (much faster)
                    const originalText = element.dataset.originalText || element.textContent;
                    
                    // Use pre-computed lowercase for case-insensitive search
                    const searchText = caseSensitive ? originalText : (element.dataset.lowerText || originalText.toLowerCase());
                    const searchQueryLower = caseSensitive ? query : query.toLowerCase();
                    
                    const elementMatches = findMatchesFast(originalText, searchText, searchQueryLower, query);
                    
                    if (elementMatches.length > 0) {{
                        // Build highlighted version
                        const highlightedText = buildHighlightedText(originalText, elementMatches, matchCount);
                        pendingUpdates.set(element, highlightedText);
                        
                        // Store match info
                        elementMatches.forEach(match => {{
                            searchMatches.push({{
                                element: element,
                                index: matchCount++,
                                textNode: element
                            }});
                        }});
                    }}
                }}
                
                currentIndex = endIndex;
                
                // Apply DOM updates in batches to avoid too many reflows
                if (pendingUpdates.size >= 20 || currentIndex >= totalElements) {{
                    applyDOMUpdates(pendingUpdates);
                    pendingUpdates.clear();
                }}
                
                if (currentIndex < totalElements) {{
                    requestAnimationFrame(processChunk);
                }} else {{
                    // Final batch update
                    applyDOMUpdates(pendingUpdates);
                    
                    if (searchMatches.length > 0) {{
                        currentMatchIndex = 0;
                        scrollToMatch(0);
                    }}
                    updateSearchResultsInfo();
                }}
            }}
            
            processChunk();
        }}
        
        function findMatchesFast(originalText, searchText, searchQueryLower, query) {{
            const matches = [];
            
            if (wholeWord) {{
                // Pre-compile regex for better performance
                const pattern = `\\\\b${{escapeRegex(searchQueryLower)}}\\\\b`;
                const regex = new RegExp(pattern, 'g');
                let match;
                while ((match = regex.exec(searchText)) !== null) {{
                    matches.push({{
                        index: match.index,
                        length: match[0].length,
                        text: originalText.substring(match.index, match.index + match[0].length)
                    }});
                }}
            }} else {{
                // Fast substring search using pre-lowercased text
                let index = 0;
                while ((index = searchText.indexOf(searchQueryLower, index)) !== -1) {{
                    matches.push({{
                        index: index,
                        length: query.length,
                        text: originalText.substring(index, index + query.length)
                    }});
                    index += query.length; // Move past this match
                }}
            }}
            
            return matches;
        }}
        
        function buildHighlightedText(text, matches, startMatchIndex) {{
            if (matches.length === 0) return text;
            
            // Sort matches by position (descending) to avoid offset issues
            const sortedMatches = [...matches].sort((a, b) => b.index - a.index);
            let result = text;
            let matchCount = startMatchIndex;
            
            sortedMatches.forEach(match => {{
                const before = result.substring(0, match.index);
                const after = result.substring(match.index + match.length);
                const escapedText = escapeHtml(match.text);
                const highlight = `<span class="search-highlight" data-match-index="${{matchCount}}">${{escapedText}}</span>`;
                result = before + highlight + after;
                matchCount++;
            }});
            
            return result;
        }}
        
        function applyDOMUpdates(updates) {{
            // Batch DOM updates to minimize reflows
            updates.forEach((newHTML, element) => {{
                element.innerHTML = newHTML;
            }});
        }}
        
        function escapeRegex(str) {{
            return str.replace(/[.*+?^${{}}()|[\]\\\\]/g, '\\\\$&');
        }}
        
        function escapeHtml(str) {{
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }}
        
        function clearHighlights() {{
            // Fast clear using pre-indexed text
            const searchableElements = document.querySelectorAll('.segment-text, .full-text, .filename');
            const total = searchableElements.length;
            let cleared = 0;
            const CHUNK_SIZE = 100;
            
            function clearChunk() {{
                const end = Math.min(cleared + CHUNK_SIZE, total);
                
                for (let i = cleared; i < end; i++) {{
                    const element = searchableElements[i];
                    if (element.dataset.originalText) {{
                        element.textContent = element.dataset.originalText;
                    }} else {{
                        // Fallback: remove highlight spans
                        const highlights = element.querySelectorAll('.search-highlight');
                        if (highlights.length > 0) {{
                            const text = element.textContent;
                            element.textContent = text;
                        }}
                    }}
                }}
                
                cleared = end;
                
                if (cleared < total) {{
                    requestAnimationFrame(clearChunk);
                }}
            }}
            
            clearChunk();
        }}
        
        function updateSearchResultsInfo(status) {{
            const infoElement = document.getElementById('searchResultsInfo');
            if (status) {{
                infoElement.textContent = status;
                infoElement.style.color = '#6c757d';
                return;
            }}
            
            if (searchMatches.length === 0) {{
                if (searchQuery) {{
                    infoElement.textContent = 'No matches found';
                    infoElement.style.color = '#dc3545';
                }} else {{
                    infoElement.textContent = '';
                }}
            }} else {{
                infoElement.textContent = `${{currentMatchIndex + 1}} / ${{searchMatches.length}}`;
                infoElement.style.color = '#495057';
            }}
            
            // Update navigation buttons
            const prevBtn = document.getElementById('prevMatch');
            const nextBtn = document.getElementById('nextMatch');
            prevBtn.disabled = searchMatches.length === 0 || currentMatchIndex <= 0;
            nextBtn.disabled = searchMatches.length === 0 || currentMatchIndex >= searchMatches.length - 1;
        }}
        
        function scrollToMatch(index) {{
            if (index < 0 || index >= searchMatches.length) return;
            
            // Remove active class from all highlights
            document.querySelectorAll('.search-highlight.active').forEach(el => {{
                el.classList.remove('active');
            }});
            
            // Find the highlight element with matching index
            const highlight = document.querySelector(`.search-highlight[data-match-index="${{index}}"]`);
            if (!highlight) return;
            
            // Find parent transcription container (recording-{idx})
            let parentContainer = highlight.closest('[id^="recording-"]');
            if (!parentContainer) {{
                // Fallback: find any parent with a visible transcription
                parentContainer = highlight.closest('.transcription');
            }}
            
            // Ensure the parent container is visible
            if (parentContainer) {{
                // Check if the transcription is collapsed and expand it
                const transcriptDiv = parentContainer.querySelector('.transcription-content');
                if (transcriptDiv && transcriptDiv.style.display === 'none') {{
                    // Find the toggle button and click it to expand
                    const transcriptionId = parentContainer.id.replace('recording-', '');
                    const toggleBtn = document.getElementById('toggle-' + transcriptionId);
                    if (toggleBtn) {{
                        toggleBtn.click();
                    }}
                }}
                
                // Ensure the transcription container itself is visible
                const containerStyle = window.getComputedStyle(parentContainer);
                if (containerStyle.display === 'none') {{
                    parentContainer.style.display = 'block';
                }}
            }}
            
            // Check if highlight is in a hidden view and switch to correct view
            // Find the segments view container (parent with id starting with "segments-")
            let segmentsView = highlight.closest('.transcription-content');
            if (segmentsView && segmentsView.id && segmentsView.id.startsWith('segments-')) {{
                // Match is in segments view - ensure it's visible
                const transcriptionId = segmentsView.id.replace('segments-', '');
                const toggleBtn = document.getElementById('toggle-' + transcriptionId);
                
                if (segmentsView && (segmentsView.style.display === 'none' || window.getComputedStyle(segmentsView).display === 'none')) {{
                    if (toggleBtn) {{
                        toggleBtn.click();
                    }}
                }}
            }} else {{
                // Check if in full text view
                const fullTextView = highlight.closest('.full-text');
                if (fullTextView && fullTextView.id && fullTextView.id.startsWith('full-')) {{
                    // Match is in full text view - ensure it's visible
                    const transcriptionId = fullTextView.id.replace('full-', '');
                    const toggleBtn = document.getElementById('toggle-' + transcriptionId);
                    
                    if (fullTextView && (fullTextView.style.display === 'none' || window.getComputedStyle(fullTextView).display === 'none')) {{
                        if (toggleBtn) {{
                            toggleBtn.click();
                        }}
                    }}
                }}
            }}
            
            // Wait for DOM updates and ensure element is visible, then scroll
            const attemptScroll = (attempt = 1, maxAttempts = 5) => {{
                // Re-find the highlight in case DOM was updated
                const currentHighlight = document.querySelector(`.search-highlight[data-match-index="${{index}}"]`);
                if (!currentHighlight) return;
                
                // Check if element is visible
                const rect = currentHighlight.getBoundingClientRect();
                const isVisible = rect.width > 0 && rect.height > 0 && 
                                 window.getComputedStyle(currentHighlight).display !== 'none' &&
                                 window.getComputedStyle(currentHighlight).visibility !== 'hidden';
                
                if (isVisible || attempt >= maxAttempts) {{
                    // Element is visible or we've tried enough times
                    currentHighlight.classList.add('active');
                    
                    // Scroll to the highlight
                    try {{
                        currentHighlight.scrollIntoView({{
                            behavior: 'smooth',
                            block: 'center',
                            inline: 'nearest'
                        }});
                    }} catch (e) {{
                        // Fallback: scroll parent container into view first
                        if (parentContainer) {{
                            parentContainer.scrollIntoView({{
                                behavior: 'smooth',
                                block: 'start'
                            }});
                            
                            // Then try scrolling to highlight again after a delay
                            setTimeout(() => {{
                                if (currentHighlight) {{
                                    currentHighlight.scrollIntoView({{
                                        behavior: 'smooth',
                                        block: 'center',
                                        inline: 'nearest'
                                    }});
                                }}
                            }}, 300);
                        }}
                    }}
                }} else {{
                    // Element not yet visible, try again after a short delay
                    setTimeout(() => attemptScroll(attempt + 1, maxAttempts), 100);
                }}
            }};
            
            // Start attempting to scroll after initial DOM update delay
            setTimeout(() => attemptScroll(), 100);
        }}
        
        function navigateMatch(direction) {{
            if (searchMatches.length === 0) return;
            
            if (direction === 'next') {{
                currentMatchIndex = (currentMatchIndex + 1) % searchMatches.length;
            }} else {{
                currentMatchIndex = (currentMatchIndex - 1 + searchMatches.length) % searchMatches.length;
            }}
            
            scrollToMatch(currentMatchIndex);
            updateSearchResultsInfo();
        }}
        
        function clearSearch() {{
            document.getElementById('searchInput').value = '';
            clearHighlights();
            searchMatches = [];
            currentMatchIndex = -1;
            searchQuery = '';
            updateSearchResultsInfo();
        }}
        
        // Keyboard shortcuts for search
        document.addEventListener('keydown', function(e) {{
            // Ctrl+F or Cmd+F to focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'f') {{
                e.preventDefault();
                document.getElementById('searchInput').focus();
                return;
            }}
            
            // Only handle search shortcuts if search input is not focused
            const searchInput = document.getElementById('searchInput');
            if (document.activeElement === searchInput) {{
                // Enter to find next, Shift+Enter for previous
                if (e.key === 'Enter') {{
                    e.preventDefault();
                    if (e.shiftKey) {{
                        navigateMatch('prev');
                    }} else {{
                        navigateMatch('next');
                    }}
                    return;
                }}
                // Escape to clear search
                if (e.key === 'Escape') {{
                    clearSearch();
                    searchInput.blur();
                    return;
                }}
            }} else {{
                // F3 for next match, Shift+F3 for previous
                if (e.key === 'F3') {{
                    e.preventDefault();
                    if (e.shiftKey) {{
                        navigateMatch('prev');
                    }} else {{
                        navigateMatch('next');
                    }}
                    return;
                }}
            }}
        }});
        
        // Initialize search on page load with pre-indexing
        document.addEventListener('DOMContentLoaded', function() {{
            // Pre-index ALL searchable text content (runs once on page load)
            const searchableElements = document.querySelectorAll('.segment-text, .full-text, .filename');
            const totalElements = searchableElements.length;
            let indexed = 0;
            
            function indexChunk() {{
                const chunkSize = 200; // Process 200 elements per frame
                const end = Math.min(indexed + chunkSize, totalElements);
                
                for (let i = indexed; i < end; i++) {{
                    const element = searchableElements[i];
                    if (!element.dataset.originalText) {{
                        element.dataset.originalText = element.textContent;
                        // Pre-compute lowercase version for faster case-insensitive search
                        element.dataset.lowerText = element.textContent.toLowerCase();
                    }}
                }}
                
                indexed = end;
                
                if (indexed < totalElements) {{
                    requestAnimationFrame(indexChunk);
                }} else {{
                    // Indexing complete, now initialize search
                    initializeSearch();
                }}
            }}
            
            indexChunk();
        }});
        
        function initializeSearch() {{
            const searchInput = document.getElementById('searchInput');
            if (searchInput) {{
                searchInput.addEventListener('input', debounce(performSearch, 300));
                searchInput.addEventListener('keyup', function(e) {{
                    if (e.key === 'Enter') {{
                        e.preventDefault();
                        navigateMatch('next');
                    }}
                }});
            }}
            
            document.getElementById('caseSensitive').addEventListener('change', performSearch);
            document.getElementById('wholeWord').addEventListener('change', performSearch);
        }}
    </script>
</head>
<body>
    <div class="search-container">
        <div class="search-wrapper">
            <div class="search-input-group">
                <span style="color: #6c757d; font-size: 16px;">🔍</span>
                <input type="text" id="searchInput" class="search-input" placeholder="Search transcriptions...">
                <button class="search-clear" onclick="clearSearch()" title="Clear search">×</button>
            </div>
            <div class="search-options">
                <label class="search-option">
                    <input type="checkbox" id="caseSensitive">
                    <span>Match case</span>
                </label>
                <label class="search-option">
                    <input type="checkbox" id="wholeWord">
                    <span>Whole words/phrases</span>
                </label>
            </div>
            <div class="search-navigation">
                <button class="search-nav-btn" id="prevMatch" onclick="navigateMatch('prev')" disabled title="Previous match (Shift+F3)">▲ Prev</button>
                <span class="search-results-info" id="searchResultsInfo"></span>
                <button class="search-nav-btn" id="nextMatch" onclick="navigateMatch('next')" disabled title="Next match (F3)">Next ▼</button>
            </div>
        </div>
    </div>
    <div class="container">
        <div class="header">
            <h1>Audio Transcription Report</h1>
            <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <div style="background: #e3f2fd; padding: 10px; border-radius: 4px; margin-top: 15px; font-size: 0.9em;">
                <strong>💡 Audio Playback Tips:</strong>
                <ul style="margin: 5px 0; padding-left: 20px;">
                    <li>Click any timestamp to jump to that moment in the audio</li>
                    <li>Use keyboard shortcuts: Space (play/pause), ← → (skip), ↑ ↓ (volume)</li>
                    <li>Click the speed button (1x) to cycle through playback speeds</li>
                </ul>
                <div style="background: #e8f5e8; border: 1px solid #c3e6cb; padding: 8px; border-radius: 4px; margin-top: 8px;">
                    <strong>🎵 Audio Enhancement:</strong> Problematic WAV files are automatically converted to web-compatible MP3 format for optimal playback.
                </div>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{len(transcriptions)}</div>
                <div class="stat-label">Total Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{success_count}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{failure_count}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{format_time(total_time)}</div>
                <div class="stat-label">Total Time</div>
            </div>
        </div>
"""
    
    # Generate Table of Contents with recordings and DV scores (only when DV analysis is provided)
    if transcriptions and dv_analysis:
        _progress("Creating table of contents...", 10)
        html_content += """
        <div class="toc-container" style="background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <h2 style="margin-top: 0; color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px;">
                📋 Recordings Index
            </h2>
            <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                <thead>
                    <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                        <th style="padding: 12px; text-align: left; font-weight: bold; width: 60px;">#</th>
                        <th style="padding: 12px; text-align: left; font-weight: bold;">Filename</th>
                        <th style="padding: 12px; text-align: center; font-weight: bold; width: 120px;">Status</th>
                        <th style="padding: 12px; text-align: center; font-weight: bold; width: 100px;">DV Score</th>
                        <th style="padding: 12px; text-align: center; font-weight: bold; width: 100px;">Matches</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Sort transcriptions by DV score if available (highest first)
        sorted_transcriptions = sorted(
            enumerate(transcriptions),
            key=lambda x: x[1].get('dv_score', -1),
            reverse=True
        )
        
        for rank, (idx, item) in enumerate(sorted_transcriptions, 1):
            filename = Path(item['file_path']).name
            status = "✅ Success" if item['success'] else "❌ Failed"
            status_color = "#28a745" if item['success'] else "#dc3545"
            
            dv_score = item.get('dv_score', 0)
            dv_match_count = item.get('dv_match_count', 0)
            
            # Determine row color based on DV score
            if dv_score >= 50:
                row_bg = "#fee"
                priority = "🔴 HIGH"
            elif dv_score >= 20:
                row_bg = "#fff3cd"
                priority = "🟡 MEDIUM"
            elif dv_score > 0:
                row_bg = "#e7f3ff"
                priority = "🔵 LOW"
            else:
                row_bg = "#ffffff"
                priority = "⚪ None"
            
            dv_score_display = f"{dv_score:.1f}" if dv_score > 0 else "—"
            match_count_display = str(dv_match_count) if dv_match_count > 0 else "—"
            
            html_content += f"""
                    <tr style="background-color: {row_bg}; border-bottom: 1px solid #dee2e6; cursor: pointer;"
                        onclick="document.getElementById('recording-{idx}').scrollIntoView({{behavior: 'smooth', block: 'start'}});">
                        <td style="padding: 10px; font-weight: bold;">{rank}</td>
                        <td style="padding: 10px;">
                            <a href="#recording-{idx}" style="color: #007bff; text-decoration: none; font-weight: 500;">
                                {escape_html(filename)}
                            </a>
                        </td>
                        <td style="padding: 10px; text-align: center; color: {status_color}; font-weight: bold;">{status}</td>
                        <td style="padding: 10px; text-align: center; font-weight: bold; color: {'#dc3545' if dv_score >= 50 else '#ffc107' if dv_score >= 20 else '#17a2b8' if dv_score > 0 else '#6c757d'};">
                            {dv_score_display}
                            {f' <span style="font-size: 0.8em; color: #666;">({priority})</span>' if dv_score > 0 else ''}
                        </td>
                        <td style="padding: 10px; text-align: center;">{match_count_display}</td>
                    </tr>
"""
        
        html_content += """
                </tbody>
            </table>
            <div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 4px; font-size: 0.9em; color: #666;">
                <strong>💡 Tip:</strong> Click any row to jump to that recording. Recordings are sorted by DV score (highest first).
            </div>
        </div>
"""
    
    total_items = len(transcriptions)
    _progress(f"Processing transcriptions (0/{total_items})...", 15)
    
    for idx, item in enumerate(transcriptions):
        # Update progress: 15% to 70% for processing transcriptions
        if total_items > 0:
            progress = 15 + int((idx / total_items) * 55)
            if idx % max(1, total_items // 10) == 0 or idx == total_items - 1:  # Update every 10% or on last item
                _progress(f"Processing transcriptions ({idx + 1}/{total_items})...", progress)
        
        status_class = "success" if item['success'] else "error"
        # Create both absolute and relative paths for audio files
        audio_file_uri = Path(item['file_path']).as_uri()
        audio_file_path = str(Path(item['file_path'])).replace('\\\\', '/')
        
        # Create relative path for the audio file (in audio subdirectory)
        # This allows the HTML report to be moved with audio files and still work
        rel_audio_file = f"audio/{Path(item['file_path']).name}"
        folder_path = str(Path(item['file_path']).parent).replace('\\\\', '/')
        
        # Detect file extension and set appropriate MIME type
        file_extension = Path(item['file_path']).suffix.lower()
        mime_type_map = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.m4a': 'audio/mp4',
            '.aac': 'audio/aac',
            '.ogg': 'audio/ogg',
            '.flac': 'audio/flac',
            '.wma': 'audio/x-ms-wma',
            '.mp4': 'audio/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska'
        }
        primary_mime_type = mime_type_map.get(file_extension, 'audio/mpeg')
        
        # Check if there's a web-compatible MP3 version of this WAV file
        web_mp3_path = None
        rel_web_mp3_path = None
        if file_extension == '.wav':
            wav_path = Path(item['file_path'])
            potential_mp3 = wav_path.parent / f"{wav_path.stem}_web.mp3"
            if potential_mp3.exists():
                web_mp3_path = potential_mp3.as_uri()
                rel_web_mp3_path = f"audio/{potential_mp3.name}"
                primary_mime_type = 'audio/mpeg'  # Use MP3 if available
        
        # Get DV analysis for this file
        filename = Path(item['file_path']).name
        dv_analysis_for_file = dv_analysis_map.get(filename)
        
        html_content += f"""
        <div class="transcription {status_class}" id="recording-{idx}">
            <div class="file-info">
                <div class="filename">{Path(item['file_path']).name}</div>
                <div class="audio-player">
                    <div class="audio-controls">
                        <button class="control-btn" onclick="playPauseAudio({idx})">⏯️ Play/Pause</button>
                        <button class="control-btn" onclick="skipTime({idx}, -10)">⏪ -10s</button>
                        <button class="control-btn" onclick="skipTime({idx}, 10)">⏩ +10s</button>
                        <button class="playback-speed" id="speed-{idx}" onclick="cyclePlaybackSpeed({idx})">1x</button>
                        <div class="time-display" id="time-{idx}">00:00 / 00:00</div>
                    </div>
                    <audio id="audio-{idx}" class="audio-element" controls preload="metadata" 
                           onloadedmetadata="initializeAudioPlayer({idx}, '{escape_js_string(rel_web_mp3_path if rel_web_mp3_path else rel_audio_file)}')">
                        <!-- Use only relative paths for portability -->
                        <source src="{escape_filename_for_html(rel_web_mp3_path if rel_web_mp3_path else rel_audio_file)}" type="{primary_mime_type}">
                        <source src="{escape_filename_for_html(rel_audio_file)}" type="{primary_mime_type}">
                        Your browser does not support the audio element.
                    </audio>
                    {f'<div style="background: #d4edda; border: 1px solid #c3e6cb; padding: 6px; border-radius: 4px; margin-top: 6px; font-size: 0.8em; color: #155724;"><strong>✅ Using web-compatible MP3 version</strong> (Original: {Path(item["file_path"]).name})</div>' if web_mp3_path else ''}
                </div>
                <div class="file-actions">
                    <a href="javascript:void(0)" onclick="openFileLocation('{escape_js_string(rel_audio_file)}')" class="folder-link" title="Open file location">📁 Open Location</a>
                </div>
                {f'<div class="duration">Duration: {format_time(item.get("duration", 0))}</div>' if item.get("duration") else ''}
            </div>
"""
        
        # Add DV score display if available
        dv_score = item.get('dv_score', 0)
        dv_match_count = item.get('dv_match_count', 0)
        if dv_score > 0:
            # Determine severity color
            if dv_score >= 50:
                score_color = "#dc3545"  # Red - high priority
                score_bg = "#fee"
                severity = "HIGH PRIORITY"
            elif dv_score >= 20:
                score_color = "#ffc107"  # Yellow - medium priority
                score_bg = "#fff3cd"
                severity = "MEDIUM PRIORITY"
            else:
                score_color = "#28a745"  # Green - low priority
                score_bg = "#d4edda"
                severity = "REVIEW RECOMMENDED"
            
            html_content += f"""
            <div class="dv-score-badge" style="background-color: {score_bg}; border: 2px solid {score_color}; padding: 10px; margin: 10px 0; border-radius: 6px; text-align: center;">
                <div style="font-size: 1.2em; font-weight: bold; color: {score_color}; margin-bottom: 5px;">
                    ⚠️ DV Review Score: <span style="font-size: 1.4em;">{dv_score:.1f}</span>
                </div>
                <div style="color: #555; font-size: 0.9em;">
                    Matches Found: {dv_match_count} | Priority: {severity}
                </div>
            </div>
"""
        
        if item['success']:
            segments = item.get('segments', [])
            if segments:
                # Show timestamped segments view
                html_content += f"""
            <div class="segment-toggle">
                <button id="toggle-{idx}" class="toggle-btn" onclick="toggleTranscriptionView({idx})">Show Timestamped Segments</button>
            </div>
            
            <div id="segments-{idx}" class="transcription-content" style="display: none;" aria-hidden="true" tabindex="-1">
"""
                for segment in segments:
                    start_time = format_time(segment.get('start', 0))
                    start_seconds = segment.get('start', 0)
                    segment_text = segment.get('text', '').strip()
                    # Highlight keywords in segment text (progress: 70-85% for highlighting)
                    # Only apply highlighting if word list detection is enabled (dv_analysis is not None)
                    if dv_analysis and idx == total_items - 1:  # Only update on last transcription
                        highlight_progress = 70 + int((len(segments) / max(1, sum(len(t.get('segments', [])) for t in transcriptions if t.get('success'))) * 15))
                        if progress_callback and len(segments) > 0:
                            _progress(f"Applying keyword highlighting...", min(85, highlight_progress))
                    if dv_analysis:
                        highlighted_segment = _highlight_keywords(segment_text, dv_analysis_for_file, category_weights)
                    else:
                        highlighted_segment = escape_html(segment_text)
                    html_content += f"""
                <div class="segment">
                    <div class="timestamp-col">
                        <div class="timestamp-text" data-seconds="{start_seconds}" onclick="seekToTime({idx}, {start_seconds})" title="Click to jump to this timestamp">{start_time}</div>
                    </div>
                    <div class="text-col">
                        <div class="segment-text">{highlighted_segment}</div>
                    </div>
                </div>
"""
                html_content += f"""
            </div>
            
            <div id="full-{idx}" class="full-text" aria-hidden="false">
{_highlight_keywords(item.get('transcription', ''), dv_analysis_for_file, category_weights) if dv_analysis else escape_html(item.get('transcription', ''))}
            </div>
"""
            else:
                # No segments available, show full text only
                if dv_analysis:
                    highlighted_full_text = _highlight_keywords(item.get('transcription', ''), dv_analysis_for_file, category_weights)
                else:
                    highlighted_full_text = escape_html(item.get('transcription', ''))
                html_content += f"""
            <div class="full-text">
{highlighted_full_text}
            </div>
"""
        else:
            html_content += f"""
            <div class="full-text">
                <div class="error-message">Error: {item.get("error", "Unknown error occurred")}</div>
            </div>
"""
        
        html_content += """
        </div>
"""
    
    # Note: Manual conversion suggestions removed since conversion is now automatic
    
    html_content += """
        <div class="timestamp">
            Report generated by Audio Transcriber App
        </div>
    </div>
</body>
</html>
"""
    
    # Save the HTML file
    _progress("Finalizing HTML report...", 90)
    report_path = os.path.join(output_dir, "transcription_report.html")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        _progress(f"✅ HTML report created successfully", 100)
        return report_path
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Error creating HTML report: {e}", 100)
        print(f"Error creating HTML report: {e}")
        return None


def get_file_duration(file_path: str) -> float:
    """Get the duration of an audio/video file in seconds."""
    try:
        import ffmpeg
        probe = ffmpeg.probe(file_path)
        duration = float(probe['streams'][0]['duration'])
        return duration
    except Exception:
        return 0.0 