import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any


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


def get_audio_files(directory: str) -> List[str]:
    """Get all supported audio files from a directory."""
    supported_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.mp4', '.avi', '.mov', '.mkv'}
    audio_files = []
    
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in supported_extensions:
                    audio_files.append(os.path.join(root, file))
    except Exception as e:
        print(f"Error scanning directory: {e}")
    
    return sorted(audio_files)


def create_html_report(transcriptions: List[Dict[str, Any]], output_dir: str, 
                      total_time: float, success_count: int, failure_count: int) -> str:
    """Create an HTML report with all transcriptions and hyperlinks to audio files."""
    
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
            background-color: #f5f5f5;
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
        }}
        .audio-link {{
            color: #007bff;
            text-decoration: none;
            font-size: 0.9em;
        }}
        .audio-link:hover {{
            text-decoration: underline;
        }}
        .transcription-text {{
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Audio Transcription Report</h1>
            <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
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
    
    for item in transcriptions:
        status_class = "success" if item['success'] else "error"
        audio_file_uri = Path(item['file_path']).as_uri()
        
        html_content += f"""
        <div class="transcription {status_class}">
            <div class="file-info">
                <div class="filename">{Path(item['file_path']).name}</div>
                <a href="{audio_file_uri}" class="audio-link">🎵 Play Audio File</a>
                {f'<div class="duration">Duration: {format_time(item.get("duration", 0))}</div>' if item.get("duration") else ''}
            </div>
            <div class="transcription-text">
"""
        
        if item['success']:
            html_content += item['transcription']
        else:
            html_content += f'<div class="error-message">Error: {item.get("error", "Unknown error occurred")}</div>'
        
        html_content += """
            </div>
        </div>
"""
    
    html_content += """
        <div class="timestamp">
            Report generated by Audio Transcriber App
        </div>
    </div>
</body>
</html>
"""
    
    # Save the HTML file
    report_path = os.path.join(output_dir, "transcription_report.html")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return report_path
    except Exception as e:
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