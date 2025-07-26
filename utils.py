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
    """Create an HTML report with all transcriptions, enhanced hyperlinks, and timestamped segments."""
    
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
        function openFileLocation(filePath) {{
            // Try to open the file directly (may auto-play depending on system settings)
            window.open(filePath, '_blank');
            
            // For Windows, we can try to open in explorer
            if (navigator.platform.indexOf('Win') !== -1) {{
                // This will work in some browsers/contexts
                try {{
                    const explorerPath = 'file:///' + filePath.replace(/\//g, '\\\\');
                    window.open(explorerPath, '_blank');
                }} catch(e) {{
                    console.log('Could not open in explorer:', e);
                }}
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
                fullView.style.display = 'none';
                toggleBtn.textContent = 'Show Full Text';
            }} else {{
                // Show full text view
                segmentView.style.display = 'none';
                fullView.style.display = 'block';
                toggleBtn.textContent = 'Show Timestamped Segments';
            }}
        }}
    </script>
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
    
    for idx, item in enumerate(transcriptions):
        status_class = "success" if item['success'] else "error"
        audio_file_uri = Path(item['file_path']).as_uri()
        audio_file_path = str(Path(item['file_path'])).replace('\\', '/')
        folder_path = str(Path(item['file_path']).parent).replace('\\', '/')
        
        html_content += f"""
        <div class="transcription {status_class}">
            <div class="file-info">
                <div class="filename">{Path(item['file_path']).name}</div>
                <div class="file-actions">
                    <a href="{audio_file_uri}" class="audio-link" title="Play audio file">🎵 Play Audio</a>
                    <a href="javascript:void(0)" onclick="openFileLocation('{audio_file_path}')" class="folder-link" title="Open file location">📁 Open Location</a>
                </div>
                {f'<div class="duration">Duration: {format_time(item.get("duration", 0))}</div>' if item.get("duration") else ''}
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
            
            <div id="segments-{idx}" class="transcription-content" style="display: none;">
"""
                for segment in segments:
                    start_time = format_time(segment.get('start', 0))
                    start_seconds = segment.get('start', 0)
                    segment_text = segment.get('text', '').strip()
                    html_content += f"""
                <div class="segment">
                    <div class="timestamp-col">
                        <div class="timestamp-text">{start_time}</div>
                    </div>
                    <div class="text-col">
                        <div class="segment-text">{segment_text}</div>
                    </div>
                </div>
"""
                html_content += f"""
            </div>
            
            <div id="full-{idx}" class="full-text">
{item['transcription']}
            </div>
"""
            else:
                # No segments available, show full text only
                html_content += f"""
            <div class="full-text">
{item['transcription']}
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