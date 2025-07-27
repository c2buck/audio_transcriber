# AI Review Feature - Audio Transcriber App

The AI Review feature provides intelligent analysis of transcribed audio content using local AI models through Ollama integration. This document covers setup, usage, and technical details of the AI review functionality.

## Table of Contents

- [Overview](#overview)
- [Setup Requirements](#setup-requirements)
- [Core Capabilities](#core-capabilities)
- [Usage Workflow](#usage-workflow)
- [Enhanced Crime Investigation Features](#enhanced-crime-investigation-features)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)

## Overview

The AI review system analyzes transcribed audio content to identify relevant information, particularly for crime investigations. It processes transcripts segment-by-segment, applying contextual understanding to highlight important sections with precise timestamp references.

Key benefits:
- **Privacy-First**: All AI analysis runs locally on your machine
- **Contextual Analysis**: AI considers case-specific facts and requirements
- **Timestamp Navigation**: Direct links to specific moments in recordings
- **Comprehensive Reporting**: Generates detailed analysis reports

## Setup Requirements

1. **Install Ollama**
   - Windows: Download and install from [ollama.ai](https://ollama.ai)
   - macOS: `brew install ollama`
   - Linux: `curl -fsSL https://ollama.ai/install.sh | sh`

2. **Download AI Model**
   - Start Ollama service (usually automatic after installation)
   - Download a compatible model: `ollama pull llama2` (or other models)
   - Verify installation: `ollama list`

3. **System Requirements**
   - Minimum 8GB RAM (16GB+ recommended)
   - SSD storage recommended
   - CPU-only operation is supported, but GPU acceleration improves performance

## Core Capabilities

### Local Processing
- All AI analysis runs on your machine via Ollama
- No data sent to external services, ensuring complete privacy
- Supports various AI models (Llama, CodeLlama, Mistral, etc.)

### Contextual Analysis
- AI considers case facts and specific requirements
- Identifies content relevant to investigations
- Understands context across multiple recordings

### Segment-by-Segment Review
- Detailed analysis of individual transcript segments
- Maintains context between segments
- Processes each recording independently

### Flexible Models
- Compatible with various Ollama models
- Adapts to different language models and capabilities
- Allows selection based on performance needs

### Progress Tracking
- Real-time progress updates during analysis
- Detailed logging of AI interactions
- Comprehensive status reporting

## Usage Workflow

1. **Complete Transcription**
   - First transcribe your audio files using the main transcription feature
   - Ensure you have a combined transcript file

2. **Switch to AI Review Tab**
   - Navigate to the "🤖 AI Review" tab in the application

3. **Load Transcript**
   - Click "Browse" to select the combined transcript file
   - The app will parse segments for analysis
   - View segment information in the interface

4. **Configure Analysis**
   - Enter case facts or specific analysis requirements in the text area
   - Select your preferred AI model from the dropdown
   - Adjust any additional settings if available

5. **Start Review**
   - Click "Start AI Review" to begin analysis
   - Monitor progress as each segment is analyzed
   - View detailed logs in the logging area

6. **Review Results**
   - When complete, examine the analysis results
   - Open generated reports for comprehensive review
   - Use timestamp links to navigate to specific audio sections

## Enhanced Crime Investigation Features

The AI review feature has been specially enhanced for crime investigation scenarios:

### Crime-Relevant Section Detection
- **Intelligent Analysis**: AI specifically looks for content relevant to crime investigations including:
  - Direct evidence related to the crime
  - Witness statements and observations
  - Suspect behavior and statements
  - Timeline information
  - Physical evidence descriptions
  - Names, locations, and identifying details
  - Contradictions or inconsistencies
  - Suspicious activities

### Timestamp-Based Navigation
- **Precise Timing**: Each relevant section includes estimated timestamps
- **Direct Audio Links**: Click-able links that jump directly to specific moments in recordings
- **Time Ranges**: Clear start and end times for each relevant section

### Crime Investigation Report
- **Dedicated HTML Report**: Automatically generates a crime-focused investigation report
- **Audio Integration**: Direct links to play audio from specific timestamps
- **Organized by Recording**: Sections grouped by source recording for easy reference
- **Visual Highlighting**: Crime-relevant content clearly marked and formatted

## Output Files

When running AI analysis, three types of reports are generated:

1. **Individual Analysis Files** (`.ai.txt`): Detailed analysis for each recording
2. **Combined Summary** (`ai_review_summary_[timestamp].txt`): Complete analysis of all recordings
3. **Crime Investigation Report** (`crime_investigation_report_[timestamp].html`): Enhanced HTML report with:
   - Quick navigation to relevant audio sections
   - Timestamp-based audio links
   - Visual crime-relevant content highlighting
   - Executive summary of findings

## Troubleshooting

### Common Issues

**"Ollama connection failed"**
- Ensure Ollama is installed and running (`ollama serve`)
- Check that the Ollama service is accessible at `http://localhost:11434`
- Verify you have downloaded at least one AI model (`ollama list`)
- Check firewall settings if connection issues persist

**Slow Analysis Performance**
- Try using a smaller/faster AI model
- Ensure your system meets the minimum requirements
- Close other resource-intensive applications
- Consider GPU acceleration if available

**Out of Memory Errors**
- Try using a smaller AI model
- Reduce batch size or segment length
- Increase system swap space
- Upgrade RAM if possible

**Missing Timestamps in Reports**
- Ensure original transcripts contain timestamp information
- Use JSON-formatted transcripts for enhanced timestamp accuracy
- Check that audio files are properly linked

### Performance Tips

1. **Model Selection**: Choose smaller models for faster analysis, larger models for better accuracy
2. **GPU Acceleration**: If available, ensure Ollama is configured to use GPU
3. **Batch Processing**: Process manageable batches of recordings
4. **JSON Transcripts**: Use JSON-formatted transcripts for enhanced timestamp accuracy
5. **Detailed Case Facts**: Provide comprehensive case facts for better relevance detection

## Technical Details

### Architecture

The AI Review system consists of several key components:

- **AIReviewManager**: Core class managing the analysis process
- **TranscriptSegment**: Represents a single recording segment
- **CrimeRelevantSection**: Represents a section relevant to investigation
- **OllamaClient**: Handles communication with local Ollama models

### JSON Support

The system provides enhanced support for JSON-formatted transcripts:

- Automatic detection of JSON structure
- Improved timestamp accuracy
- Better segment boundary detection
- Enhanced metadata processing

### Prompt Engineering

The system uses carefully crafted prompts to guide the AI analysis:

- Context-aware instructions
- Case-specific guidance
- Structured response formats
- Timestamp extraction guidance

### Threading Model

Analysis runs in a separate thread to keep the UI responsive:

- Non-blocking operation
- Real-time progress updates
- Cancellation support
- Detailed logging

### Report Generation

The system generates comprehensive reports with:

- HTML formatting with responsive design
- Direct audio file links
- Timestamp navigation
- Content highlighting
- Executive summaries

---

For general application information, please refer to the main README.md file. 