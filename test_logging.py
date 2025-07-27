#!/usr/bin/env python3
"""
Test script to demonstrate comprehensive logging in AI Review system.
This script helps verify that the logging system is working properly for debugging crashes.
"""

import sys
import os
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_review import AIReviewManager


def simple_callback(message):
    """Simple callback function to capture log messages."""
    print(f"CALLBACK: {message}")


def test_ollama_connection():
    """Test Ollama connection with comprehensive logging."""
    print("=" * 80)
    print("TESTING OLLAMA CONNECTION")
    print("=" * 80)
    
    try:
        # Initialize AI Review Manager
        ai_manager = AIReviewManager()
        
        # Test connection
        connection_result = ai_manager.test_ollama_connection(simple_callback)
        
        print(f"\nConnection Result: {connection_result}")
        
        if connection_result['success']:
            print(f"✓ Connected successfully!")
            print(f"Available models: {len(connection_result.get('models', []))}")
        else:
            print(f"✗ Connection failed: {connection_result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"CRITICAL ERROR in test_ollama_connection: {e}")
        import traceback
        traceback.print_exc()


def test_transcript_loading():
    """Test transcript loading with comprehensive logging."""
    print("\n" + "=" * 80)
    print("TESTING TRANSCRIPT LOADING")
    print("=" * 80)
    
    try:
        # Initialize AI Review Manager
        ai_manager = AIReviewManager()
        
        # Test with a non-existent file to see error handling
        print("\n--- Testing with non-existent file ---")
        result = ai_manager.load_combined_transcript("nonexistent_file.txt", simple_callback)
        print(f"Load result: {result}")
        
        # Test with current directory (should fail gracefully)
        print("\n--- Testing with directory path ---")
        result = ai_manager.load_combined_transcript(".", simple_callback)
        print(f"Load result: {result}")
        
    except Exception as e:
        print(f"CRITICAL ERROR in test_transcript_loading: {e}")
        import traceback
        traceback.print_exc()


def test_ai_analysis():
    """Test AI analysis with comprehensive logging."""
    print("\n" + "=" * 80)
    print("TESTING AI ANALYSIS")
    print("=" * 80)
    
    try:
        # Initialize AI Review Manager
        ai_manager = AIReviewManager()
        
        # Create a dummy transcript segment
        from ai_review import TranscriptSegment
        
        dummy_segment = TranscriptSegment(
            filename="test_recording.mp3",
            content="This is a test transcript with some suspicious activity mentioned.",
            segment_index=0
        )
        
        case_facts = "Test case involving suspicious activity and criminal behavior."
        
        print("\n--- Testing AI analysis (may fail if Ollama not available) ---")
        result = ai_manager.analyze_segment(
            segment=dummy_segment,
            case_facts=case_facts,
            model_name="mistral",
            progress_callback=simple_callback
        )
        
        print(f"Analysis result success: {result.get('success', False)}")
        if not result.get('success'):
            print(f"Analysis error: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"CRITICAL ERROR in test_ai_analysis: {e}")
        import traceback
        traceback.print_exc()


def test_segmentation():
    """Test transcript segmentation with comprehensive logging."""
    print("\n" + "=" * 80)
    print("TESTING TRANSCRIPT SEGMENTATION")
    print("=" * 80)
    
    try:
        # Initialize AI Review Manager
        ai_manager = AIReviewManager()
        
        # Test with sample transcript content
        sample_transcript = """
==== recording_001.mp3 ====
This is the first recording with some content about a meeting.
The participants discussed various topics.

==== recording_002.mp3 ====
This is the second recording with different content.
Some suspicious activities were mentioned here.

==== recording_003.mp3 ====
Final recording with additional evidence and statements.
"""
        
        print("\n--- Testing transcript segmentation ---")
        segments = ai_manager.segment_transcript(sample_transcript, simple_callback)
        
        print(f"Segmentation result: {len(segments)} segments created")
        for i, segment in enumerate(segments):
            print(f"  Segment {i+1}: {segment.filename} ({segment.word_count} words)")
        
    except Exception as e:
        print(f"CRITICAL ERROR in test_segmentation: {e}")
        import traceback
        traceback.print_exc()


def test_error_handling():
    """Test error handling with comprehensive logging."""
    print("\n" + "=" * 80)
    print("TESTING ERROR HANDLING")
    print("=" * 80)
    
    try:
        # Initialize AI Review Manager
        ai_manager = AIReviewManager()
        
        # Test with None inputs to trigger error handling
        print("\n--- Testing with None segment ---")
        result = ai_manager.analyze_segment(None, "test case", "mistral", simple_callback)
        print(f"None segment result: {result.get('success', False)} - {result.get('error', 'No error')}")
        
        # Test with empty content
        print("\n--- Testing with empty content ---")
        segments = ai_manager.segment_transcript("", simple_callback)
        print(f"Empty content segmentation: {len(segments)} segments")
        
        # Test with invalid model
        print("\n--- Testing with invalid model ---")
        from ai_review import TranscriptSegment
        dummy_segment = TranscriptSegment("test.mp3", "test content", 0)
        result = ai_manager.analyze_segment(dummy_segment, "test case", "invalid_model_name", simple_callback)
        print(f"Invalid model result: {result.get('success', False)} - {result.get('error', 'No error')}")
        
    except Exception as e:
        print(f"CRITICAL ERROR in test_error_handling: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function."""
    print("AI REVIEW SYSTEM - COMPREHENSIVE LOGGING TEST")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This test will generate extensive logging output to help debug crashes.")
    print("\nThe logging will appear in both:")
    print("1. Terminal output (what you see here)")
    print("2. Callback messages (prefixed with 'CALLBACK:')")
    print("\nWatching for any crashes or unexpected behavior...")
    
    try:
        # Run all tests
        test_ollama_connection()
        test_transcript_loading()
        test_segmentation()
        test_error_handling()
        test_ai_analysis()  # Run this last as it may take longer or fail
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        print("If you see this message, the logging system is working properly.")
        print("Check the output above for any errors or unexpected behavior.")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR in main test: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main() 