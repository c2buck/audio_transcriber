import requests
import json
import time
import traceback
from typing import Dict, Any, Optional, Callable
from datetime import datetime


class OllamaClient:
    """Client for communicating with local Ollama instance."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Base URL for Ollama API (default: http://localhost:11434)
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"
        self._log_request_details = True  # Enable detailed request logging
        
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
        
    def test_connection(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Test connection to Ollama instance with comprehensive logging.
        
        Args:
            progress_callback: Optional callback for logging progress
            
        Returns:
            Dict with success status and detailed diagnostic information
        """
        self._log_info(f"Testing connection to Ollama at {self.base_url}...", progress_callback)
        
        try:
            start_time = time.time()
            self._log_debug(f"Making GET request to {self.api_url}/tags", progress_callback)
            
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            response_time = time.time() - start_time
            
            self._log_debug(f"Response received in {response_time:.3f}s - Status: {response.status_code}", progress_callback)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    models = data.get('models', [])
                    
                    self._log_info(f"✓ Successfully connected to Ollama", progress_callback)
                    self._log_debug(f"Response data: {json.dumps(data, indent=2)[:500]}...", progress_callback)
                    self._log_info(f"Found {len(models)} available models", progress_callback)
                    
                    model_names = []
                    for i, model in enumerate(models):
                        name = model.get('name', f'Unknown_{i}')
                        size = model.get('size', 0)
                        modified = model.get('modified_at', 'Unknown')
                        model_names.append(name)
                        self._log_debug(f"  Model {i+1}: {name} (Size: {size} bytes, Modified: {modified})", progress_callback)
                    
                    return {
                        'success': True,
                        'message': f"Connected to Ollama. Found {len(models)} models.",
                        'models': model_names,
                        'response_time': response_time,
                        'ollama_version': data.get('version', 'Unknown'),
                        'raw_response': data
                    }
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON response from Ollama: {str(e)}"
                    self._log_error(error_msg, progress_callback)
                    self._log_debug(f"Raw response: {response.text[:500]}", progress_callback)
                    return {
                        'success': False,
                        'message': error_msg,
                        'models': [],
                        'response_time': response_time
                    }
            else:
                error_msg = f"Ollama responded with HTTP {response.status_code}"
                self._log_error(error_msg, progress_callback)
                self._log_debug(f"Response headers: {dict(response.headers)}", progress_callback)
                self._log_debug(f"Response body: {response.text[:500]}", progress_callback)
                return {
                    'success': False,
                    'message': error_msg,
                    'models': [],
                    'response_time': response_time,
                    'status_code': response.status_code,
                    'response_text': response.text
                }
                
        except requests.exceptions.ConnectionError as e:
            error_msg = "Cannot connect to Ollama. Is it running on localhost:11434?"
            self._log_error(f"{error_msg} - {str(e)}", progress_callback)
            self._log_debug(f"Connection error details: {traceback.format_exc()}", progress_callback)
            return {
                'success': False,
                'message': error_msg,
                'models': [],
                'error_type': 'ConnectionError',
                'error_details': str(e)
            }
        except requests.exceptions.Timeout as e:
            error_msg = "Connection timeout. Ollama may be starting up."
            self._log_error(f"{error_msg} - {str(e)}", progress_callback)
            return {
                'success': False,
                'message': error_msg,
                'models': [],
                'error_type': 'Timeout',
                'error_details': str(e)
            }
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self._log_error(error_msg, progress_callback)
            self._log_debug(f"Full traceback: {traceback.format_exc()}", progress_callback)
            return {
                'success': False,
                'message': error_msg,
                'models': [],
                'error_type': type(e).__name__,
                'error_details': str(e),
                'traceback': traceback.format_exc()
            }

    def get_available_models(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Get list of available models from Ollama with detailed logging.
        
        Args:
            progress_callback: Optional callback for logging progress
            
        Returns:
            Dict with success status and detailed model information
        """
        self._log_info("Retrieving available models from Ollama...", progress_callback)
        
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_url}/tags", timeout=10)
            response_time = time.time() - start_time
            
            self._log_debug(f"Models request completed in {response_time:.3f}s", progress_callback)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                model_names = []
                
                self._log_info(f"Processing {len(models)} available models...", progress_callback)
                
                for i, model in enumerate(models):
                    name = model.get('name', f'Unknown_{i}')
                    size = model.get('size', 0)
                    size_gb = size / (1024**3) if size > 0 else 0
                    
                    model_info = {
                        'name': name,
                        'size': f"{size_gb:.1f} GB" if size_gb > 0 else "Unknown size",
                        'size_bytes': size,
                        'modified_at': model.get('modified_at', 'Unknown'),
                        'digest': model.get('digest', 'Unknown')
                    }
                    model_names.append(model_info)
                    
                    self._log_debug(f"Model {i+1}: {name} ({model_info['size']})", progress_callback)

                self._log_info(f"✓ Successfully retrieved {len(model_names)} models", progress_callback)
                return {
                    'success': True,
                    'models': model_names,
                    'response_time': response_time,
                    'total_models': len(model_names)
                }
            else:
                error_msg = f"Failed to get models: HTTP {response.status_code}"
                self._log_error(error_msg, progress_callback)
                self._log_debug(f"Response: {response.text[:300]}", progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'models': [],
                    'status_code': response.status_code
                }
        except Exception as e:
            error_msg = f"Error retrieving models: {str(e)}"
            self._log_error(error_msg, progress_callback)
            self._log_debug(f"Full traceback: {traceback.format_exc()}", progress_callback)
            return {
                'success': False,
                'error': error_msg,
                'models': [],
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }

    def check_model_availability(self, model_name: str, progress_callback: Optional[Callable] = None) -> bool:
        """
        Check if a specific model is available with logging.
        
        Args:
            model_name: Name of the model to check
            progress_callback: Optional callback for logging progress

        Returns:
            True if model is available, False otherwise
        """
        self._log_debug(f"Checking availability of model: {model_name}", progress_callback)
        
        models_info = self.get_available_models(progress_callback)
        if models_info['success']:
            available_models = [m['name'] for m in models_info['models']]
            is_available = model_name in available_models
            
            if is_available:
                self._log_info(f"✓ Model '{model_name}' is available", progress_callback)
            else:
                self._log_info(f"✗ Model '{model_name}' not found in available models", progress_callback)
                self._log_debug(f"Available models: {', '.join(available_models)}", progress_callback)
                
            return is_available
        else:
            self._log_error(f"Could not check model availability: {models_info.get('error', 'Unknown error')}", progress_callback)
            return False

    def generate_response(self, model: str, prompt: str,
                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Generate response from Ollama model with comprehensive logging.

        Args:
            model: Model name (e.g., "mistral")
            prompt: Input prompt text
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with success status, response text, and detailed metrics
        """
        self._log_info(f"Starting AI analysis with model: {model}", progress_callback)
        self._log_debug(f"Prompt length: {len(prompt)} characters", progress_callback)
        
        if self._log_request_details and progress_callback:
            # Log first and last 100 chars of prompt for debugging
            prompt_preview = prompt[:100] + "..." + prompt[-100:] if len(prompt) > 200 else prompt
            self._log_debug(f"Prompt preview: {prompt_preview}", progress_callback)

        try:
            # Check if model is available
            self._log_debug(f"Verifying model '{model}' availability...", progress_callback)
            if not self.check_model_availability(model, progress_callback):
                self._log_info(f"Model '{model}' not found. Attempting to pull...", progress_callback)

                # Try to pull the model if it's not available
                pull_result = self._pull_model(model, progress_callback)
                if not pull_result['success']:
                    error_msg = f"Model '{model}' not available and failed to pull: {pull_result['error']}"
                    self._log_error(error_msg, progress_callback)
                    return {
                        'success': False,
                        'error': error_msg,
                        'response': '',
                        'pull_attempted': True,
                        'pull_result': pull_result
                    }

            # Prepare the request payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Lower temperature for more consistent analysis
                    "top_p": 0.9,
                    "top_k": 40
                }
            }

            self._log_debug(f"Request payload prepared: {json.dumps({k: v if k != 'prompt' else f'<prompt:{len(v)} chars>' for k, v in payload.items()})}", progress_callback)
            self._log_info("Sending request to AI model...", progress_callback)

            # Make the request
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.api_url}/generate",
                    json=payload,
                    timeout=120  # 2 minute timeout for AI processing
                )
                
                processing_time = time.time() - start_time
                self._log_debug(f"Request completed in {processing_time:.1f} seconds", progress_callback)

                if response.status_code == 200:
                    try:
                        data = response.json()
                        ai_response = data.get('response', '').strip()
                        
                        # Log response metrics
                        response_length = len(ai_response)
                        words_in_response = len(ai_response.split()) if ai_response else 0
                        
                        self._log_info(f"✓ AI analysis completed successfully", progress_callback)
                        self._log_debug(f"Response length: {response_length} characters, {words_in_response} words", progress_callback)
                        self._log_debug(f"Processing time: {processing_time:.1f}s", progress_callback)
                        
                        # Log additional response metadata
                        if 'eval_count' in data:
                            self._log_debug(f"Tokens generated: {data.get('eval_count', 'Unknown')}", progress_callback)
                        if 'eval_duration' in data:
                            eval_duration = data.get('eval_duration', 0) / 1e9  # Convert nanoseconds to seconds
                            self._log_debug(f"Evaluation duration: {eval_duration:.2f}s", progress_callback)
                        
                        # Log first 200 chars of response for debugging
                        if ai_response and self._log_request_details:
                            response_preview = ai_response[:200] + "..." if len(ai_response) > 200 else ai_response
                            self._log_debug(f"Response preview: {response_preview}", progress_callback)

                        return {
                            'success': True,
                            'response': ai_response,
                            'processing_time': processing_time,
                            'model_used': model,
                            'response_length': response_length,
                            'response_words': words_in_response,
                            'eval_count': data.get('eval_count'),
                            'eval_duration': data.get('eval_duration'),
                            'raw_response': data
                        }
                        
                    except json.JSONDecodeError as e:
                        error_msg = f"Invalid JSON in AI response: {str(e)}"
                        self._log_error(error_msg, progress_callback)
                        self._log_debug(f"Raw response: {response.text[:500]}", progress_callback)
                        return {
                            'success': False,
                            'error': error_msg,
                            'response': '',
                            'processing_time': processing_time,
                            'raw_response_text': response.text
                        }
                else:
                    error_msg = f"AI request failed: HTTP {response.status_code}"
                    self._log_error(error_msg, progress_callback)
                    self._log_debug(f"Response headers: {dict(response.headers)}", progress_callback)
                    self._log_debug(f"Response body: {response.text[:500]}", progress_callback)

                    return {
                        'success': False,
                        'error': error_msg,
                        'response': '',
                        'processing_time': processing_time,
                        'status_code': response.status_code,
                        'response_text': response.text
                    }
                    
            except requests.exceptions.Timeout:
                error_msg = f"Request timeout after {processing_time:.1f}s - AI model took too long to respond"
                self._log_error(error_msg, progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'response': '',
                    'processing_time': processing_time,
                    'error_type': 'Timeout'
                }
            except requests.exceptions.ConnectionError as e:
                error_msg = "Cannot connect to Ollama. Is it running?"
                self._log_error(f"{error_msg} - {str(e)}", progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'response': '',
                    'error_type': 'ConnectionError',
                    'error_details': str(e)
                }

        except Exception as e:
            error_msg = f"Unexpected error during AI generation: {str(e)}"
            self._log_error(error_msg, progress_callback)
            self._log_debug(f"Full traceback: {traceback.format_exc()}", progress_callback)
            return {
                'success': False,
                'error': error_msg,
                'response': '',
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }

    def _pull_model(self, model_name: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Pull a model from Ollama registry with detailed logging.

        Args:
            model_name: Name of the model to pull
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with success status and detailed pull information
        """
        self._log_info(f"Pulling model '{model_name}' from Ollama registry...", progress_callback)
        
        try:
            payload = {"name": model_name}
            self._log_debug(f"Pull request payload: {json.dumps(payload)}", progress_callback)
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/pull",
                json=payload,
                timeout=300  # 5 minute timeout for model pulling
            )
            pull_time = time.time() - start_time

            if response.status_code == 200:
                self._log_info(f"✓ Successfully pulled model '{model_name}' in {pull_time:.1f}s", progress_callback)
                return {
                    'success': True,
                    'pull_time': pull_time,
                    'model_name': model_name
                }
            else:
                error_msg = f"Failed to pull model: HTTP {response.status_code}"
                self._log_error(f"{error_msg} - {response.text[:300]}", progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'status_code': response.status_code,
                    'response_text': response.text,
                    'pull_time': pull_time
                }

        except Exception as e:
            error_msg = f"Error pulling model: {str(e)}"
            self._log_error(error_msg, progress_callback)
            self._log_debug(f"Full traceback: {traceback.format_exc()}", progress_callback)
            return {
                'success': False,
                'error': error_msg,
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            } 