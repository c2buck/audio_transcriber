import requests
import json
import time
import traceback
import sys
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
        self.base_url = base_url.rstrip('/') if base_url else "http://localhost:11434"
        self.api_url = f"{self.base_url}/api"
        self._log_request_details = True  # Enable detailed request logging
        self.enable_terminal_logging = True  # Enable comprehensive terminal logging
        
        # Initialize terminal logging
        self._log_terminal_info("Ollama Client initialized")
        self._log_terminal_debug(f"Base URL: {self.base_url}")
        self._log_terminal_debug(f"API URL: {self.api_url}")
        
    def _log_terminal(self, level: str, message: str):
        """Log message directly to terminal with timestamp and level."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            formatted_msg = f"[{timestamp}] OLLAMA {level}: {message}"
            
            if level in ["ERROR", "CRITICAL"]:
                print(formatted_msg, file=sys.stderr, flush=True)
            else:
                print(formatted_msg, flush=True)
        except Exception as e:
            # Fallback logging if terminal logging fails
            try:
                print(f"[OLLAMA LOGGING ERROR] {e}: {message}", file=sys.stderr, flush=True)
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
        
    def test_connection(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Test connection to Ollama instance with comprehensive logging.
        
        Args:
            progress_callback: Optional callback for logging progress
            
        Returns:
            Dict with success status and detailed diagnostic information
        """
        self._log_info(f"=== TESTING OLLAMA CONNECTION ===", progress_callback)
        self._log_info(f"Testing connection to Ollama at {self.base_url}...", progress_callback)
        
        try:
            start_time = time.time()
            test_url = f"{self.api_url}/tags"
            self._log_debug(f"Making GET request to {test_url}", progress_callback)
            
            try:
                response = requests.get(test_url, timeout=5)
                response_time = time.time() - start_time
                
                self._log_debug(f"Response received in {response_time:.3f}s - Status: {response.status_code}", progress_callback)
                self._log_debug(f"Response headers: {dict(response.headers)}", progress_callback)
            except Exception as request_error:
                self._log_exception("Error during connection request", request_error, progress_callback)
                raise
            
            if response.status_code == 200:
                try:
                    self._log_debug("Parsing JSON response...", progress_callback)
                    data = response.json()
                    models = data.get('models', [])
                    
                    self._log_info(f"✓ Successfully connected to Ollama", progress_callback)
                    self._log_debug(f"Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}", progress_callback)
                    self._log_info(f"Found {len(models)} available models", progress_callback)
                    
                    model_names = []
                    for i, model in enumerate(models):
                        try:
                            name = model.get('name', f'Unknown_{i}') if isinstance(model, dict) else f'Invalid_Model_{i}'
                            size = model.get('size', 0) if isinstance(model, dict) else 0
                            modified = model.get('modified_at', 'Unknown') if isinstance(model, dict) else 'Unknown'
                            model_names.append(name)
                            self._log_debug(f"  Model {i+1}: {name} (Size: {size} bytes, Modified: {modified})", progress_callback)
                        except Exception as model_error:
                            self._log_exception(f"Error processing model {i}", model_error, progress_callback)
                            continue
                    
                    self._log_info(f"=== CONNECTION TEST SUCCESSFUL ===", progress_callback)
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
                    self._log_exception("JSON parsing error", e, progress_callback)
                    self._log_debug(f"Raw response: {response.text[:500]}", progress_callback)
                    return {
                        'success': False,
                        'message': error_msg,
                        'models': [],
                        'response_time': response_time
                    }
                except Exception as json_error:
                    self._log_exception("Error processing JSON response", json_error, progress_callback)
                    return {
                        'success': False,
                        'message': f"Response processing error: {str(json_error)}",
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
            self._log_exception(f"Connection error: {error_msg}", e, progress_callback)
            return {
                'success': False,
                'message': error_msg,
                'models': [],
                'error_type': 'ConnectionError',
                'error_details': str(e)
            }
        except requests.exceptions.Timeout as e:
            error_msg = "Connection timeout. Ollama may be starting up."
            self._log_exception(f"Timeout error: {error_msg}", e, progress_callback)
            return {
                'success': False,
                'message': error_msg,
                'models': [],
                'error_type': 'Timeout',
                'error_details': str(e)
            }
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self._log_exception("Unexpected error during connection test", e, progress_callback)
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
        self._log_info("=== RETRIEVING AVAILABLE MODELS ===", progress_callback)
        self._log_info("Retrieving available models from Ollama...", progress_callback)
        
        try:
            start_time = time.time()
            models_url = f"{self.api_url}/tags"
            self._log_debug(f"Making GET request to {models_url}", progress_callback)
            
            response = None
            try:
                response = requests.get(models_url, timeout=10)
                response_time = time.time() - start_time
                
                self._log_debug(f"Models request completed in {response_time:.3f}s", progress_callback)
                self._log_debug(f"Response status: {response.status_code}", progress_callback)
            except Exception as request_error:
                response_time = time.time() - start_time
                self._log_exception("Error during models request", request_error, progress_callback)
                return {
                    'success': False,
                    'error': f"Request failed: {str(request_error)}",
                    'models': [],
                    'response_time': response_time,
                    'error_type': type(request_error).__name__
                }
            
            if response and response.status_code == 200:
                try:
                    self._log_debug("Parsing models response JSON...", progress_callback)
                    data = response.json()
                    models = data.get('models', [])
                    model_names = []
                    
                    self._log_info(f"Processing {len(models)} available models...", progress_callback)
                    
                    for i, model in enumerate(models):
                        try:
                            if not isinstance(model, dict):
                                self._log_warning(f"Model {i} is not a dictionary: {type(model)}", progress_callback)
                                continue
                                
                            name = model.get('name', f'Unknown_{i}')
                            size = model.get('size', 0)
                            size_gb = size / (1024**3) if size > 0 else 0
                            
                            model_info = {
                                'name': name,
                                'size': f"{size_gb:.1f} GB" if size_gb > 0 else "Unknown size",
                                'size_bytes': size,
                                'modified_at': model.get('modified_at', 'Unknown'),
                                'digest': model.get('digest', 'Unknown')[:12] + "..." if model.get('digest') else 'Unknown'  # Truncate digest
                            }
                            model_names.append(model_info)
                            
                            self._log_debug(f"Model {i+1}: {name} ({model_info['size']})", progress_callback)
                        except Exception as model_error:
                            self._log_exception(f"Error processing model {i}", model_error, progress_callback)
                            continue

                    self._log_info(f"✓ Successfully retrieved {len(model_names)} models", progress_callback)
                    self._log_info("=== MODEL RETRIEVAL COMPLETED ===", progress_callback)
                    
                    return {
                        'success': True,
                        'models': model_names,
                        'response_time': response_time,
                        'total_models': len(model_names)
                    }
                except json.JSONDecodeError as e:
                    self._log_exception("JSON decode error in models response", e, progress_callback)
                    return {
                        'success': False,
                        'error': f"Invalid JSON response: {str(e)}",
                        'models': [],
                        'response_time': response_time
                    }
                except Exception as parse_error:
                    self._log_exception("Error parsing models response", parse_error, progress_callback)
                    return {
                        'success': False,
                        'error': f"Response parsing error: {str(parse_error)}",
                        'models': [],
                        'response_time': response_time
                    }
            elif response:
                error_msg = f"Failed to get models: HTTP {response.status_code}"
                self._log_error(error_msg, progress_callback)
                self._log_debug(f"Response: {response.text[:300]}", progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'models': [],
                    'status_code': response.status_code
                }
            else:
                error_msg = "No response received from Ollama"
                self._log_error(error_msg, progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'models': []
                }
        except Exception as e:
            self._log_exception("Critical error retrieving models", e, progress_callback)
            return {
                'success': False,
                'error': f"Error retrieving models: {str(e)}",
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
        try:
            if not model_name or not isinstance(model_name, str):
                self._log_error("Invalid model name provided for availability check", progress_callback)
                return False
                
            self._log_debug(f"Checking availability of model: {model_name}", progress_callback)
            
            models_info = self.get_available_models(progress_callback)
            if models_info and models_info.get('success'):
                try:
                    models_list = models_info.get('models', [])
                    self._log_debug(f"Retrieved {len(models_list)} models for availability check", progress_callback)
                    available_models = []
                    for m in models_list:
                        if isinstance(m, dict) and 'name' in m:
                            available_models.append(m.get('name', ''))
                        else:
                            self._log_warning(f"Invalid model format in availability check: {type(m)}", progress_callback)
                except Exception as processing_error:
                    self._log_exception("Error processing models list for availability check", processing_error, progress_callback)
                    return False
                is_available = model_name in available_models
                
                if is_available:
                    self._log_info(f"✓ Model '{model_name}' is available", progress_callback)
                else:
                    self._log_info(f"✗ Model '{model_name}' not found in available models", progress_callback)
                    self._log_debug(f"Available models: {', '.join(available_models[:5])}{'...' if len(available_models) > 5 else ''}", progress_callback)
                    
                return is_available
            else:
                error_msg = models_info.get('error', 'Unknown error') if models_info else 'No response from server'
                self._log_error(f"Could not check model availability: {error_msg}", progress_callback)
                return False
                
        except Exception as e:
            self._log_exception("Error checking model availability", e, progress_callback)
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
        self._log_info(f"=== STARTING AI GENERATION ===", progress_callback)
        generation_start_time = time.time()
        
        try:
            # Validate inputs
            self._log_debug("Validating inputs...", progress_callback)
            if not model or not isinstance(model, str):
                error_msg = "Invalid model name provided"
                self._log_error(error_msg, progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'response': ''
                }
                
            if not prompt or not isinstance(prompt, str):
                error_msg = "Invalid prompt provided"
                self._log_error(error_msg, progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'response': ''
                }
            
            self._log_info(f"Starting AI analysis with model: {model}", progress_callback)
            self._log_debug(f"Prompt length: {len(prompt)} characters", progress_callback)
            self._log_debug(f"Prompt word count: {len(prompt.split())} words", progress_callback)
            
            if self._log_request_details and progress_callback:
                # Log first and last 100 chars of prompt for debugging
                try:
                    if len(prompt) > 200:
                        prompt_preview = prompt[:100] + "..." + prompt[-100:]
                    else:
                        prompt_preview = prompt
                    self._log_debug(f"Prompt preview: {prompt_preview}", progress_callback)
                except Exception as preview_error:
                    self._log_exception("Error creating prompt preview", preview_error, progress_callback)

            # Check if model is available
            self._log_debug(f"Verifying model '{model}' availability...", progress_callback)
            try:
                if not self.check_model_availability(model, progress_callback):
                    self._log_info(f"Model '{model}' not found. Attempting to pull...", progress_callback)

                    # Try to pull the model if it's not available
                    pull_result = self._pull_model(model, progress_callback)
                    if not pull_result.get('success', False):
                        error_msg = f"Model '{model}' not available and failed to pull: {pull_result.get('error', 'Unknown error')}"
                        self._log_error(error_msg, progress_callback)
                        return {
                            'success': False,
                            'error': error_msg,
                            'response': '',
                            'pull_attempted': True,
                            'pull_result': pull_result
                        }
                else:
                    self._log_debug(f"Model '{model}' is available", progress_callback)
            except Exception as model_check_error:
                self._log_exception("Error checking model availability", model_check_error, progress_callback)
                # Continue anyway - let the generate request fail if model isn't available

            # Prepare the request payload
            self._log_debug("Preparing request payload...", progress_callback)
            try:
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

                # Safe payload logging
                try:
                    safe_payload = {k: v if k != 'prompt' else f'<prompt:{len(v)} chars>' for k, v in payload.items()}
                    self._log_debug(f"Request payload prepared: {json.dumps(safe_payload)}", progress_callback)
                except Exception as payload_log_error:
                    self._log_debug("Request payload prepared (logging failed)", progress_callback)
                    
            except Exception as payload_error:
                self._log_exception("Error preparing request payload", payload_error, progress_callback)
                return {
                    'success': False,
                    'error': f'Payload preparation failed: {str(payload_error)}',
                    'response': '',
                    'processing_time': time.time() - generation_start_time
                }
                
            self._log_info("Sending request to AI model...", progress_callback)

            # Make the request
            request_start_time = time.time()
            
            try:
                generate_url = f"{self.api_url}/generate"
                self._log_debug(f"Making POST request to {generate_url}", progress_callback)
                
                response = requests.post(
                    generate_url,
                    json=payload,
                    timeout=120  # 2 minute timeout for AI processing
                )
                
                processing_time = time.time() - request_start_time
                self._log_debug(f"Request completed in {processing_time:.1f} seconds", progress_callback)
                self._log_debug(f"Response status code: {response.status_code}", progress_callback)
                
            except requests.exceptions.Timeout:
                processing_time = time.time() - request_start_time
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
                self._log_exception(f"Connection error during generation: {error_msg}", e, progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'response': '',
                    'error_type': 'ConnectionError',
                    'error_details': str(e)
                }
            except Exception as request_error:
                self._log_exception("Error during AI request", request_error, progress_callback)
                return {
                    'success': False,
                    'error': f'Request failed: {str(request_error)}',
                    'response': '',
                    'processing_time': time.time() - request_start_time,
                    'error_type': type(request_error).__name__
                }

            if response.status_code == 200:
                try:
                    self._log_debug("Parsing AI response JSON...", progress_callback)
                    data = response.json()
                    ai_response = data.get('response', '').strip() if data.get('response') else ""
                    
                    # Log response metrics
                    response_length = len(ai_response)
                    words_in_response = len(ai_response.split()) if ai_response else 0
                    
                    self._log_info(f"✓ AI analysis completed successfully", progress_callback)
                    self._log_debug(f"Response length: {response_length} characters, {words_in_response} words", progress_callback)
                    self._log_debug(f"Processing time: {processing_time:.1f}s", progress_callback)
                    
                    # Log additional response metadata safely
                    try:
                        if 'eval_count' in data:
                            self._log_debug(f"Tokens generated: {data.get('eval_count', 'Unknown')}", progress_callback)
                        
                        if 'eval_duration' in data:
                            try:
                                eval_duration_raw = data.get('eval_duration', 0)
                                eval_duration = eval_duration_raw / 1e9 if eval_duration_raw and eval_duration_raw > 0 else 0
                                self._log_debug(f"Evaluation duration: {eval_duration:.2f}s", progress_callback)
                            except Exception as eval_error:
                                self._log_debug(f"Evaluation duration: Error parsing ({str(eval_error)})", progress_callback)
                        
                        if 'total_duration' in data:
                            try:
                                total_duration_raw = data.get('total_duration', 0)
                                total_duration = total_duration_raw / 1e9 if total_duration_raw and total_duration_raw > 0 else 0
                                self._log_debug(f"Total duration: {total_duration:.2f}s", progress_callback)
                            except Exception as total_error:
                                self._log_debug(f"Total duration: Error parsing ({str(total_error)})", progress_callback)
                        
                        if 'load_duration' in data:
                            try:
                                load_duration_raw = data.get('load_duration', 0)
                                load_duration = load_duration_raw / 1e9 if load_duration_raw and load_duration_raw > 0 else 0
                                self._log_debug(f"Model load duration: {load_duration:.2f}s", progress_callback)
                            except Exception as load_error:
                                self._log_debug(f"Model load duration: Error parsing ({str(load_error)})", progress_callback)
                    except Exception as metadata_error:
                        self._log_exception("Error logging response metadata", metadata_error, progress_callback)
                    
                    # Log first 200 chars of response for debugging
                    if ai_response and self._log_request_details:
                        try:
                            if len(ai_response) > 200:
                                response_preview = ai_response[:200] + "..."
                            else:
                                response_preview = ai_response
                            self._log_debug(f"Response preview: {response_preview}", progress_callback)
                        except Exception as preview_error:
                            self._log_exception("Error creating response preview", preview_error, progress_callback)

                    self._log_info(f"=== AI GENERATION COMPLETED SUCCESSFULLY ===", progress_callback)
                    
                    return {
                        'success': True,
                        'response': ai_response,
                        'processing_time': processing_time,
                        'model_used': model,
                        'response_length': response_length,
                        'response_words': words_in_response,
                        'eval_count': data.get('eval_count'),
                        'eval_duration': data.get('eval_duration'),
                        'total_duration': data.get('total_duration'),
                        'load_duration': data.get('load_duration'),
                        'raw_response': data
                    }
                    
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON in AI response: {str(e)}"
                    self._log_exception("JSON decode error in AI response", e, progress_callback)
                    try:
                        self._log_debug(f"Raw response: {response.text[:500]}", progress_callback)
                    except Exception:
                        pass
                    return {
                        'success': False,
                        'error': error_msg,
                        'response': '',
                        'processing_time': processing_time,
                        'raw_response_text': getattr(response, 'text', 'No response text available')
                    }
                except Exception as parse_error:
                    self._log_exception("Error parsing AI response", parse_error, progress_callback)
                    return {
                        'success': False,
                        'error': f'Response parsing failed: {str(parse_error)}',
                        'response': '',
                        'processing_time': processing_time,
                        'error_type': type(parse_error).__name__
                    }
            else:
                error_msg = f"AI request failed: HTTP {response.status_code}"
                self._log_error(error_msg, progress_callback)
                try:
                    self._log_debug(f"Response headers: {dict(response.headers)}", progress_callback)
                    self._log_debug(f"Response body: {response.text[:500]}", progress_callback)
                except Exception as log_error:
                    self._log_exception("Error logging response details", log_error, progress_callback)

                return {
                    'success': False,
                    'error': error_msg,
                    'response': '',
                    'processing_time': processing_time,
                    'status_code': response.status_code,
                    'response_text': getattr(response, 'text', 'No response text available')
                }

        except Exception as e:
            error_msg = f"Unexpected error during AI generation: {str(e)}"
            total_time = time.time() - generation_start_time
            self._log_exception("Critical error in generate_response", e, progress_callback)
            return {
                'success': False,
                'error': error_msg,
                'response': '',
                'processing_time': total_time,
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
        self._log_info(f"=== PULLING MODEL: {model_name} ===", progress_callback)
        
        try:
            if not model_name or not isinstance(model_name, str):
                error_msg = "Invalid model name for pulling"
                self._log_error(error_msg, progress_callback)
                return {
                    'success': False,
                    'error': error_msg
                }
                
            self._log_info(f"Pulling model '{model_name}' from Ollama registry...", progress_callback)
            
            payload = {"name": model_name}
            try:
                self._log_debug(f"Pull request payload: {json.dumps(payload)}", progress_callback)
            except Exception:
                self._log_debug("Pull request payload prepared (logging failed)", progress_callback)
            
            start_time = time.time()
            pull_url = f"{self.api_url}/pull"
            self._log_debug(f"Making POST request to {pull_url}", progress_callback)
            
            try:
                response = requests.post(
                    pull_url,
                    json=payload,
                    timeout=300  # 5 minute timeout for model pulling
                )
                pull_time = time.time() - start_time
                
                self._log_debug(f"Pull request completed in {pull_time:.1f}s", progress_callback)
                self._log_debug(f"Pull response status: {response.status_code}", progress_callback)
            except requests.exceptions.Timeout:
                pull_time = time.time() - start_time
                error_msg = f"Model pull timeout after {pull_time:.1f}s"
                self._log_error(error_msg, progress_callback)
                return {
                    'success': False,
                    'error': error_msg,
                    'pull_time': pull_time,
                    'error_type': 'Timeout'
                }
            except Exception as request_error:
                self._log_exception("Error during model pull request", request_error, progress_callback)
                pull_time = time.time() - start_time
                return {
                    'success': False,
                    'error': f"Pull request failed: {str(request_error)}",
                    'pull_time': pull_time,
                    'error_type': type(request_error).__name__
                }

            if response.status_code == 200:
                self._log_info(f"✓ Successfully pulled model '{model_name}' in {pull_time:.1f}s", progress_callback)
                self._log_info(f"=== MODEL PULL COMPLETED ===", progress_callback)
                return {
                    'success': True,
                    'pull_time': pull_time,
                    'model_name': model_name
                }
            else:
                error_msg = f"Failed to pull model: HTTP {response.status_code}"
                try:
                    self._log_error(f"{error_msg} - {response.text[:300]}", progress_callback)
                    response_text = response.text
                except Exception:
                    self._log_error(error_msg, progress_callback)
                    response_text = "Response text unavailable"
                    
                return {
                    'success': False,
                    'error': error_msg,
                    'status_code': response.status_code,
                    'response_text': response_text,
                    'pull_time': pull_time
                }

        except Exception as e:
            self._log_exception("Critical error pulling model", e, progress_callback)
            return {
                'success': False,
                'error': f"Error pulling model: {str(e)}",
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            } 