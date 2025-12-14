"""
LLM Provider Abstraction Layer
Supports Ollama (default, free) and Google Gemini (free tier)
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass


class OllamaLLM(LLMProvider):
    """Ollama local LLM provider (Optimized for Qwen/Coder models)"""
    
    def __init__(self, model: Optional[str] = None, base_url: str = "http://localhost:11434"):
        # Use env default if model not provided, consistent with get_llm_provider()
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
        self.base_url = base_url
        # Qwen models need /api/chat, others can use /api/generate
        self.is_qwen = "qwen" in self.model.lower()
        if self.is_qwen:
            self.api_url = f"{base_url}/api/chat"
        else:
            self.api_url = f"{base_url}/api/generate"
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using Ollama with High Context & JSON Mode"""
        request_timeout = kwargs.get("timeout", 300)
        
        # Increased Context Window to 32k for Qwen (allows reading entire file trees)
        # Lower temperature for precise coding (0.1 for Qwen, 0.7 for others)
        if self.is_qwen:
            options = {
                "temperature": kwargs.get("temperature", 0.1),
                "num_ctx": 32768,
                "num_predict": 4096,
                "top_p": 0.9,
            }
        else:
            options = {
                "temperature": kwargs.get("temperature", 0.7),
                "num_ctx": 4096,
            }
        
        # Prepare payload with optional JSON enforcement
        payload = {
            "model": self.model,
            "stream": False,
            "options": options
        }
        
        # Force JSON mode if requested (Crucial for Qwen reliability)
        # JSON extraction handled by ActionParser, not here
        if kwargs.get("format") == "json":
            payload["format"] = "json"
        
        try:
            if self.is_qwen:
                # Qwen models work best with chat interface
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                payload["messages"] = messages
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=request_timeout
                )
                response.raise_for_status()
                result = response.json()
                
                if "message" in result and isinstance(result["message"], dict):
                    content = result["message"].get("content", "")
                else:
                    content = str(result)
                
                # Return raw content - JSON extraction handled by ActionParser
                return content
            else:
                # Other models using generate endpoint
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                
                payload["prompt"] = full_prompt
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=request_timeout
                )
                response.raise_for_status()
                content = response.json().get("response", "")
                
                # Return raw content - JSON extraction handled by ActionParser
                return content
                
        except requests.exceptions.RequestException as e:
            if "Read timed out" in str(e):
                raise Exception(f"Ollama timed out ({request_timeout}s). Try reducing file size or using Gemini.")
            raise Exception(f"Ollama error: {e}")


class GeminiLLM(LLMProvider):
    """Google Gemini API provider (free tier: 60 RPM, 1.5K/day)"""
    
    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # Simplified model selection: Try requested model first, then short fallback list
            # Do not auto-discover models (reduces latency and failure modes)
            requested_model = model.replace('models/', '') if model else None
            fallback_models = [
                "gemini-2.5-flash",      # Fast and capable (best for speed)
                "gemini-2.5-pro",       # Best quality (slower but more capable)
                "gemini-2.0-flash",     # Fast alternative
                "gemini-1.5-pro",        # Fallback
                "gemini-1.5-flash"      # Fallback
            ]
            
            # Try requested model first if provided
            models_to_try = []
            if requested_model:
                models_to_try.append(requested_model)
            models_to_try.extend([m for m in fallback_models if m not in models_to_try])
            
            # Try each model until one works
            self.client = None
            last_error = None
            for model_name in models_to_try:
                try:
                    self.client = genai.GenerativeModel(model_name)
                    self.model = model_name
                    break
                except Exception as e:
                    last_error = e
                    continue
            
            if self.client is None:
                raise Exception(
                    f"Failed to initialize Gemini model. "
                    f"Tried: {', '.join(models_to_try[:3])}. "
                    f"Last error: {last_error}. "
                    f"Please verify your API key is valid."
                )
                    
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")
    
    def is_available(self) -> bool:
        """Check if Gemini API is available"""
        return self.api_key is not None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using Gemini"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Configure JSON mode for Gemini if requested
        strict_json = kwargs.get("format") == "json"
        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
        }
        
        if strict_json:
            generation_config["response_mime_type"] = "application/json"
        
        # Add timeout (Gemini API doesn't have explicit timeout, but we can wrap it)
        request_timeout = kwargs.get("timeout", 120)  # Default 120s for Gemini
        
        try:
            # Use threading-based timeout instead of signal (works in all threads)
            import threading
            import queue
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def call_api():
                try:
                    response = self.client.generate_content(
                        full_prompt,
                        generation_config=generation_config
                    )
                    result_queue.put(response)
                except Exception as e:
                    exception_queue.put(e)
            
            # Start API call in a thread
            api_thread = threading.Thread(target=call_api, daemon=True)
            api_thread.start()
            api_thread.join(timeout=request_timeout)
            
            # Check if thread is still alive (timed out)
            if api_thread.is_alive():
                raise TimeoutError(f"Gemini request timed out after {request_timeout}s")
            
            # Check for exceptions
            if not exception_queue.empty():
                raise exception_queue.get()
            
            # Get result
            if result_queue.empty():
                raise Exception("Gemini API call completed but no response received")
            
            response = result_queue.get()
            
            # Handle Gemini API response - robust parsing
            content = None
            
            # Primary: Check text attribute
            if hasattr(response, 'text') and response.text:
                content = response.text
            # Secondary: Check candidates structure
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    if candidate.content.parts:
                        content = candidate.content.parts[0].text if hasattr(candidate.content.parts[0], 'text') else str(candidate.content.parts[0])
            
            # If no content found, raise clear error
            if content is None:
                raise Exception(
                    f"Gemini response parsing failed: No text content found. "
                    f"Response structure: {type(response).__name__}. "
                    f"Check API response format."
                )
            
            # Return raw content - JSON extraction handled by ActionParser
            return content
            
        except TimeoutError:
            raise Exception(f"Gemini request timed out after {request_timeout}s")
        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages
            if "API_KEY" in error_msg or "api key" in error_msg.lower():
                raise Exception(f"Gemini API key error: {error_msg}. Check GOOGLE_API_KEY environment variable.")
            elif "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                raise Exception(f"Gemini quota/rate limit exceeded: {error_msg}")
            else:
                raise Exception(f"Gemini error: {error_msg}")


def get_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """
    Get LLM provider instance
    
    Args:
        provider_name: 'ollama', 'gemini', or None (auto-detect, defaults to ollama)
    
    Returns:
        LLMProvider instance
    """
    provider_name = provider_name or os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider_name == "ollama":
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaLLM(model=model, base_url=base_url)
    
    elif provider_name == "gemini":
        # Default to gemini-2.5-flash for speed (best fast model for Pro accounts)
        # Or use gemini-2.5-pro for best quality (slower but more capable)
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return GeminiLLM(model=model)
    
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Supported: 'ollama', 'gemini'")


def get_available_providers() -> Dict[str, bool]:
    """Check which providers are available"""
    providers = {}
    
    # Check Ollama - use env model for consistency
    try:
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama = OllamaLLM(model=model, base_url=base_url)
        providers["ollama"] = ollama.is_available()
    except:
        providers["ollama"] = False
    
    # Check Gemini
    try:
        # Check if API key is set first
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            gemini = GeminiLLM(model=model)
            providers["gemini"] = gemini.is_available()
        else:
            providers["gemini"] = False
    except Exception:
        providers["gemini"] = False
    
    return providers

