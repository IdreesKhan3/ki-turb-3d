"""
LLM Provider Abstraction Layer
Supports Ollama (qwen2.5-coder:32b , mistral:7b), Google Gemini, and DeepSeek API
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from .config import (
    OLLAMA_CHECK_TIMEOUT,
    OLLAMA_DEFAULT_TIMEOUT,
    GEMINI_DEFAULT_TIMEOUT,
    DEEPSEEK_DEFAULT_TIMEOUT,
)


class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """Generate streaming response from LLM (yields chunks)"""
        pass
    
    @abstractmethod
    def generate_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response using structured message history
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     Roles: 'system', 'user', 'assistant', 'tool'
            **kwargs: Additional parameters (temperature, etc.)
        
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def generate_stream_with_messages(self, messages: List[Dict[str, str]], **kwargs):
        """Yield response chunks from a structured message history."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class OllamaLLM(LLMProvider):
    """Local Ollama provider (Qwen/coder defaults)."""
    
    def __init__(self, model: Optional[str] = None, base_url: str = "http://localhost:11434"):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
        self.base_url = base_url
        # Qwen uses /api/chat; other models use /api/generate
        self.is_qwen = "qwen" in self.model.lower()
        if self.is_qwen:
            self.api_url = f"{base_url}/api/chat"
        else:
            self.api_url = f"{base_url}/api/generate"
    
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=OLLAMA_CHECK_TIMEOUT)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Call Ollama; Qwen uses num_ctx=32k and lower temperature."""
        request_timeout = kwargs.get("timeout", OLLAMA_DEFAULT_TIMEOUT)
        
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
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """Generate streaming response using Ollama"""
        request_timeout = kwargs.get("timeout", OLLAMA_DEFAULT_TIMEOUT)
        
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
        
        payload = {
            "model": self.model,
            "stream": True,
            "options": options
        }
        
        if kwargs.get("format") == "json":
            payload["format"] = "json"
        
        try:
            if self.is_qwen:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                payload["messages"] = messages
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=request_timeout,
                    stream=True
                )
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and isinstance(chunk["message"], dict):
                                content = chunk["message"].get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
            else:
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                payload["prompt"] = full_prompt
                
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=request_timeout,
                    stream=True
                )
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            content = chunk.get("response", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
                            
        except requests.exceptions.RequestException as e:
            if "Read timed out" in str(e):
                raise Exception(f"Ollama timed out ({request_timeout}s).")
            raise Exception(f"Ollama error: {e}")
    
    def generate_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using structured message history"""
        request_timeout = kwargs.get("timeout", OLLAMA_DEFAULT_TIMEOUT)
        
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
        
        payload = {
            "model": self.model,
            "stream": False,
            "options": options,
            "messages": messages  # Use messages directly
        }
        
        # Allow optional JSON format override
        if kwargs.get("format") == "json":
            payload["format"] = "json"
        
        try:
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
            
            return content
                
        except requests.exceptions.RequestException as e:
            if "Read timed out" in str(e):
                raise Exception(f"Ollama timed out ({request_timeout}s).")
            raise Exception(f"Ollama error: {e}")
    
    def generate_stream_with_messages(self, messages: List[Dict[str, str]], **kwargs):
        """Generate streaming response using structured message history"""
        request_timeout = kwargs.get("timeout", OLLAMA_DEFAULT_TIMEOUT)
        
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
        
        payload = {
            "model": self.model,
            "stream": True,
            "options": options,
            "messages": messages
        }
        
        # Allow optional JSON format override
        if kwargs.get("format") == "json":
            payload["format"] = "json"
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=request_timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and isinstance(chunk["message"], dict):
                            content = chunk["message"].get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.RequestException as e:
            if "Read timed out" in str(e):
                raise Exception(f"Ollama timed out ({request_timeout}s).")
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
                "gemini-2.5-flash",    
                "gemini-2.5-pro",      
                "gemini-2.0-flash",     
                "gemini-1.5-pro",       
                "gemini-1.5-flash"      
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
        """Generate response using Gemini (supports vision with images)"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Get images from kwargs (if provided)
        images = kwargs.get("images", None)
        
        # Configure JSON mode for Gemini if requested
        strict_json = kwargs.get("format") == "json"
        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
        }
        
        if strict_json:
            generation_config["response_mime_type"] = "application/json"
        
        # Add timeout (Gemini API doesn't have explicit timeout, but we can wrap it)
        request_timeout = kwargs.get("timeout", GEMINI_DEFAULT_TIMEOUT)
        
        try:
            # Use threading-based timeout instead of signal (works in all threads)
            import threading
            import queue
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def call_api():
                try:
                    # Build content: text + images (if any)
                    if images and isinstance(images, list) and len(images) > 0:
                        # Multi-modal: text + images (Gemini expects inline_data structure)
                        content_parts = [full_prompt]
                        for img in images:
                            if isinstance(img, dict) and "mime_type" in img and "data" in img:
                                content_parts.append({
                                    "inline_data": {
                                        "mime_type": img["mime_type"],
                                        "data": img["data"]
                                    }
                                })
                        response = self.client.generate_content(
                            content_parts,
                            generation_config=generation_config
                        )
                    else:
                        # Text-only
                        response = self.client.generate_content(
                            full_prompt,
                            generation_config=generation_config
                        )
                    result_queue.put(response)
                except Exception as e:
                    exception_queue.put(e)
            
            api_thread = threading.Thread(target=call_api, daemon=True)
            api_thread.start()
            api_thread.join(timeout=request_timeout)
            
            if api_thread.is_alive():
                raise TimeoutError(f"Gemini request timed out after {request_timeout}s")
            
            if not exception_queue.empty():
                raise exception_queue.get()
            
            if result_queue.empty():
                raise Exception("Gemini API call completed but no response received")
            
            response = result_queue.get()
            
            # response.text raises when no valid Part (finish_reason RECITATION/safety, etc.)
            content = None
            finish_reason = None
            try:
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                        part = candidate.content.parts[0]
                        content = part.text if hasattr(part, 'text') else str(part)
            except Exception:
                pass
            if content is None:
                try:
                    content = response.text
                except Exception:
                    pass
            if content is None:
                reason_hint = f" (finish_reason={finish_reason})" if finish_reason is not None else ""
                raise Exception(
                    f"Gemini returned empty or blocked response{reason_hint}. "
                    "Common causes: safety filter, recitation block, or max_tokens. Try rephrasing or shortening the request."
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
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """Generate streaming response using Gemini"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        images = kwargs.get("images", None)
        strict_json = kwargs.get("format") == "json"
        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
        }
        
        if strict_json:
            generation_config["response_mime_type"] = "application/json"
        
        try:
            # Build content
            if images and isinstance(images, list) and len(images) > 0:
                content_parts = [full_prompt]
                for img in images:
                    if isinstance(img, dict) and "mime_type" in img and "data" in img:
                        content_parts.append({
                            "inline_data": {
                                "mime_type": img["mime_type"],
                                "data": img["data"]
                            }
                        })
                response_stream = self.client.generate_content(
                    content_parts,
                    generation_config=generation_config,
                    stream=True
                )
            else:
                response_stream = self.client.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                    stream=True
                )
            
            for chunk in response_stream:
                text = None
                try:
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        candidate = chunk.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                            part = candidate.content.parts[0]
                            text = part.text if hasattr(part, 'text') else str(part)
                except Exception:
                    pass
                if text is None:
                    try:
                        text = chunk.text
                    except Exception:
                        pass
                if text:
                    yield text
                                
        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg or "api key" in error_msg.lower():
                raise Exception(f"Gemini API key error: {error_msg}.")
            elif "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                raise Exception(f"Gemini quota/rate limit exceeded: {error_msg}")
            else:
                raise Exception(f"Gemini error: {error_msg}")
    
    def generate_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response using structured message history
        Note: Gemini doesn't have native multi-turn with tool role,
        so we convert messages to a single prompt with clear structure
        """
        # Convert messages to Gemini format
        # System message becomes part of first user message
        # Tool results become part of conversation context
        formatted_parts = []
        
        system_msg = None
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_msg = content
            elif role == "tool":
                # Format tool results clearly
                tool_name = msg.get("name", "tool")
                formatted_parts.append(f"[Tool: {tool_name}]\n{content}\n")
            elif role == "user":
                if system_msg and not formatted_parts:
                    # Prepend system message to first user message
                    formatted_parts.append(f"{system_msg}\n\n{content}")
                    system_msg = None
                else:
                    formatted_parts.append(content)
            elif role == "assistant":
                formatted_parts.append(content)
        
        full_prompt = "\n\n".join(formatted_parts)
        
        # Use existing generate method
        return self.generate(full_prompt, system_prompt=None, **kwargs)
    
    def generate_stream_with_messages(self, messages: List[Dict[str, str]], **kwargs):
        """Generate streaming response using structured message history"""
        # Convert messages to Gemini format (same as generate_with_messages)
        formatted_parts = []
        
        system_msg = None
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_msg = content
            elif role == "tool":
                tool_name = msg.get("name", "tool")
                formatted_parts.append(f"[Tool: {tool_name}]\n{content}\n")
            elif role == "user":
                if system_msg and not formatted_parts:
                    formatted_parts.append(f"{system_msg}\n\n{content}")
                    system_msg = None
                else:
                    formatted_parts.append(content)
            elif role == "assistant":
                formatted_parts.append(content)
        
        full_prompt = "\n\n".join(formatted_parts)
        
        # Use existing generate_stream method
        for chunk in self.generate_stream(full_prompt, system_prompt=None, **kwargs):
            yield chunk



class DeepSeekLLM(LLMProvider):
    """DeepSeek API provider (OpenAI-compatible chat completions)."""

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")

    def is_available(self) -> bool:
        return self.api_key is not None

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        if messages:
            return messages
        built: List[Dict[str, str]] = []
        if system_prompt:
            built.append({"role": "system", "content": system_prompt})
        built.append({"role": "user", "content": prompt})
        return built

    def _request_payload(self, messages: List[Dict[str, str]], *, stream: bool, **kwargs) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "stream": stream,
        }
        if kwargs.get("format") == "json":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _extract_content(self, data: Dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise Exception(f"DeepSeek returned an unexpected response: {data}") from exc

    def _post(self, payload: Dict[str, Any], *, stream: bool, request_timeout: float):
        return requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=request_timeout,
            stream=stream,
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        request_timeout = kwargs.get("timeout", DEEPSEEK_DEFAULT_TIMEOUT)
        messages = self._build_messages(prompt, system_prompt=system_prompt)
        payload = self._request_payload(messages, stream=False, **kwargs)
        try:
            response = self._post(payload, stream=False, request_timeout=request_timeout)
            response.raise_for_status()
            return self._extract_content(response.json())
        except requests.exceptions.Timeout:
            raise Exception(f"DeepSeek timed out ({request_timeout}s).")
        except requests.exceptions.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise Exception(f"DeepSeek API error: {detail}")
        except Exception as exc:
            raise Exception(f"DeepSeek error: {exc}")

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        request_timeout = kwargs.get("timeout", DEEPSEEK_DEFAULT_TIMEOUT)
        messages = self._build_messages(prompt, system_prompt=system_prompt)
        payload = self._request_payload(messages, stream=True, **kwargs)
        try:
            response = self._post(payload, stream=True, request_timeout=request_timeout)
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                chunk = line[6:].strip()
                if chunk == "[DONE]":
                    break
                data = json.loads(chunk)
                delta = data.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    yield delta
        except requests.exceptions.Timeout:
            raise Exception(f"DeepSeek timed out ({request_timeout}s).")
        except Exception as exc:
            raise Exception(f"DeepSeek error: {exc}")

    def generate_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        request_timeout = kwargs.get("timeout", DEEPSEEK_DEFAULT_TIMEOUT)
        payload = self._request_payload(messages, stream=False, **kwargs)
        try:
            response = self._post(payload, stream=False, request_timeout=request_timeout)
            response.raise_for_status()
            return self._extract_content(response.json())
        except requests.exceptions.Timeout:
            raise Exception(f"DeepSeek timed out ({request_timeout}s).")
        except Exception as exc:
            raise Exception(f"DeepSeek error: {exc}")

    def generate_stream_with_messages(self, messages: List[Dict[str, str]], **kwargs):
        request_timeout = kwargs.get("timeout", DEEPSEEK_DEFAULT_TIMEOUT)
        payload = self._request_payload(messages, stream=True, **kwargs)
        try:
            response = self._post(payload, stream=True, request_timeout=request_timeout)
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                chunk = line[6:].strip()
                if chunk == "[DONE]":
                    break
                data = json.loads(chunk)
                delta = data.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    yield delta
        except requests.exceptions.Timeout:
            raise Exception(f"DeepSeek timed out ({request_timeout}s).")
        except Exception as exc:
            raise Exception(f"DeepSeek error: {exc}")



def get_llm_provider(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> LLMProvider:
    """
    Get LLM provider instance

    Args:
        provider_name: 'ollama', 'gemini', 'deepseek', or None (auto-detect, defaults to ollama)
        model_name: For Ollama: model tag (e.g. 'qwen2.5-coder:32b', 'mistral:7b').
                    For Gemini: model name. For DeepSeek: model id (e.g. 'deepseek-v4-pro').
                    Overrides env vars.

    Returns:
        LLMProvider instance
    """
    provider_name = provider_name or os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider_name == "ollama":
        model = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaLLM(model=model, base_url=base_url)
    
    elif provider_name == "gemini":
        # Default to gemini-2.5-flash 
        # Or use gemini-2.5-pro 
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        # model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

        return GeminiLLM(model=model)

    elif provider_name == "deepseek":
        model = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        return DeepSeekLLM(model=model, base_url=base_url)
    
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Supported: 'ollama', 'gemini', 'deepseek'")


def get_available_providers() -> Dict[str, bool]:
    """Return {provider_name: is_available} for ollama/gemini/deepseek."""
    providers = {}
    
    try:
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama = OllamaLLM(model=model, base_url=base_url)
        providers["ollama"] = ollama.is_available()
    except:
        providers["ollama"] = False
    
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            gemini = GeminiLLM(model=model)
            providers["gemini"] = gemini.is_available()
        else:
            providers["gemini"] = False
    except Exception:
        providers["gemini"] = False

    # Check DeepSeek
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
            deepseek = DeepSeekLLM(model=model)
            providers["deepseek"] = deepseek.is_available()
        else:
            providers["deepseek"] = False
    except Exception:
        providers["deepseek"] = False
    
    return providers

