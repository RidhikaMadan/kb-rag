"""
LLM Provider abstraction to support both OpenAI and Local Llama models
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os
import sys

# Ensure default encoding is UTF-8
if sys.getdefaultencoding() != 'utf-8':
    # Force UTF-8 encoding for string operations
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate a response from the LLM"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please set it as an environment variable or pass it as a parameter.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
    
    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate response from OpenAI"""
        try:
            # Ensure all message content is UTF-8 safe
            clean_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, str):
                    content = content.encode('utf-8', errors='replace').decode('utf-8')
                clean_messages.append({"role": role, "content": content})
            
            # Wrap the API call to catch encoding errors at the source
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=clean_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            except UnicodeEncodeError as uee:
                # If UnicodeEncodeError occurs in the API call itself, wrap it
                raise ValueError("OpenAI API encoding error - please check your input for special characters")
            except Exception as api_err:
                # Re-raise to be handled by outer exception handler
                raise api_err
            
            result = response.choices[0].message.content
            if isinstance(result, str):
                result = result.encode('utf-8', errors='replace').decode('utf-8')
            return result if result else ""
        
        except UnicodeEncodeError:
            # If UnicodeEncodeError is raised, it means there's an encoding issue
            raise ValueError("OpenAI API encoding error - please check your input")
        except Exception as e:
            # Handle status code errors first (these are safe)
            if hasattr(e, 'status_code'):
                status_code = e.status_code
                if status_code == 401:
                    raise ValueError("OpenAI API authentication failed. Please check your API key is valid.")
                elif status_code == 429:
                    raise ValueError("OpenAI API rate limit exceeded. Please try again later.")
                elif status_code == 500:
                    raise ValueError("OpenAI API server error. Please try again later.")
                else:
                    raise ValueError("OpenAI API error (status code: {})".format(status_code))
            
            # For all other errors, try to get basic info without triggering UnicodeEncodeError
            error_type_name = "Exception"
            try:
                error_type_name = type(e).__name__
            except Exception:
                pass
            
            # Try to get a safe error message without calling str(e) which might fail
            # Check exception args first
            has_unicode_error = False
            if hasattr(e, 'args') and e.args:
                try:
                    # Check if first arg is a string that might cause issues
                    first_arg = e.args[0]
                    if isinstance(first_arg, str):
                        # Try to encode it - if it fails, we know there's a Unicode issue
                        try:
                            first_arg.encode('ascii', errors='strict')
                            # If we get here, it's ASCII-safe
                            error_msg_part = first_arg
                        except UnicodeEncodeError:
                            # Contains non-ASCII, encode with replace
                            error_msg_part = first_arg.encode('ascii', errors='replace').decode('ascii')
                            has_unicode_error = True
                    else:
                        error_msg_part = "Error occurred"
                except Exception:
                    error_msg_part = "Error occurred"
                    has_unicode_error = True
            else:
                error_msg_part = "Error occurred"
            
            # If we detected Unicode issues, use a generic message
            if has_unicode_error or isinstance(e, UnicodeEncodeError):
                raise ValueError("OpenAI API error occurred - encoding issue with special characters")
            
            # Otherwise, try to format a safe error message
            try:
                safe_type = error_type_name.encode('ascii', errors='replace').decode('ascii')
                safe_msg = error_msg_part.encode('ascii', errors='replace').decode('ascii')
                final_error = "OpenAI API error: {} | {}".format(safe_type, safe_msg)
                raise ValueError(final_error)
            except (UnicodeEncodeError, UnicodeDecodeError, Exception):
                # Ultimate fallback
                raise ValueError("OpenAI API error occurred")


class LocalLlamaProvider(LLMProvider):
    """Local Llama model provider"""
    
    def __init__(self, model_path: str = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf", n_ctx: int = 8192):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python package is required. Install with: pip install llama-cpp-python")
        
        import platform
        use_metal = False
        if platform.system() == "Darwin":
            try:
                from torch.backends import mps
                use_metal = mps.is_available()
            except ImportError:
                use_metal = False
        
        print(f"Loading local LLM model from {model_path}...")
        print(f"  - Using Metal GPU acceleration: {use_metal}")
        print(f"  - Context window: {n_ctx} tokens")
        print("  - This may take 30-60 seconds depending on your hardware...")
        
        self.model = Llama(
            model_path=model_path,
            verbose=False,
            metal=use_metal,
            n_ctx=n_ctx
        )
        
        print("✓ Local LLM model loaded successfully.")
    
    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate response from local Llama model"""
        full_prompt = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                content = content.encode('utf-8', errors='replace').decode('utf-8')
            
            if role == "system":
                full_prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                full_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                full_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        
        full_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        response = self.model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        
        result = response['choices'][0]['text'].strip()
        if isinstance(result, str):
            result = result.encode('utf-8', errors='replace').decode('utf-8')
        return result if result else ""


def get_llm_provider(use_local: bool = False, api_key: Optional[str] = None) -> LLMProvider:
    """Factory function to get the appropriate LLM provider"""
    if use_local:
        try:
            return LocalLlamaProvider()
        except Exception:
            return OpenAIProvider(api_key=api_key)
    else:
        return OpenAIProvider(api_key=api_key)
