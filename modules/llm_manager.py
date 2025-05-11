# modules/llm_manager.py

import logging
import os
import queue
import threading
import asyncio
import re
import time
from typing import Optional, List, Dict, Any, Union

# --- Dependency Imports with Checks ---
try:
    from openai import (
        AsyncOpenAI, APIError, APIConnectionError, APITimeoutError,
        RateLimitError, InternalServerError, AuthenticationError, BadRequestError
    )
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Dummy classes for type hints if library not installed
    class AsyncOpenAI:
        def __init__(self, *args, **kwargs): pass
        class chat:
            class completions:
                @staticmethod
                async def create(*args, **kwargs):
                    if False: yield # Make it an async generator type
                    return
    class APIError(Exception): status_code: Optional[int] = None; message: str = "OpenAI API Error"
    class APIConnectionError(APIError): pass
    class APITimeoutError(APIConnectionError): pass
    class RateLimitError(APIError): status_code = 429; message = "Rate limit exceeded."
    class InternalServerError(APIError): status_code = 500; message = "Internal server error."
    class AuthenticationError(APIError): status_code = 401; message = "Authentication error."
    class BadRequestError(APIError): status_code = 400; message = "Bad request."

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    # from tiktoken import Encoding as TiktokenEncoding # More specific type
    TiktokenEncoding = Any
except ImportError:
    TIKTOKEN_AVAILABLE = False
    class TiktokenEncoding: pass
    class tiktoken:
        @staticmethod
        def encoding_for_model(model: str) -> Optional[TiktokenEncoding]: return None
        @staticmethod
        def get_encoding(encoding: str) -> Optional[TiktokenEncoding]: return None

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    class PreTrainedTokenizerBase: pass # Dummy for type hints
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs) -> Optional[PreTrainedTokenizerBase]: return None
    class AutoConfig:
         @staticmethod
         def from_pretrained(model_name: str, **kwargs) -> Any: return None


# Local Imports
from .config_manager import ConfigManager
from .memory_manager import MemoryManager # Changed from ContextManager

logger = logging.getLogger(__name__)

class LLMManagerError(Exception):
    """Custom exception for LLMManager operational errors."""
    pass

class LLMManager:
    """
    Manages asynchronous communication with an OpenAI-compatible LLM API,
    integrating context from MemoryManager and appropriate tokenization.
    """
    DEFAULT_SYSTEM_PROMPT = "You are Mirei, a helpful and concise AI assistant. Respond clearly and directly using markdown. Use provided context when relevant."
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 1536
    DEFAULT_TIMEOUT = 120.0
    DEFAULT_RETRIES = 1
    DEFAULT_CONTEXT_WINDOW = 0 # 0 means disable token checks / truncation
    DEFAULT_TOKENIZER_PREFERENCE = "auto" # "auto", "tiktoken", "transformers", "heuristic"
    DEFAULT_CHARS_PER_TOKEN = 4 # For heuristic estimation
    PROMPT_TRUNCATION_BUFFER = 100 # Tokens reserved (max_tokens for response + buffer)

    def __init__(self, config: ConfigManager, gui_queue: queue.Queue):
        logger.info("Initializing LLMManager (with MemoryManager & Tokenizer Logic)...")

        if not OPENAI_AVAILABLE:
            raise LLMManagerError("Required 'openai' library not installed. Run: uv add openai")

        self.config = config
        self.gui_queue = gui_queue
        self.llm_config = config.get_llm_config()
        self.memory_manager: Optional[MemoryManager] = None # Will be linked via set_memory_manager

        self._load_config_values()
        self._validate_config()

        self.api_key = self._load_api_key()
        self.tokenizer: Optional[Union[TiktokenEncoding, PreTrainedTokenizerBase]] = None
        self.tokenizer_type: Optional[str] = None
        self._initialize_tokenizer() # Now uses refined logic
        self._log_tokenizer_status()
        self._initialize_openai_client()

        self._is_processing_lock = threading.Lock()
        self._is_processing = False
        logger.info("LLMManager initialized successfully.")

    def _load_config_values(self) -> None:
        self.api_base_url = self.llm_config.get("api_base_url")
        self.api_key_env_var = self.llm_config.get("api_key_env_var")
        self.model_name = self.llm_config.get("model_name") # Used for API call
        # For tokenizer loading, we might use a different identifier if specified
        self.tokenizer_source_identifier = self.llm_config.get("tokenizer_source_for_estimation", self.model_name)

        self.system_prompt = self.llm_config.get("system_prompt", self.DEFAULT_SYSTEM_PROMPT)
        self.temperature = float(self.llm_config.get("temperature", self.DEFAULT_TEMPERATURE))
        self.max_tokens = int(self.llm_config.get("max_tokens", self.DEFAULT_MAX_TOKENS))
        self.timeout = float(self.llm_config.get("timeout_seconds", self.DEFAULT_TIMEOUT))
        self.max_retries = int(self.llm_config.get("max_retries", self.DEFAULT_RETRIES))
        self.model_context_window = int(self.llm_config.get("model_context_window", self.DEFAULT_CONTEXT_WINDOW))
        self.tokenizer_preference = self.llm_config.get("tokenizer_preference", self.DEFAULT_TOKENIZER_PREFERENCE).lower()
        self.chars_per_token_estimate = int(self.llm_config.get("chars_per_token_estimate", self.DEFAULT_CHARS_PER_TOKEN))
        if self.chars_per_token_estimate <= 0: self.chars_per_token_estimate = self.DEFAULT_CHARS_PER_TOKEN

    def _validate_config(self) -> None:
        if not self.api_base_url: raise LLMManagerError("LLM 'api_base_url' missing.")
        if not self.model_name: raise LLMManagerError("LLM 'model_name' missing.")
        if self.model_context_window > 0 and self.max_tokens >= self.model_context_window:
            logger.warning(
                f"Configured 'max_tokens' ({self.max_tokens}) is >= 'model_context_window' ({self.model_context_window}). "
                "This leaves no room for the prompt. LLM calls may fail. Adjust config."
            )

    def _load_api_key(self) -> Optional[str]:
        key, env_var = None, self.api_key_env_var
        if env_var and env_var.upper() != "NONE":
            key = os.environ.get(env_var)
            if not key: logger.warning(f"LLM API key env var '{env_var}' set but not found.")
            else: logger.debug("LLM API key loaded from environment.")
        else: logger.info("No LLM API key env var (or set to NONE).")
        return key

    def _initialize_tokenizer(self) -> None:
        if self.model_context_window <= 0:
            logger.warning("model_context_window <= 0. Token checking/specific tokenizer loading disabled. Using heuristic.")
            self.tokenizer_type = 'heuristic'; self.tokenizer = None; return

        pref = self.tokenizer_preference
        # Use tokenizer_source_identifier for loading the tokenizer
        identifier_for_tokenizer = self.tokenizer_source_identifier
        logger.info(f"Initializing tokenizer (Preference: '{pref}', Source for Tokenizer: '{identifier_for_tokenizer}')...")

        load_successful = False
        if pref == "tiktoken": load_successful = self._try_load_tiktoken(identifier_for_tokenizer)
        elif pref == "transformers": load_successful = self._try_load_transformers(identifier_for_tokenizer)
        elif pref == "auto": load_successful = self._try_auto_load_tokenizer(identifier_for_tokenizer)
        elif pref == "heuristic": self.tokenizer_type = 'heuristic'; load_successful = True
        else:
            logger.error(f"Invalid 'tokenizer_preference': '{pref}'. Defaulting to heuristic.")
            self.tokenizer_type = 'heuristic'; load_successful = True

        if not load_successful:
            logger.warning(f"Tokenizer init failed for '{identifier_for_tokenizer}' (pref: '{pref}'). Falling back to heuristic.")
            self.tokenizer_type = 'heuristic'; self.tokenizer = None

    def _try_auto_load_tokenizer(self, model_identifier: str) -> bool:
        logger.debug(f"Auto-detecting tokenizer for: '{model_identifier}'")
        model_lower = model_identifier.lower()
        if model_lower.startswith("gpt-") or "ada" in model_lower or "babbage" in model_lower or "curie" in model_lower or "davinci" in model_lower or "text-embedding-" in model_lower :
            logger.debug(f"Auto-detect: '{model_identifier}' suggests OpenAI model. Trying Tiktoken first.")
            if self._try_load_tiktoken(model_identifier): return True
            logger.debug(f"Tiktoken failed for '{model_identifier}'. Trying Transformers as broader attempt.")
            if self._try_load_transformers(model_identifier): return True
            return False
        logger.debug(f"Auto-detect: '{model_identifier}' not an explicit OpenAI pattern. Trying Transformers first.")
        if self._try_load_transformers(model_identifier): return True
        logger.debug(f"Transformers failed for '{model_identifier}'. Trying Tiktoken as general fallback.")
        if self._try_load_tiktoken(model_identifier): return True # Tiktoken tries cl100k_base
        logger.warning(f"Auto-detection failed for '{model_identifier}'. No suitable tokenizer found by auto logic."); return False

    def _try_load_tiktoken(self, model_identifier: str) -> bool:
        if not TIKTOKEN_AVAILABLE: logger.warning("Tiktoken library not available."); return False
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_identifier)
            self.tokenizer_type = 'tiktoken'
            logger.info(f"Successfully loaded Tiktoken for model: '{model_identifier}'")
            return True
        except KeyError:
            logger.debug(f"Tiktoken: No direct encoding for '{model_identifier}'. Trying 'cl100k_base'.")
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.tokenizer_type = 'tiktoken'
                logger.info("Successfully loaded Tiktoken with 'cl100k_base' fallback.")
                return True
            except Exception as e_fallback: logger.warning(f"Tiktoken fallback load failed: {e_fallback}"); return False
        except Exception as e: logger.error(f"Tiktoken init error for '{model_identifier}': {e}", exc_info=True); return False

    def _try_load_transformers(self, model_identifier: str) -> bool:
        if not TRANSFORMERS_AVAILABLE: logger.warning("Transformers library not available."); return False
        try:
            # AutoConfig.from_pretrained(model_identifier, trust_remote_code=True) # Optional pre-check
            self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code=True, use_fast=True)
            self.tokenizer_type = 'transformers'
            logger.info(f"Successfully loaded Transformers tokenizer for: '{model_identifier}'")
            return True
        except OSError as e:
            logger.warning(f"Transformers: Failed to load tokenizer for '{model_identifier}'. If local, ensure tokenizer files (tokenizer.model, etc.) are present or provide Hub ID. Error: {e}")
            return False
        except Exception as e:
            logger.error(f"Transformers: Unexpected error for '{model_identifier}': {e}", exc_info=True)
            return False

    def _log_tokenizer_status(self) -> None:
        if self.model_context_window <= 0:
             logger.warning("LLM 'model_context_window' <= 0. Token length checking disabled.")
             self.tokenizer_type = 'heuristic' # Ensure type reflects disabled checks
        elif self.tokenizer_type == 'heuristic':
             logger.warning(f"Using heuristic token counting (1 token ≈ {self.chars_per_token_estimate} chars).")
        elif self.tokenizer:
             logger.info(f"Initialized '{self.tokenizer_type}' tokenizer for '{self.tokenizer_source_identifier}'.")
        else: # Should be covered by heuristic fallback, but as a safeguard
             logger.error("Tokenizer initialization failed. Using heuristic token counting.")
             self.tokenizer_type = 'heuristic'

    def _initialize_openai_client(self) -> None:
        try:
            client_api_key = self.api_key if self.api_key else "placeholder_if_not_needed"
            self.client = AsyncOpenAI(
                base_url=self.api_base_url, api_key=client_api_key,
                timeout=self.timeout, max_retries=self.max_retries
            )
            logger.info(f"AsyncOpenAI client initialized. Target: {self.api_base_url}")
        except Exception as e:
            logger.critical(f"Failed to initialize AsyncOpenAI client: {e}", exc_info=True)
            raise LLMManagerError(f"OpenAI client initialization failed: {e}") from e

    def set_memory_manager(self, memory_manager: MemoryManager) -> None: # Changed type hint
        if not isinstance(memory_manager, MemoryManager):
            raise TypeError("Invalid MemoryManager provided to LLMManager.")
        self.memory_manager = memory_manager
        logger.info("MemoryManager linked successfully to LLMManager.")

    def _get_current_system_prompt(self) -> Dict[str, str]:
        return {"role": "system", "content": self.system_prompt}

    def _filter_think_tags(self, text: str) -> str:
        if not text or "<think>" not in text: return text
        try:
            filtered_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if len(text) != len(filtered_text): logger.debug("Filtered <think> blocks.")
            if not filtered_text and text: logger.warning("LLM response was only <think> blocks.")
            return filtered_text
        except Exception as e: logger.error(f"Error filtering <think> tags: {e}"); return text

    def _estimate_prompt_tokens(self, messages: List[Dict[str, Any]]) -> int:
        if self.tokenizer_type == 'heuristic' or self.tokenizer is None or self.model_context_window <= 0:
            char_count = sum(len(str(msg.get("content", ""))) for msg in messages)
            estimated = char_count // self.chars_per_token_estimate
            overhead = len(messages) * 4 # Rough overhead per message
            final_estimate = estimated + overhead
            logger.debug(f"Token estimation (heuristic): ~{final_estimate} tokens for {len(messages)} messages.")
            return final_estimate

        num_tokens = 0
        try:
            if self.tokenizer_type == 'tiktoken':
                for message in messages:
                    num_tokens += 4
                    for key, value in message.items():
                        if value: num_tokens += len(self.tokenizer.encode(str(value)))
                    if message.get("role") == "assistant": num_tokens += 1
                num_tokens += 3
            elif self.tokenizer_type == 'transformers' and isinstance(self.tokenizer, PreTrainedTokenizerBase):
                # Try to apply chat template for more accuracy if available, else sum parts.
                try:
                    # This is the ideal way IF the tokenizer has a well-defined chat template
                    # and the messages are in the format it expects.
                    # We might need to convert our messages list to what tokenizer.apply_chat_template expects
                    # For now, using a simpler sum as a robust estimation.
                    # chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    # num_tokens = len(self.tokenizer.encode(chat_prompt))

                    # Simpler sum-of-parts approach (fallback if apply_chat_template is tricky)
                    current_tokens = 0
                    for message in messages:
                        role_tokens = len(self.tokenizer.encode(str(message.get("role", "")), add_special_tokens=False))
                        content_tokens = len(self.tokenizer.encode(str(message.get("content", "")), add_special_tokens=False))
                        current_tokens += role_tokens + content_tokens + 4 # Rough overhead
                    if self.tokenizer.bos_token_id is not None: current_tokens +=1
                    if self.tokenizer.eos_token_id is not None: current_tokens +=1
                    num_tokens = current_tokens

                except Exception as template_e:
                    logger.warning(f"Failed to apply chat template for Transformers token estimation: {template_e}. Summing parts.")
                    current_tokens = 0
                    for message in messages:
                        role_tokens = len(self.tokenizer.encode(str(message.get("role", "")), add_special_tokens=False))
                        content_tokens = len(self.tokenizer.encode(str(message.get("content", "")), add_special_tokens=False))
                        current_tokens += role_tokens + content_tokens + 4
                    if self.tokenizer.bos_token_id is not None: current_tokens +=1
                    if self.tokenizer.eos_token_id is not None: current_tokens +=1
                    num_tokens = current_tokens
            else:
                logger.error(f"Unknown tokenizer type '{self.tokenizer_type}' during estimation. Using heuristic.")
                return self._fallback_to_heuristic_estimation(messages)

            logger.debug(f"Token estimation ({self.tokenizer_type}): {num_tokens} tokens for {len(messages)} messages.")
            return num_tokens
        except Exception as e:
            logger.error(f"{self.tokenizer_type} estimation error: {e}. Falling back to heuristic.", exc_info=True)
            return self._fallback_to_heuristic_estimation(messages)

    def _fallback_to_heuristic_estimation(self, messages: List[Dict[str, Any]]) -> int:
        """Utility to force heuristic estimation and log it."""
        original_tokenizer_type = self.tokenizer_type
        self.tokenizer_type = 'heuristic' # Force heuristic for this call
        self.tokenizer = None # Clear potentially problematic tokenizer for safety
        estimated = self._estimate_prompt_tokens(messages) # Recursive call hits heuristic branch
        # Don't restore tokenizer_type here; if we fell back, we stay heuristic until next re-init or successful load.
        logger.debug(f"Fell back to heuristic, estimated ~{estimated} tokens for {len(messages)} messages.")
        return estimated

    def _truncate_prompt(self, messages: List[Dict[str, Any]], max_prompt_tokens: int) -> List[Dict[str, Any]]:
        estimated_tokens = self._estimate_prompt_tokens(messages)
        logger.debug(f"Truncating prompt. Current estimated tokens: {estimated_tokens}, Target: <= {max_prompt_tokens}")

        if len(messages) <= 1: # Should at least have system or user
            if estimated_tokens > max_prompt_tokens:
                raise LLMManagerError(f"Cannot truncate: Single message prompt ({estimated_tokens} tokens) exceeds limit ({max_prompt_tokens}).")
            return messages

        # Identify system prompt (if any) and last user message to preserve them
        system_prompt_msg: Optional[Dict[str, Any]] = None
        last_user_msg_idx = -1

        if messages[0].get("role") == "system":
            system_prompt_msg = messages[0]
            core_messages_start_idx = 1
        else:
            core_messages_start_idx = 0
        
        # Find the last user message
        for i in range(len(messages) - 1, core_messages_start_idx -1, -1):
            if messages[i].get("role") == "user":
                last_user_msg_idx = i
                break
        
        if last_user_msg_idx == -1 and messages[-1].get("role") != "system": # No user message, but not just system
            last_user_msg_idx = len(messages) -1 # Treat the last message as immutable if no explicit user message
        
        # Messages that can be removed (between system prompt and last user message, or all but last if no system/user)
        mutable_history: List[Dict[str, Any]] = []
        final_fixed_messages: List[Dict[str, Any]] = []

        if system_prompt_msg:
            mutable_history = messages[core_messages_start_idx : last_user_msg_idx if last_user_msg_idx != -1 else len(messages)]
            final_fixed_messages.append(system_prompt_msg)
        else:
            mutable_history = messages[core_messages_start_idx : last_user_msg_idx if last_user_msg_idx != -1 else len(messages)]

        if last_user_msg_idx != -1 and last_user_msg_idx < len(messages): # Ensure last_user_msg_idx is valid
             # Add messages before last user message to mutable
             if last_user_msg_idx > core_messages_start_idx :
                  mutable_history = messages[core_messages_start_idx : last_user_msg_idx]
             else: # last_user_msg_idx is the first core message or doesn't exist
                  mutable_history = [] # No history to remove before last user message

             if last_user_msg_idx < len(messages): # If there IS a last user message (not just system)
                final_fixed_messages.append(messages[last_user_msg_idx])
        elif not final_fixed_messages: # No system and no identified last user, keep last message
             if messages:
                  mutable_history = messages[:-1]
                  final_fixed_messages.append(messages[-1])


        current_messages_for_truncation = (([system_prompt_msg] if system_prompt_msg else []) +
                                          mutable_history +
                                          ([messages[last_user_msg_idx]] if last_user_msg_idx != -1 and last_user_msg_idx < len(messages) else []))


        while self._estimate_prompt_tokens(current_messages_for_truncation) > max_prompt_tokens and mutable_history:
            removed = mutable_history.pop(0) # Remove oldest from the mutable part
            logger.debug(f"Truncating message: Role={removed.get('role')}, Content='{str(removed.get('content'))[:30]}...'")
            current_messages_for_truncation = (([system_prompt_msg] if system_prompt_msg else []) +
                                              mutable_history +
                                              ([messages[last_user_msg_idx]] if last_user_msg_idx != -1 and last_user_msg_idx < len(messages) else []))
            logger.debug(f" > New estimated tokens: {self._estimate_prompt_tokens(current_messages_for_truncation)}")

        final_prompt_construct = current_messages_for_truncation
        final_tokens = self._estimate_prompt_tokens(final_prompt_construct)

        if final_tokens > max_prompt_tokens:
            raise LLMManagerError(f"Prompt too long ({final_tokens} > {max_prompt_tokens}) after trying to truncate. Critical messages might be too large.")

        logger.info(f"Prompt truncated to approx {final_tokens} tokens.")
        return final_prompt_construct

    async def _stream_llm_response(self, user_input: str) -> None:
        if not self.client: self._signal_error("LLM Client Error"); return
        if not self.memory_manager: self._signal_error("MemoryManager Link Error"); return # Use MemoryManager

        logger.info("Starting LLM stream (using MemoryManager).")
        self.gui_queue.put({"type": "status", "payload": "Constructing Context..."})

        full_response_text = ""; error_occurred = False; error_message = ""; status_code = None
        final_messages_sent: List[Dict[str, Any]] = []

        try:
            context_messages = self.memory_manager.construct_prompt_context(user_input)
            messages = [self._get_current_system_prompt()] + context_messages + [{"role": "user", "content": user_input}]
            final_messages_sent = messages

            if self.model_context_window > 0:
                max_prompt_tokens = self.model_context_window - self.max_tokens - self.PROMPT_TRUNCATION_BUFFER
                if max_prompt_tokens <= 50: max_prompt_tokens = 50 # Min sensible limit
                estimated_tokens = self._estimate_prompt_tokens(messages)
                if estimated_tokens > max_prompt_tokens:
                    logger.warning(f"Prompt tokens ({estimated_tokens}) > limit ({max_prompt_tokens}). Truncating...")
                    final_messages_sent = self._truncate_prompt(messages, max_prompt_tokens)
            # ... (Rest of API call and streaming logic remains the same as your provided `_stream_llm_response`) ...
            # Ensure to use `final_messages_sent` for the API call.
            # When assistant response is received, add it using `self.memory_manager.add_message("assistant", ...)`.

            # (Beginning of existing API call block)
            logger.info(f"Sending {len(final_messages_sent)} messages to LLM API.")
            self.gui_queue.put({"type": "status", "payload": "Thinking..."})
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=final_messages_sent, # Use the potentially truncated list
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )

            logger.debug("LLM response stream opened.")
            async for chunk in stream:
                delta_content = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                finish_reason = chunk.choices[0].finish_reason if chunk.choices and chunk.choices[0].finish_reason else None
                if delta_content:
                    full_response_text += delta_content
                    self.gui_queue.put({"type": "llm_chunk", "payload": {"delta": delta_content}})
                if finish_reason:
                    logger.info(f"LLM stream ended. Finish reason: '{finish_reason}'")
                    if finish_reason == "length":
                        logger.warning("LLM response potentially truncated due to 'max_tokens' limit.")
                        self.gui_queue.put({"type": "log", "payload": "Assistant response may be incomplete (max tokens).", "tag": "warning"})
                    break 

            if not full_response_text.strip():
                logger.warning("LLM stream completed but yielded empty text content.")
            # (End of existing API call block - ensure error handling follows)

        except LLMManagerError as prep_err:
            error_occurred = True; error_message = str(prep_err); logger.error(error_message, exc_info=False)
        except AuthenticationError as e: error_occurred=True; status_code=getattr(e,'status_code',401); error_message=f"Auth Error ({status_code}): Check API key. {getattr(e,'message',str(e))}"
        except BadRequestError as e: error_occurred=True; status_code=getattr(e,'status_code',400); error_message=f"Bad Request ({status_code}): Invalid prompt/model? {getattr(e,'message',str(e))}"
        except APIConnectionError as e: error_occurred=True; error_message=f"Network Error connecting to {self.api_base_url}. {e}"
        except APITimeoutError as e: error_occurred=True; error_message=f"LLM request timed out ({self.timeout}s). {e}"
        except RateLimitError as e: error_occurred=True; status_code=getattr(e,'status_code',429); error_message=f"Rate Limit Error ({status_code}). {getattr(e,'message',str(e))}"
        except InternalServerError as e: error_occurred=True; status_code=getattr(e,'status_code',500); error_message=f"LLM Server Error ({status_code}): {getattr(e,'message',str(e))}"
        except APIError as e: error_occurred=True; status_code=getattr(e,'status_code','N/A'); error_message=f"Generic LLM API Error ({status_code}): {getattr(e,'message',str(e))}"
        except Exception as e: # Catch other prep or API errors
            error_occurred = True; error_message = f"Unexpected error during LLM processing: {e}"; logger.error(error_message, exc_info=True)

        filtered_response_text = self._filter_think_tags(full_response_text)

        if not error_occurred and filtered_response_text.strip():
            try:
                if self.memory_manager: # Use MemoryManager
                    self.memory_manager.add_message("assistant", filtered_response_text)
                    logger.debug("Assistant response added via MemoryManager.")
                else: logger.error("MemoryManager reference lost post-processing.")
            except Exception as ctx_e: logger.error(f"Failed to add assistant response via MemoryManager: {ctx_e}", exc_info=True)

        final_payload = {
            "text": filtered_response_text if not error_occurred else None,
            "error": error_occurred, "error_message": error_message if error_occurred else None,
            "status_code": status_code if error_occurred and status_code else (200 if not error_occurred else None)
        }
        self.gui_queue.put({"type": "llm_result", "payload": final_payload})

        if error_occurred: self._signal_error(f"LLM Failed ({status_code or 'N/A'})", error_message)
        logger.info("LLM stream processing method finished.")


    def run_llm_in_background(self, user_input: str) -> None:
        if not self._is_processing_lock.acquire(blocking=False):
            logger.warning("LLM busy. Request ignored.")
            self.gui_queue.put({"type": "log", "payload": "Assistant is busy.", "tag": "warning"})
            return
        self._is_processing = True
        logger.info(f"Starting LLM background thread for: '{user_input[:50]}...'")
        thread = threading.Thread(target=self._run_llm_thread_target, args=(user_input,), daemon=True, name="LLMStreamThread")
        thread.start()

    def _run_llm_thread_target(self, user_input: str) -> None:
        loop = None
        try:
            try: loop = asyncio.get_running_loop()
            except RuntimeError: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            loop.run_until_complete(self._stream_llm_response(user_input))
        except Exception as e:
            logger.error(f"Critical error in LLM background thread: {e}", exc_info=True)
            self._signal_error("LLM Task Failed Unexpectedly", f"Error: {e}")
        finally:
            self._is_processing = False
            self._is_processing_lock.release()
            logger.debug("LLM processing lock released.")

    def _signal_error(self, status_message: str, log_message: Optional[str] = None) -> None:
        self.gui_queue.put({"type": "status", "payload": f"ERROR: {status_message}"})
        self.gui_queue.put({"type": "log", "payload": log_message or status_message, "tag": "error"})