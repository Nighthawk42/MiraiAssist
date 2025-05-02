# ================================================
# FILE: modules/llm_manager.py
# ================================================

import logging
import os
import queue
import threading
import asyncio
import re
from typing import Optional, List, Dict, Any

# Import OpenAI library
try:
    from openai import (
        AsyncOpenAI, APIError, APIConnectionError, APITimeoutError,
        RateLimitError, InternalServerError, AuthenticationError, BadRequestError
    )
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Dummy classes for OpenAI API errors
    class AsyncOpenAI:
        pass
    class APIError(Exception):
        pass
    class APIConnectionError(APIError):
        pass
    class APITimeoutError(APIConnectionError):
        pass
    class RateLimitError(APIError):
        status_code = 429
        message = "Rate limit exceeded."
    class InternalServerError(APIError):
        status_code = 500
        message = "Internal server error."
    class AuthenticationError(APIError):
        status_code = 401
        message = "Authentication error."
    class BadRequestError(APIError):
        status_code = 400
        message = "Bad request."

# Import Tiktoken
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    class Tiktoken: pass # Dummy

from .config_manager import ConfigManager
from .context_manager import ContextManager

logger = logging.getLogger(__name__)

class LLMManagerError(Exception): pass

class LLMManager:
    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 1536
    DEFAULT_TIMEOUT = 120.0
    DEFAULT_RETRIES = 1

    def __init__(self, config: ConfigManager, gui_queue: queue.Queue):
        logger.info("Initializing LLMManager...")
        if not OPENAI_AVAILABLE: raise LLMManagerError("Required library 'openai' not installed.")
        # Warn if tiktoken missing but don't block unless context window configured
        if not TIKTOKEN_AVAILABLE: logger.warning("Tiktoken library not found. Token length checking will be unavailable.")

        self.config = config
        self.gui_queue = gui_queue
        self.llm_config = config.get_llm_config()
        self.context_manager: Optional[ContextManager] = None

        self.api_base_url: Optional[str] = self.llm_config.get("api_base_url")
        self.api_key_env_var: Optional[str] = self.llm_config.get("api_key_env_var")
        self.model_name: Optional[str] = self.llm_config.get("model_name")
        self.system_prompt: str = self.llm_config.get("system_prompt", self.DEFAULT_SYSTEM_PROMPT)
        self.temperature: float = float(self.llm_config.get("temperature", self.DEFAULT_TEMPERATURE))
        self.max_tokens: int = int(self.llm_config.get("max_tokens", self.DEFAULT_MAX_TOKENS))
        self.timeout: float = float(self.llm_config.get("timeout_seconds", self.DEFAULT_TIMEOUT))
        self.max_retries: int = int(self.llm_config.get("max_retries", self.DEFAULT_RETRIES))

        # API Key handling... (same as before)
        self.api_key: Optional[str] = None
        if self.api_key_env_var and self.api_key_env_var.upper() != "NONE":
            self.api_key = os.environ.get(self.api_key_env_var)
            if not self.api_key: logger.warning(f"LLM API key env var '{self.api_key_env_var}' set but not found.")
        else: logger.info("No LLM API key environment variable specified.")

        if not self.api_base_url: raise LLMManagerError("LLM 'api_base_url' is missing.")
        if not self.model_name: raise LLMManagerError("LLM 'model_name' is missing.")

        # --- Tiktoken Initialization ---
        self.encoder = None
        self.model_context_window = int(self.llm_config.get("model_context_window", 0))

        if self.model_context_window <= 0:
            logger.warning("LLM 'model_context_window' not configured or invalid in config. Token length checking disabled.")
        elif not TIKTOKEN_AVAILABLE:
            logger.warning("LLM 'model_context_window' configured, but Tiktoken library not found. Token length checking disabled.")
            self.model_context_window = 0 # Disable checking if lib missing
        else:
            # Try getting encoder for the specific model name
            model_name_for_encoder = self.llm_config.get("model_name") # Use the model name from config
            try:
                self.encoder = tiktoken.encoding_for_model(model_name_for_encoder)
                logger.info(f"Initialized tiktoken encoder for model: {model_name_for_encoder}")
            except KeyError:
                logger.warning(f"Tiktoken encoder not found for model '{model_name_for_encoder}'. Falling back to 'cl100k_base'.")
                try:
                    self.encoder = tiktoken.get_encoding("cl100k_base") # Common base
                except Exception as enc_e:
                     logger.error(f"Failed to load fallback tiktoken encoder 'cl100k_base': {enc_e}")
                     self.encoder = None; self.model_context_window = 0 # Disable if fallback fails
            except Exception as e:
                logger.error(f"Failed to initialize tiktoken encoder: {e}", exc_info=True)
                self.encoder = None; self.model_context_window = 0 # Disable on other errors
        # --- END Tiktoken Initialization ---

        # Initialize OpenAI Client
        try:
            client_api_key = self.api_key if self.api_key else "DUMMY_KEY"
            self.client = AsyncOpenAI(
                base_url=self.api_base_url, api_key=client_api_key,
                timeout=self.timeout, max_retries=self.max_retries
            )
            logger.info(f"AsyncOpenAI client initialized. Base URL: {self.api_base_url}, Model: {self.model_name}")
        except Exception as e:
            logger.critical(f"Failed to initialize AsyncOpenAI client: {e}", exc_info=True)
            raise LLMManagerError(f"OpenAI client initialization failed: {e}") from e

        self._is_processing_lock = threading.Lock()
        self._is_processing = False
        logger.info("LLMManager initialized successfully.")

    def set_context_manager(self, context_manager: ContextManager) -> None:
        if not isinstance(context_manager, ContextManager):
            raise TypeError(f"context_manager must be instance of ContextManager, got {type(context_manager)}")
        self.context_manager = context_manager
        logger.info("RAG ContextManager linked to LLMManager.")

    def _get_current_system_prompt(self) -> Dict[str, str]:
        return {"role": "system", "content": self.system_prompt}

    def _filter_think_tags(self, text: str) -> str:
        if not text or "<think>" not in text: return text
        try:
             think_pattern = r"<think>.*?</think>"
             filtered_text = re.sub(think_pattern, "", text, flags=re.DOTALL).strip()
             if len(text) != len(filtered_text):
                 logger.debug("Filtered <think> blocks.")
                 if not filtered_text and text: logger.warning("LLM response was only <think> blocks.")
             return filtered_text
        except Exception as e:
             logger.error(f"Error filtering <think> tags: {e}", exc_info=True)
             return text

    def _estimate_prompt_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimates token count for messages using tiktoken, including overhead."""
        if not self.encoder: return 0 # Cannot estimate without encoder

        num_tokens = 0
        try:
            for message in messages:
                num_tokens += 4  # Approximation for message overhead (role, separators)
                for key, value in message.items():
                    if value:
                        # Ensure value is a string before encoding
                        value_str = str(value)
                        num_tokens += len(self.encoder.encode(value_str))
            num_tokens += 3  # Approximation for priming assistant response
            return num_tokens
        except Exception as e:
            logger.error(f"Error during token estimation: {e}", exc_info=True)
            return 999999 # Return large number to likely trigger truncation on error

    async def _stream_llm_response(self, user_input: str):
        if not self.client: logger.error("LLM client not initialized."); self._signal_error("LLM Client Not Ready"); return
        if not self.context_manager: logger.error("ContextManager not set."); self._signal_error("Context Manager Not Set"); return

        logger.info("Starting LLM RAG request.")
        self.gui_queue.put({"type": "status", "payload": "Retrieving Context..."})

        full_response_text = ""
        error_occurred = False; error_message = ""; status_code = None
        retrieved_context: List[Dict[str, str]] = []; recent_messages: List[Dict[str, str]] = []

        try:
            # 1. Retrieve Context
            try: retrieved_context = self.context_manager.retrieve_relevant_context(user_input)
            except Exception as rag_e: logger.error(f"Failed RAG retrieval: {rag_e}", exc_info=True); self.gui_queue.put({"type": "log", "payload": "Warn: History retrieval failed.", "tag": "warning"})

            # 2. Get Recent Messages
            if self.context_manager.n_include_recent > 0:
                 try: recent_messages = self.context_manager.get_recent_messages(self.context_manager.n_include_recent)
                 except Exception as recent_e: logger.error(f"Failed get recent msgs: {recent_e}", exc_info=True)

            # 3. Construct Initial Message List
            messages: List[Dict[str, Any]] = []
            messages.append(self._get_current_system_prompt())
            if retrieved_context:
                try: retrieved_context.sort(key=lambda x: x.get('metadata', {}).get('index', float('inf')))
                except Exception: pass
                context_str = "\n".join([f"- {msg['role']}: {msg['content']}" for msg in retrieved_context])
                messages.append({"role": "system", "content": f"## Relevant Context:\n{context_str}\n## End Context"})
                logger.debug(f"Added {len(retrieved_context)} retrieved messages.")
            if recent_messages:
                 messages.extend(recent_messages); logger.debug(f"Added {len(recent_messages)} recent messages.")
            messages.append({"role": "user", "content": user_input})

            # 4. Check and Truncate Prompt if Necessary
            if self.encoder and self.model_context_window > 0:
                # Reserve space for response and a small buffer
                buffer = 50 # Add a small buffer for safety
                max_prompt_tokens = self.model_context_window - self.max_tokens - buffer
                if max_prompt_tokens <= 0:
                     logger.warning("Configured model_context_window too small for max_tokens. Check config.")
                     max_prompt_tokens = self.model_context_window // 2 # Use half window as fallback limit

                estimated_tokens = self._estimate_prompt_tokens(messages)
                logger.debug(f"Est. prompt tokens: {estimated_tokens}. Max allowed: {max_prompt_tokens}")

                # Truncation Loop
                while estimated_tokens > max_prompt_tokens and len(messages) > 2: # Keep SysPrompt+UserQuery minimum
                    logger.warning(f"Prompt too long ({estimated_tokens} > {max_prompt_tokens}). Reducing...")
                    # Remove the oldest context message (index 1, after system prompt)
                    removed_message = messages.pop(1)
                    logger.debug(f"Removed oldest context: Role={removed_message.get('role')}, Content='{str(removed_message.get('content'))[:30]}...'")
                    estimated_tokens = self._estimate_prompt_tokens(messages) # Recalculate

                # Final check after truncation attempts
                if estimated_tokens > max_prompt_tokens:
                    logger.error(f"Cannot reduce prompt ({estimated_tokens}) below limit ({max_prompt_tokens}). Aborting request.")
                    self._signal_error("Prompt Too Long", f"Cannot fit prompt within model limit ({self.model_context_window}).")
                    return # Stop processing

            # 5. Make API Call
            logger.debug(f"Sending {len(messages)} final messages to LLM.")
            self.gui_queue.put({"type": "status", "payload": "Thinking..."})
            stream = await self.client.chat.completions.create(
                model=self.model_name, messages=messages,
                temperature=self.temperature, max_tokens=self.max_tokens, stream=True
            )

            # 6. Process Stream Safely
            logger.debug("LLM stream opened. Reading chunks...")
            async for chunk in stream:
                choice = None; delta_content = None; finish_reason = None
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.delta: delta_content = choice.delta.content
                    finish_reason = choice.finish_reason
                if delta_content:
                    full_response_text += delta_content
                    self.gui_queue.put({"type": "llm_chunk", "payload": {"delta": delta_content}})
                if finish_reason:
                     logger.info(f"LLM stream finished. Reason: {finish_reason}")
                     if finish_reason == "length":
                         logger.warning("LLM response may be truncated (max_tokens).")
                         self.gui_queue.put({"type": "log", "payload": "Assistant response might be cut short.", "tag": "warning"})
                     elif finish_reason not in ["stop", None]:
                         logger.warning(f"LLM stopped unexpectedly: {finish_reason}")

            if not full_response_text.strip() and not error_occurred:
                logger.warning("LLM returned empty response after stream.")

        # 7. Handle API Errors
        except AuthenticationError as e: error_occurred=True; status_code=getattr(e,'status_code',401); error_message=f"Auth error ({status_code}): Check API key/perms. {getattr(e,'message',str(e))}"; logger.error(error_message,exc_info=True)
        except BadRequestError as e: error_occurred=True; status_code=getattr(e,'status_code',400); error_message=f"Bad request ({status_code}): Prompt issue? {getattr(e,'message',str(e))}"; logger.error(error_message,exc_info=True)
        except APIConnectionError as e: error_occurred=True; error_message=f"Network error: {self.api_base_url}. {e}"; logger.error(error_message,exc_info=True)
        except APITimeoutError as e: error_occurred=True; error_message=f"LLM request timed out ({self.timeout}s). {e}"; logger.error(error_message)
        except RateLimitError as e: error_occurred=True; status_code=getattr(e,'status_code',429); error_message=f"Rate limit ({status_code}). {getattr(e,'message',str(e))}"; logger.error(error_message)
        except InternalServerError as e: error_occurred=True; status_code=getattr(e,'status_code',500); error_message=f"LLM server error ({status_code}): {getattr(e,'message',str(e))}"; logger.error(error_message,exc_info=True)
        except APIError as e: error_occurred=True; status_code=getattr(e,'status_code','N/A'); error_message=f"LLM API error ({status_code}): {getattr(e,'message',str(e))}"; logger.error(error_message,exc_info=True)
        except Exception as e: error_occurred=True; error_message=f"Unexpected LLM comm error: {type(e).__name__}: {e}"; logger.error(error_message,exc_info=True)

        # 8. Finalize
        finally:
            filtered_response_text = self._filter_think_tags(full_response_text)
            if not error_occurred and filtered_response_text.strip():
                try:
                    if self.context_manager:
                        self.context_manager.add_message("assistant", filtered_response_text)
                        logger.debug("Assistant response added/indexed.")
                    else: error_occurred=True; error_message="Internal error: Context manager lost."
                except Exception as e: logger.error(f"Could not save/index assist reply: {e}",exc_info=True); error_occurred=True; error_message=f"Failed to save context: {e}"

            final_payload = {"text": filtered_response_text if not error_occurred else None, "error": error_occurred, "error_message": error_message if error_occurred else None, "status_code": status_code if error_occurred else None}
            self.gui_queue.put({"type": "llm_result", "payload": final_payload})
            if error_occurred: self._signal_error(f"LLM Failed: {error_message.split('.')[0]}", log_message=error_message)
            logger.info("LLM RAG stream processing finished.")

    def run_llm_in_background(self, user_input: str):
        if not self._is_processing_lock.acquire(blocking=False):
            logger.warning("LLM busy. Request ignored."); self.gui_queue.put({"type": "log", "payload": "Assistant is busy.", "tag": "warning"}); return
        self._is_processing = True
        logger.info("Starting LLM background thread.")
        thread = threading.Thread(target=self._run_llm_thread_target, args=(user_input,), daemon=True, name="LLMStreamThread")
        thread.start()

    def _run_llm_thread_target(self, user_input: str):
        loop = None
        try:
            try: loop = asyncio.get_running_loop()
            except RuntimeError: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            loop.run_until_complete(self._stream_llm_response(user_input))
        except Exception as e: logger.error(f"Fatal error in LLM thread: {e}", exc_info=True); self._signal_error("LLM Task Failed", str(e))
        finally:
            self._is_processing = False
            self._is_processing_lock.release()
            logger.debug("LLM processing lock released.")

    def _signal_error(self, status_message: str, log_message: Optional[str] = None):
        self.gui_queue.put({"type": "status", "payload": f"ERROR: {status_message}"})
        log_msg = log_message if log_message else status_message
        self.gui_queue.put({"type": "log", "payload": log_msg, "tag": "error"})