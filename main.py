# C:\Users\Nighthawk\Desktop\MiraiAssist\main.py

from __future__ import annotations

import os
import sys
import signal
import queue
import threading
import tkinter # Keep for basic error dialogs
import traceback
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Use tkinter for basic error dialogs if GUI fails early
from tkinter import Tk, messagebox

# --- PATH & LOGGING SETUP ---
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
     sys.path.insert(0, str(project_root))

# --- Import customtkinter only if GUI is intended ---
try:
    import customtkinter
    CUSTOMTKINTER_AVAILABLE = True
except ImportError:
    CUSTOMTKINTER_AVAILABLE = False
    # We will check this later before initializing the UI part

# Basic logging setup FIRST (refined later by SystemManager)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)] # Log to stderr initially
)
logger = logging.getLogger(__name__) # Logger for main.py

# --- IMPORT CORE MANAGERS ---
try:
    from modules.config_manager import ConfigManager, ConfigError
    from modules.system_manager import SystemManager, LoggingSetupError, RequirementError
    from modules.audio_manager import AudioManager, AudioManagerError
    from modules.stt_manager import STTManager, STTManagerError
    from modules.context_manager import ContextManager, ContextManagerError
    from modules.memory_manager import MemoryManager, MemoryManagerError # IMPORT MemoryManager
    from modules.llm_manager import LLMManager, LLMManagerError
    from modules.tts_manager import TTSManager, TTSManagerError
    from modules.ui_manager import UIManager
except ImportError as exc:
    print(f"FATAL ERROR: Failed to import core modules: {exc}", file=sys.stderr)
    traceback.print_exc()
    try: root = Tk(); root.withdraw(); messagebox.showerror("Startup Error", f"Failed to import core modules:\n\n{exc}\n\nPlease check dependencies (e.g., run 'uv sync') and logs.")
    except Exception: pass
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: Unexpected error during initial imports: {e}", file=sys.stderr)
    traceback.print_exc()
    try: root = Tk(); root.withdraw(); messagebox.showerror("Startup Error", f"An unexpected error occurred during startup:\n\n{e}\n\nPlease check logs.")
    except Exception: pass
    sys.exit(1)

if not CUSTOMTKINTER_AVAILABLE:
    print("FATAL ERROR: CustomTkinter library is required but not found. Please install it (`uv add customtkinter`).", file=sys.stderr)
    sys.exit(1)


gui_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

class MiraiAppController:
    APP_NAME    = UIManager.APP_NAME
    APP_VERSION = UIManager.APP_VERSION

    def __init__(self) -> None:
        logger.info(f"Initializing {self.APP_NAME} Controller v{self.APP_VERSION}")

        self.cfg: Optional[ConfigManager] = None
        self.sysman: Optional[SystemManager] = None
        self.ctx: Optional[ContextManager] = None
        self.memman: Optional[MemoryManager] = None # ADDED MemoryManager attribute
        self.audio: Optional[AudioManager] = None
        self.stt: Optional[STTManager] = None
        self.llm: Optional[LLMManager] = None
        self.tts: Optional[TTSManager] = None
        self.ui: Optional[UIManager] = None

        self._backend_ready = False
        self._shutting_down = False

        try:
            self._initialize_backend()
            self._backend_ready = True
            logger.info("Backend initialization successful.")
        except (ConfigError, LoggingSetupError, RequirementError, ContextManagerError,
                  MemoryManagerError, # ADDED MemoryManagerError to exception list
                  AudioManagerError, STTManagerError, LLMManagerError, TTSManagerError) as e:
            logger.critical(f"CRITICAL Backend initialization failed: {type(e).__name__}: {e}", exc_info=True)
            self._show_startup_error_dialog(f"Backend Initialization Failed:\n\n{type(e).__name__}: {e}\n\nPlease check configuration and logs.")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"Unexpected critical error during backend init: {e}", exc_info=True)
            self._show_startup_error_dialog(f"Unexpected Critical Error during Startup:\n\n{e}\n\nPlease check logs.")
            sys.exit(1)

        try:
            self.ui = UIManager(gui_queue, self.cfg)
            self.ui.build_window(
                on_close_callback=self._handle_close_request,
                record_callback=self._handle_record_button_click,
                theme_change_callback=self._handle_theme_change,
                clear_history_callback=self._handle_clear_history,
                about_callback=self._handle_about,
                text_submit_callback=self._handle_text_input_submit
            )
            self.ui.update_status("Ready")
            self.ui.set_record_button_state("Record", enabled=True)
            self._setup_ptt_bindings_via_ui()
            self._populate_initial_ui_history()
            logger.info("UI initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize UI Manager: {e}", exc_info=True)
            self._stop_backend_managers()
            self._show_startup_error_dialog(f"FATAL ERROR: UI Initialization failed: {e}\n\nPlease check logs.")
            sys.exit(1)

        if self.ui and self.ui.root and self.ui.root.winfo_exists():
            self.ui.root.after(100, self._poll_gui_queue)
        else:
            logger.critical("UI Root window not available after initialization. Cannot poll queue.")
            self._handle_close_request()
            sys.exit(1)

    def _show_startup_error_dialog(self, message: str):
        try:
            root = Tk(); root.withdraw()
            messagebox.showerror(f"{self.APP_NAME} - Critical Startup Error", message)
        except Exception as tk_error:
            print(f"\nCRITICAL STARTUP ERROR (Tkinter Dialog Failed: {tk_error}):\n{message}\n", file=sys.stderr)

    def _initialize_backend(self) -> None:
        logger.info("Starting backend initialization sequence...")
        self.cfg = ConfigManager(); self.cfg.load(); logger.info("Configuration loaded.")

        self.sysman = SystemManager(self.cfg); self.sysman.setup_logging(); logger.info("Logging configured.")
        self.sysman.log_system_info(); self.sysman.verify_requirements(); logger.info("Requirement checks passed.")

        # Initialize ContextManager first (data layer)
        self.ctx = ContextManager(self.cfg); logger.info("Context manager (RAG data layer) initialized.")

        # Initialize MemoryManager (strategy layer, depends on ContextManager)
        if not self.ctx: # Should ideally be caught by earlier checks if ctx fails
            raise MemoryManagerError("ContextManager failed to initialize before MemoryManager.")
        self.memman = MemoryManager(self.cfg, self.ctx); logger.info("Memory manager (context strategy) initialized.")

        self.audio = AudioManager(self.cfg, gui_queue); logger.info("Audio manager initialized.")
        self.stt = STTManager(self.cfg); logger.info("STT manager initialized.")

        # Initialize LLMManager (depends on MemoryManager)
        self.llm = LLMManager(self.cfg, gui_queue)
        if not self.memman: # Should be caught if memman fails
            raise LLMManagerError("MemoryManager failed to initialize before LLMManager.")
        self.llm.set_memory_manager(self.memman) # Link MemoryManager to LLMManager
        logger.info("LLM manager initialized and linked with MemoryManager.")

        if not self.audio: raise TTSManagerError("AudioManager failed to initialize before TTSManager.")
        self.tts = TTSManager(self.cfg, gui_queue, self.audio); logger.info("TTS manager initialized.")

    def _populate_initial_ui_history(self):
        if self.memman and self.ui and self.ui.root: # Check and use memman
            initial_history = self.memman.get_full_history() # Get history via memman
            if initial_history:
                logger.info(f"Populating UI with {len(initial_history)} messages from loaded history.")
                def populate_task():
                    if not self.ui or not self.ui.history_textbox or not self.ui.history_textbox.winfo_exists():
                        logger.warning("Cannot populate history, UI textbox not available.")
                        return
                    self.ui.clear_history_display()
                    for message in initial_history:
                        role = message.get("role")
                        content = message.get("content")
                        if role and content and role in ["user", "assistant", "system"]:
                            self.ui.append_history(role, content)
                    self.ui.update_status("Ready (History Loaded)")
                self.ui.schedule_task(populate_task)
            else:
                logger.info("No previous conversation history found or loaded.")
        else:
             logger.warning("Cannot populate initial UI history: MemoryManager or UIManager not ready.") # Updated log message

    def _poll_gui_queue(self) -> None:
        if self._shutting_down: return
        try:
            while not gui_queue.empty():
                msg = gui_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty: pass
        except Exception as e:
            logger.error(f"Error processing GUI queue: {e}", exc_info=True)
            if self.ui: self.ui.log(f"Queue processing error: {e}", tag="critical")
        finally:
            if not self._shutting_down and self.ui and self.ui.root:
                try:
                    if self.ui.root.winfo_exists():
                        self.ui.root.after(100, self._poll_gui_queue)
                    else:
                        logger.warning("UI root window destroyed, stopping GUI queue polling.")
                except tkinter.TclError:
                    logger.warning("TclError checking UI window, stopping GUI queue polling (likely closing).")

    def _handle_message(self, message: Dict[str, Any]) -> None:
        if self._shutting_down or not self.ui: return
        msg_type = message.get("type")
        payload = message.get("payload")
        tag = message.get("tag", "info")
        try:
            if msg_type == "log": self.ui.log(str(payload), tag=tag)
            elif msg_type == "status": self.ui.update_status(str(payload))
            elif msg_type == "audio_ready": self._process_audio_ready(payload)
            elif msg_type == "stt_result": self._process_stt_result(payload)
            elif msg_type == "llm_chunk": self._process_llm_chunk(payload)
            elif msg_type == "llm_result": self._process_llm_result(payload)
            elif msg_type == "shutdown_request": self._handle_close_request()
            else:
                logger.warning(f"Received unknown message type in GUI queue: {msg_type}")
                self.ui.log(f"Unknown message type received: {msg_type}", tag="warning")
        except AttributeError as e:
            if "winfo_exists" in str(e):
                 logger.warning(f"UI widget likely destroyed during message handling ({msg_type}). Error: {e}")
            else:
                 logger.error(f"AttributeError handling message type '{msg_type}': {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error handling message type '{msg_type}': {e}", exc_info=True)
            if self.ui: self.ui.log(f"Error handling message: {e}", tag="error")

    def _process_audio_ready(self, payload: Any):
        if isinstance(payload, dict) and "filepath" in payload:
            path = str(payload["filepath"])
            duration = payload.get("duration", 0.0)
            self.ui.update_status("Transcribing...")
            self.ui.log(f"Transcribing recorded audio ({duration:.2f}s)...", tag="info")
            self._run_stt_in_background(path)
        else:
            if self.ui:
                self.ui.log("Error: Invalid audio data received.", tag="error")
                self.ui.update_status("ERROR: Invalid Audio")

    def _process_stt_result(self, payload: Any):
        if isinstance(payload, str):
            text = payload.strip()
            if text:
                if self.ui:
                    self.ui.log("Transcription complete.", tag="info")
                    self.ui.append_history("user", text)
                self._submit_user_input(text)
            else:
                if self.ui:
                    self.ui.log("No speech detected in audio.", tag="warning")
                    self.ui.update_status("Ready (No speech)")
                    self.ui.append_history_event("(No speech detected)")
        elif payload is None:
            if self.ui:
                self.ui.log("Speech transcription failed. Check logs.", tag="error")
                self.ui.update_status("ERROR: Transcription Failed")
                self.ui.append_history_event("(Transcription Error)")
        else:
            if self.ui:
                self.ui.log(f"Invalid STT payload type: {type(payload)}", tag="error")
                self.ui.update_status("ERROR: Invalid STT Data")

    def _process_llm_chunk(self, payload: Any):
        if isinstance(payload, dict) and "delta" in payload:
            if self.ui:
                if not self.ui._assistant_streaming: # Assuming _assistant_streaming is an attribute in UIManager
                    self.ui.start_assistant_stream()
                self.ui.append_stream_chunk(str(payload["delta"]))
        else:
            logger.error(f"Invalid llm_chunk payload: {payload}")

    def _process_llm_result(self, payload: Any):
        if self.ui: self.ui.finish_assistant_stream()

        if not isinstance(payload, dict):
            if self.ui: self.ui.log(f"Invalid llm_result payload type: {type(payload)}", tag="error"); self.ui.update_status("ERROR: Invalid LLM Data")
            return

        text = payload.get("text")
        error = payload.get("error", False)
        err_msg = payload.get("error_message", "")

        if error:
            if self.ui: self.ui.log(f"LLM Error: {err_msg}", tag="error"); self.ui.update_status("ERROR: LLM Failed"); self.ui.append_history_event(f"(LLM Error: {err_msg[:60]}...)")
        elif text:
            if self.tts and self.audio:
                if self.ui: self.ui.update_status("Speaking..."); self.ui.log("Sending response to TTS...", tag="info")
                if not self.audio.is_playing and not self.audio.is_recording:
                    self.tts.speak_text(text)
                else:
                    logger.warning("Audio manager busy. Cannot speak TTS response now.")
                    if self.ui: self.ui.log("Audio busy; cannot speak.", tag="warning"); self.ui.update_status("Ready (Audio busy)")
            else:
                logger.error("TTS or Audio manager not available to speak.")
                if self.ui: self.ui.update_status("ERROR: TTS Not Ready"); self.ui.update_status("Ready") # Reset status
        else:
            if self.ui: self.ui.log("Assistant provided no speakable response.", tag="warning"); self.ui.update_status("Ready (No response)"); self.ui.append_history_event("(Assistant gave no response)")

    def _run_stt_in_background(self, filepath: str) -> None:
        if not self.stt:
            logger.error("STT Manager not available.")
            gui_queue.put({"type": "stt_result", "payload": None})
            return
        def worker():
            result: Optional[str] = None
            try:
                result = self.stt.transcribe(filepath)
            except Exception as e:
                logger.error(f"STT background thread exception: {e}", exc_info=True)
            finally:
                gui_queue.put({"type": "stt_result", "payload": result})
                try:
                    p = Path(filepath)
                    if p.exists() and "temp_input" in p.name and p.suffix == ".wav":
                         p.unlink()
                         logger.debug(f"Deleted temporary STT input file: {filepath}")
                except Exception as e:
                    logger.warning(f"Error deleting temp STT file {filepath}: {e}")
        threading.Thread(target=worker, name="STTWorker", daemon=True).start()

    def _submit_user_input(self, text: str):
        if not text: return
        if not self.memman or not self.llm or not self.ui: # Check memman
             logger.error("Cannot process user input: Core components missing (memman, llm, or ui).")
             if self.ui:
                self.ui.update_status("ERROR: Core component error")
                self.ui.log("Internal error processing input.", tag="error")
             return
        try:
            self.memman.add_message("user", text) # Use memman to add the user's message
            logger.debug(f"User input added to memory via MemoryManager: '{text[:50]}...'")
        except Exception as e:
            logger.error(f"Error adding user message to memory: {e}", exc_info=True)
            if self.ui: self.ui.log(f"Error saving to memory: {e}", tag="error")
            # Decide if we should stop if context add fails. For now, let LLM proceed.

        if self.ui:
            self.ui.update_status("Thinking...")
            self.ui.log("Sending request to LLM (with MemoryManager)...", tag="info") # Updated log
        self.llm.run_llm_in_background(text)

    def _handle_record_button_click(self) -> None:
        if self._shutting_down or not self._backend_ready or not self.audio or not self.ui:
            logger.debug("Ignoring record button click (shutting down or backend not ready).")
            return
        if self.audio.is_playing:
            logger.info("Record button clicked while playing; stopping playback first.")
            self.audio.stop_playback()
            self.ui.schedule_task(self._toggle_recording)
        else:
            self._toggle_recording()

    def _toggle_recording(self) -> None:
        if not self.audio or not self.ui: return
        default_color = "grey"
        try: default_color = customtkinter.ThemeManager.theme["CTkButton"]["fg_color"]
        except (KeyError, TypeError): logger.warning("Could not get default button theme color.")

        if self.audio.is_recording:
            self.audio.stop_recording()
            self.ui.set_record_button_state("Record", color=default_color, enabled=True)
        else:
            self.ui.set_record_button_state("Stop Recording", color="red", enabled=True)
            self.audio.start_recording()

    def _setup_ptt_bindings_via_ui(self) -> None:
        if not self.cfg or not self.audio or not self.ui: return
        combo = self.cfg.get("activation", "push_to_talk_key")
        if not combo or not isinstance(combo, str):
            logger.info("Push-to-Talk key not configured or invalid. PTT disabled.")
            return
        try:
            if not (combo.startswith("<") and combo.endswith(">") and "-" in combo):
                logger.error(f"Invalid PTT key format in config: '{combo}'. Expected like '<Control-space>'.")
                return
            parts = combo.strip("<>").split('-')
            key = parts[-1] if parts else ""
            if not key:
                 logger.error(f"Could not extract key from PTT combo: '{combo}'")
                 return
            release_event = f"<KeyRelease-{key}>"
            success = self.ui.bind_ptt(
                press_event=combo,
                release_event=release_event,
                ptt_start_callback=self._handle_ptt_start,
                ptt_stop_callback=self._handle_ptt_stop
            )
            if success:
                self.ui.log(f"Push-to-Talk enabled ({combo})", tag="info")
                self.ui.set_record_button_state(f"Record (Hold {combo})", enabled=True)
            else:
                if self.ui: self.ui.log(f"Failed to bind PTT key ({combo}). See logs.", tag="error")
        except Exception as e:
            logger.error(f"Error setting up PTT bindings for '{combo}': {e}", exc_info=True)
            if self.ui: self.ui.log(f"Error binding PTT key ({combo}).", tag="error")

    def _handle_ptt_start(self) -> None:
        if self._shutting_down or not self._backend_ready or not self.audio or not self.ui: return
        if self.audio.is_playing:
            logger.info("PTT pressed while playing; stopping playback first.")
            self.audio.stop_playback()
            if self.ui and self.ui.root: # Check if UI root exists before scheduling
                self.ui.schedule_task(lambda: self.ui.root.after(50, self._handle_ptt_start))
            return
        if not self.audio.is_recording:
            logger.debug("PTT Start: Triggering recording toggle.")
            self._toggle_recording()

    def _handle_ptt_stop(self) -> None:
        if self._shutting_down or not self._backend_ready or not self.audio or not self.ui: return
        if self.audio.is_recording:
            logger.debug("PTT Stop: Triggering recording toggle.")
            self._toggle_recording()

    def _handle_theme_change(self, mode: str) -> None:
        if self.cfg and self.ui:
            logger.info(f"Saving theme preference: {mode}")
            self.cfg.update_value("gui", "theme_preference", mode.lower())
            self.cfg.save()
            self.ui.log(f"Theme preference saved: {mode}", tag="info")

    def _handle_clear_history(self) -> None:
        logger.info("Clear History requested by user.")
        if self.memman and self.ui: # Use memman
            try:
                self.memman.clear_memory() # Use memman to clear context
                self.ui.clear_history_display()
                self.ui.log("Conversation history, memory, and vector index cleared.", tag="info")
                self.ui.update_status("Ready (History Cleared)")
            except Exception as e:
                logger.error(f"Error clearing memory: {e}", exc_info=True)
                if self.ui: self.ui.log(f"Error clearing memory: {e}", tag="error")
        else:
            logger.warning("MemoryManager or UIManager not available for clear history.")

    def _handle_about(self) -> None:
        if self.ui:
            self.ui.show_about_dialog(self.APP_NAME, self.APP_VERSION)

    def _handle_text_input_submit(self, text: str) -> None:
        logger.info(f"Text input submitted: '{text[:60]}...'")
        if self._shutting_down or not self._backend_ready:
            logger.warning("Ignoring text input during shutdown or if backend not ready.")
            return
        if not text:
            logger.warning("Received empty text input from UI callback.")
            return
        if self.ui:
            self.ui.append_history("user", text)
        else:
            logger.error("UI not available to display submitted text.")
            return
        self._submit_user_input(text)

    def _handle_close_request(self) -> None:
        if self._shutting_down:
            logger.debug("Shutdown already in progress.")
            return
        self._shutting_down = True
        logger.info("Shutdown requested. Initiating graceful shutdown...")

        if self.ui and self.ui.root and self.ui.root.winfo_exists():
            try: self.ui.update_status("Shutting down...")
            except Exception: pass

        # Save raw history via ContextManager (which MemoryManager uses indirectly for data persistence)
        if self.ctx:
            try:
                logger.info("Saving final conversation history (via ContextManager)...")
                self.ctx.save_context()
                logger.info("Conversation history saved.")
            except Exception as e:
                logger.error(f"Failed to save context during shutdown: {e}", exc_info=True)
            # Shutdown ContextManager (e.g., release Chroma resources)
            try:
                self.ctx.shutdown()
            except Exception as e:
                 logger.error(f"Error shutting down ContextManager: {e}", exc_info=True)
        
        # Shutdown MemoryManager (if it has specific shutdown tasks in the future)
        if self.memman:
            try:
                self.memman.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down MemoryManager: {e}", exc_info=True)

        self._stop_backend_managers()
        if self.ui:
            try:
                logger.info("Destroying UI window...")
                self.ui.destroy()
                logger.info("UI window destroyed.")
            except Exception as e:
                logger.error(f"Error destroying UI window during shutdown: {e}", exc_info=True)
        logger.info(f"{self.APP_NAME} shutdown complete.")

    def _stop_backend_managers(self) -> None:
        logger.info("Stopping backend managers...")
        if self.audio:
            try: logger.debug("Stopping AudioManager..."); self.audio.stop(); logger.debug("AudioManager stopped.")
            except Exception as e: logger.warning(f"Error stopping AudioManager: {e}", exc_info=True)
        # Add other explicit stop calls here if managers require them
        logger.info("Backend managers stopped.")

    def run(self) -> None:
        if not self._backend_ready:
            logger.critical("Backend not ready. Cannot start UI main loop.")
            sys.exit(1)
        if not self.ui:
            logger.critical("Cannot run: UI Manager not initialized.")
            sys.exit(1)
        try:
            logger.info("Entering UI main loop.")
            self.ui.run()
            logger.info("UI main loop exited normally.")
        except Exception as e:
            logger.critical(f"UI main loop crashed: {e}", exc_info=True)
            self._handle_close_request()
            sys.exit(1)

_controller_instance: Optional[MiraiAppController] = None
def _signal_handler(sig, frame) -> None:
    logger.warning(f"Received signal {signal.Signals(sig).name} ({sig}). Requesting graceful shutdown.")
    global _controller_instance
    if _controller_instance and not _controller_instance._shutting_down:
        try:
            gui_queue.put_nowait({"type": "shutdown_request"})
        except queue.Full:
            logger.error("GUI queue full during signal handling. Forcing exit.")
            os._exit(1) # Force exit if queue is unresponsive
        except Exception as e:
            logger.error(f"Error putting shutdown request on queue: {e}. Forcing exit.")
            os._exit(1)
    elif _controller_instance and _controller_instance._shutting_down:
        logger.warning("Shutdown already in progress. Signal ignored.")
    else:
        logger.info("Controller instance not found during signal handling. Exiting.")
        sys.exit(0)

def main() -> None:
    global _controller_instance
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    controller = None
    try:
        controller = MiraiAppController()
        _controller_instance = controller
        controller.run()
    except SystemExit:
        logger.info("SystemExit caught. Exiting application.")
    except KeyboardInterrupt:
         logger.info("KeyboardInterrupt caught. Initiating shutdown...")
         if _controller_instance and not _controller_instance._shutting_down:
              _controller_instance._handle_close_request()
    except Exception as e:
        logger.critical(f"Unhandled top-level exception in main: {e}", exc_info=True)
        try:
            root = Tk(); root.withdraw()
            messagebox.showerror("Critical Error", f"A critical unhandled error occurred:\n\n{e}\n\nPlease check logs.")
        except Exception:
            print(f"\nCRITICAL UNHANDLED ERROR (Dialog Failed): {e}\n", file=sys.stderr)
            traceback.print_exc()
        finally:
            if _controller_instance and hasattr(_controller_instance, '_handle_close_request') and not _controller_instance._shutting_down:
                 logger.info("Attempting cleanup after top-level error...")
                 try: _controller_instance._handle_close_request()
                 except Exception as cleanup_e: logger.error(f"Error during final cleanup attempt: {cleanup_e}")
            sys.exit(1)

if __name__ == "__main__":
    main()