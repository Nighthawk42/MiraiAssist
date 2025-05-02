# ================================================
# FILE: main.py (RAG + UI Text Input Integration - Corrected)
# ================================================

from __future__ import annotations

import sys
import signal
import queue
import threading
import tkinter
import traceback
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Use tkinter for basic error dialogs if GUI fails early
from tkinter import Tk, messagebox

# Import customtkinter only if GUI is intended
try:
    import customtkinter
    CUSTOMTKINTER_AVAILABLE = True
except ImportError:
    CUSTOMTKINTER_AVAILABLE = False

# --- PATH & LOGGING SETUP ---
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
     sys.path.insert(0, str(project_root))

# Basic logging setup FIRST (refined later by SystemManager)
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__) # Logger for main.py

# --- IMPORT CORE MANAGERS ---
try:
    from modules.config_manager import ConfigManager, ConfigError
    from modules.system_manager import SystemManager, LoggingSetupError, RequirementError
    from modules.audio_manager import AudioManager, AudioManagerError
    from modules.stt_manager import STTManager, STTManagerError
    from modules.context_manager import ContextManager, ContextManagerError # RAG version
    from modules.llm_manager import LLMManager, LLMManagerError       # RAG version
    from modules.tts_manager import TTSManager, TTSManagerError
    from modules.ui_manager import UIManager                          # UI with text input
except ImportError as exc:
    print(f"FATAL ERROR: Failed to import core modules: {exc}", file=sys.stderr)
    traceback.print_exc()
    try:
        root = Tk(); root.withdraw()
        messagebox.showerror("Startup Error", f"Failed to import core modules:\n\n{exc}\n\nPlease check dependencies (e.g., run 'uv sync') and logs.")
    except Exception: pass
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: Unexpected error during initial imports: {e}", file=sys.stderr)
    traceback.print_exc()
    try:
         root = Tk(); root.withdraw()
         messagebox.showerror("Startup Error", f"An unexpected error occurred during startup:\n\n{e}\n\nPlease check logs.")
    except Exception: pass
    sys.exit(1)

# Check if customtkinter loaded if we intend to use it
if not CUSTOMTKINTER_AVAILABLE:
     print("FATAL ERROR: CustomTkinter library is required but not found. Please install it.", file=sys.stderr)
     sys.exit(1)

# --- SHARED QUEUE ---
gui_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

# --- APPLICATION CONTROLLER ---
class MiraiAppController:
    APP_NAME    = UIManager.APP_NAME
    APP_VERSION = UIManager.APP_VERSION

    def __init__(self) -> None:
        logger.info(f"Initializing {self.APP_NAME} Controller v{self.APP_VERSION}")

        # Manager references initialization
        self.cfg: Optional[ConfigManager] = None
        self.sysman: Optional[SystemManager] = None
        self.ctx: Optional[ContextManager] = None
        self.audio: Optional[AudioManager] = None
        self.stt: Optional[STTManager] = None
        self.llm: Optional[LLMManager] = None
        self.tts: Optional[TTSManager] = None
        self.ui: Optional[UIManager] = None

        self._backend_ready = False
        self._shutting_down = False

        # --- Initialize Backend ---
        try:
            self._initialize_backend()
        except (ConfigError, LoggingSetupError, RequirementError, ContextManagerError,
                  AudioManagerError, STTManagerError, LLMManagerError, TTSManagerError) as e:
            logger.critical(f"CRITICAL Backend initialization failed: {type(e).__name__}: {e}", exc_info=True)
            self._show_startup_error_dialog(f"Backend Initialization Failed:\n\n{type(e).__name__}: {e}\n\nPlease check configuration and logs.")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"Unexpected critical error during backend init: {e}", exc_info=True)
            self._show_startup_error_dialog(f"Unexpected Critical Error during Startup:\n\n{e}\n\nPlease check logs.")
            sys.exit(1)

        if not self._backend_ready:
            logger.critical("Backend initialization process completed but backend is not marked as ready.")
            self._show_startup_error_dialog("Backend initialization did not complete successfully.\nCannot start UI. Check logs for details.")
            sys.exit(1)

        # --- Initialize UI ---
        try:
            self.ui = UIManager(gui_queue, self.cfg)
            # ---> Make sure the callback method exists before calling build_window <---
            if not hasattr(self, '_handle_text_input_submit'):
                 # This is a sanity check, the AttributeError should prevent getting here usually
                 raise AttributeError("Internal Error: _handle_text_input_submit method is missing in MiraiAppController.")

            self.ui.build_window(
                on_close_callback      = self._handle_close_request,
                record_callback        = self._handle_record_button_click,
                theme_change_callback  = self._handle_theme_change,
                clear_history_callback = self._handle_clear_history,
                about_callback         = self._handle_about,
                text_submit_callback   = self._handle_text_input_submit # Correctly pass the method reference
            )
            self.ui.update_status("Ready")
            self.ui.set_record_button_state("Record", enabled=True)
            self._setup_ptt_bindings_via_ui()

            # --- Populate UI with loaded history ---
            self._populate_initial_ui_history()

        except Exception as e:
            logger.critical(f"Failed to initialize UI Manager: {e}", exc_info=True)
            self._stop_backend_managers() # Attempt cleanup
            self._show_startup_error_dialog(f"FATAL ERROR: UI Initialization failed: {e}\n\nPlease check logs.")
            sys.exit(1)

        # --- Start GUI Queue Polling ---
        if self.ui and self.ui.root:
            self.ui.root.after(100, self._poll_gui_queue)
        else:
             logger.critical("UI Root window not available after initialization. Cannot poll queue.")
             self._handle_close_request()
             sys.exit(1)


    def _show_startup_error_dialog(self, message: str):
        """Helper to show critical startup errors via Tkinter."""
        try:
            root = Tk(); root.withdraw()
            messagebox.showerror(f"{self.APP_NAME} - Critical Startup Error", message)
        except Exception as tk_error:
            print(f"\nCRITICAL STARTUP ERROR (Tkinter Dialog Failed: {tk_error}):\n{message}\n", file=sys.stderr)


    def _initialize_backend(self) -> None:
        """Initializes backend managers sequentially. Raises exceptions on failure."""
        logger.info("Starting backend initialization...")
        self.cfg = ConfigManager()
        self.cfg.load()
        logger.info("Configuration loaded.")

        self.sysman = SystemManager(self.cfg)
        self.sysman.setup_logging()
        self.sysman.log_system_info()

        self.sysman.verify_requirements()
        logger.info("Requirement checks passed.")

        self.ctx = ContextManager(self.cfg)
        logger.info("Context manager (RAG) ready.")

        self.audio = AudioManager(self.cfg, gui_queue)
        logger.info("Audio manager ready.")

        self.stt = STTManager(self.cfg)
        logger.info("STT manager ready.")

        if not self.ctx: raise LLMManagerError("ContextManager failed to initialize before LLMManager.")
        self.llm = LLMManager(self.cfg, gui_queue)
        self.llm.set_context_manager(self.ctx)
        logger.info("LLM manager ready.")

        if not self.audio: raise TTSManagerError("AudioManager failed to initialize before TTSManager.")
        self.tts = TTSManager(self.cfg, gui_queue, self.audio)
        logger.info("TTS manager ready.")

        self._backend_ready = True
        logger.info("Backend initialization successful.")


    def _populate_initial_ui_history(self):
        """Loads history from ContextManager into the UI."""
        if self.ctx and self.ui:
            initial_history = self.ctx.history
            if initial_history:
                logger.info(f"Populating UI with {len(initial_history)} loaded messages from history file.")
                def populate_ui():
                    if not self.ui or not self.ui.history_textbox or not self.ui.history_textbox.winfo_exists():
                        logger.warning("Cannot populate history, UI textbox not available.")
                        return
                    for message in initial_history:
                        role = message.get("role")
                        content = message.get("content")
                        if role and content and role in ["user", "assistant"]:
                            self.ui.append_history(role, content)
                    self.ui.update_status("Ready (History Loaded)")
                self.ui.schedule_task(populate_ui)
            else:
                 logger.info("No previous conversation history found to display.")


    def _poll_gui_queue(self) -> None:
        """Polls the shared queue for messages from backend threads."""
        if self._shutting_down: return
        try:
            while not gui_queue.empty():
                msg = gui_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty: pass
        except Exception as e: logger.error(f"Error processing GUI queue: {e}", exc_info=True)
        finally:
            if not self._shutting_down and self.ui and self.ui.root:
                try:
                    if self.ui.root.winfo_exists():
                        self.ui.root.after(100, self._poll_gui_queue)
                    else: logger.warning("UI root window destroyed, stopping GUI queue polling.")
                except tkinter.TclError: logger.warning("TclError checking UI window, stopping GUI queue polling (likely closing).")


    def _handle_message(self, message: Dict[str, Any]) -> None:
        """Routes messages from the queue to appropriate UI actions."""
        if self._shutting_down or not self.ui: return

        msg_type = message.get("type")
        payload = message.get("payload")
        tag = message.get("tag", "info")

        if msg_type == "log": self.ui.log(str(payload), tag=tag)
        elif msg_type == "status": self.ui.update_status(str(payload))
        elif msg_type == "audio_ready": self._process_audio_ready(payload)
        elif msg_type == "stt_result": self._process_stt_result(payload)
        elif msg_type == "llm_chunk": self._process_llm_chunk(payload)
        elif msg_type == "llm_result": self._process_llm_result(payload)
        elif msg_type == "shutdown_request": self._handle_close_request()
        else:
            logger.warning(f"Received unknown message type in GUI queue: {msg_type}")
            self.ui.log(f"Unknown message type: {msg_type}", tag="warning")


    def _process_audio_ready(self, payload: Any):
        """Handles the 'audio_ready' message."""
        if isinstance(payload, dict) and "filepath" in payload:
            path = str(payload["filepath"])
            duration = payload.get("duration", 0.0)
            self.ui.update_status("Transcribing…")
            self.ui.log(f"Transcribing recorded audio ({duration:.2f}s)…", tag="info")
            self._run_stt_in_background(path)
        else:
            self.ui.log("Error: Invalid audio data received.", tag="error")
            self.ui.update_status("ERROR: Invalid Audio")

    def _process_stt_result(self, payload: Any):
        """Handles the 'stt_result' message and triggers the LLM."""
        if isinstance(payload, str):
            text = payload.strip()
            if text:
                self.ui.log("Transcription complete.", tag="info")
                # Display the user's transcribed message *before* submitting
                self.ui.append_history("user", text)
                # Use common submission logic
                self._submit_user_input(text)
            else:
                self.ui.log("No speech detected in audio.", tag="warning")
                self.ui.update_status("Ready (No speech detected)")
                self.ui.append_history_event("(No speech detected)")
        elif payload is None: # Explicit None indicates error
            self.ui.log("Speech transcription failed. Check logs.", tag="error")
            self.ui.update_status("ERROR: Transcription Failed")
            self.ui.append_history_event("(Transcription Error)")
        else:
            self.ui.log(f"Invalid STT payload type: {type(payload)}", tag="error")
            self.ui.update_status("ERROR: Invalid STT Data")

    def _process_llm_chunk(self, payload: Any):
        """Handles incoming LLM stream chunks."""
        if isinstance(payload, dict) and "delta" in payload:
            if not self.ui._assistant_streaming:
                self.ui.start_assistant_stream()
            self.ui.append_stream_chunk(str(payload["delta"]))
        else:
            logger.error(f"Invalid llm_chunk payload: {payload}")

    def _process_llm_result(self, payload: Any):
        """Handles the final LLM result and triggers TTS."""
        self.ui.finish_assistant_stream()

        if not isinstance(payload, dict):
            self.ui.log(f"Invalid llm_result payload type: {type(payload)}", tag="error")
            self.ui.update_status("ERROR: Invalid LLM Data")
            return

        text = payload.get("text")
        error = payload.get("error", False)
        err_msg = payload.get("error_message", "")

        if error:
            self.ui.log(f"LLM Error: {err_msg}", tag="error")
            self.ui.update_status("ERROR: LLM Failed")
            self.ui.append_history_event(f"(LLM Error: {err_msg[:60]}...)")
        elif text:
            if self.tts and self.audio:
                self.ui.update_status("Speaking…")
                self.ui.log("Sending response to TTS…", tag="info")
                if not self.audio.is_playing and not self.audio.is_recording:
                    self.tts.speak_text(text)
                else:
                    logger.warning("Audio manager busy. Cannot speak TTS response.")
                    self.ui.log("Audio busy; cannot speak.", tag="warning")
                    self.ui.update_status("Ready (Audio busy)")
            else:
                logger.error("TTS or Audio manager not available to speak.")
                self.ui.update_status("ERROR: TTS Not Ready")
        else: # No error, but no text
            self.ui.log("Assistant provided no response.", tag="warning")
            self.ui.update_status("Ready (No response)")
            self.ui.append_history_event("(Assistant gave no response)")


    def _run_stt_in_background(self, filepath: str) -> None:
        """Runs STT transcription in a separate thread."""
        if not self.stt:
            logger.error("STT Manager not available for background task.")
            gui_queue.put({"type": "stt_result", "payload": None})
            return

        def worker():
            result: Optional[str] = None
            try: result = self.stt.transcribe(filepath)
            except Exception as e: logger.error(f"STT background worker exception: {e}", exc_info=True)
            finally:
                gui_queue.put({"type": "stt_result", "payload": result})
                try:
                    p = Path(filepath)
                    if p.exists() and "temp_input" in p.name: p.unlink(); logger.debug(f"Deleted temporary STT input file: {filepath}")
                except Exception as e: logger.warning(f"Error deleting temp file {filepath}: {e}")

        threading.Thread(target=worker, name="STTWorker", daemon=True).start()


    def _submit_user_input(self, text: str):
        """Handles user text input, updating context and triggering LLM."""
        if not text: return
        if not self.ctx or not self.llm or not self.ui:
             logger.error("Cannot process user input: Core components missing.")
             self.ui.update_status("ERROR: Core component missing")
             self.ui.log("Internal error: Cannot process user input.", tag="error")
             return

        # Note: UI display (append_history) is now handled by the *caller*
        # (_process_stt_result or _handle_text_input_submit) before calling this.

        try:
            self.ctx.add_message("user", text)
            logger.debug(f"User input added to RAG context: '{text[:50]}...'")
        except Exception as e:
            logger.error(f"Error adding user message to context: {e}", exc_info=True)
            self.ui.log(f"Error saving context: {e}", tag="error")
            # Decide if we should stop here if context fails
            # return # Optionally stop if context add fails

        # Run LLM
        self.ui.update_status("Thinking...")
        self.ui.log("Sending text to LLM (with RAG)...", tag="info")
        self.llm.run_llm_in_background(text)


    def _handle_record_button_click(self) -> None:
        if self._shutting_down or not self._backend_ready or not self.audio or not self.ui: return
        if self.audio.is_playing:
            logger.info("Record button clicked while playing; stopping playback first.")
            self.audio.stop_playback()
            self.ui.schedule_task(self._toggle_recording)
        else: self._toggle_recording()

    def _toggle_recording(self) -> None:
        if not self.audio or not self.ui: return
        default_color = customtkinter.ThemeManager.theme["CTkButton"]["fg_color"]
        if self.audio.is_recording:
            self.audio.stop_recording()
            self.ui.set_record_button_state("Record", color=default_color, enabled=True)
        else:
            self.ui.set_record_button_state("Stop Recording", color="red", enabled=True)
            self.audio.start_recording()

    def _setup_ptt_bindings_via_ui(self) -> None:
        if not self.cfg or not self.audio or not self.ui: return
        combo = self.cfg.get("activation", "push_to_talk_key", "<Control-space>")
        if not combo: logger.info("Push-to-Talk key not configured. PTT disabled."); return
        try:
            parts = combo.strip("<>").split("-"); key = parts[-1] if parts else ""
            if not key: logger.error(f"Invalid PTT key format: '{combo}'"); return
            success = self.ui.bind_ptt(combo, f"<KeyRelease-{key}>", self._handle_ptt_start, self._handle_ptt_stop)
            if success: self.ui.log(f"Push-to-Talk enabled ({combo})", tag="info")
            else: self.ui.log(f"Failed to bind PTT key ({combo}). See logs.", tag="error")
        except Exception as e: logger.error(f"Error setting up PTT bindings for '{combo}': {e}", exc_info=True)

    def _handle_ptt_start(self) -> None:
        if self._shutting_down or not self._backend_ready or not self.audio or not self.ui: return
        if self.audio.is_playing:
            logger.info("PTT pressed while playing; stopping playback first.")
            self.audio.stop_playback()
            self.ui.schedule_task(self._handle_ptt_start)
            return
        if not self.audio.is_recording:
            logger.debug("PTT Start: Triggering recording toggle.")
            self._toggle_recording()

    def _handle_ptt_stop(self) -> None:
        if self._shutting_down or not self._backend_ready or not self.audio or not self.ui: return
        if self.audio.is_recording:
            logger.debug("PTT Stop: Triggering recording toggle.")
            self._toggle_recording()


    # ────────────────── UI MENU & TEXT INPUT CALLBACKS ────────────────────────
    def _handle_theme_change(self, mode: str) -> None:
        if self.cfg:
            logger.info(f"Saving theme preference: {mode}")
            self.cfg.update_value("gui", "theme_preference", mode.lower())
            self.cfg.save()
            self.ui.log(f"Theme preference saved: {mode}", tag="info")

    def _handle_clear_history(self) -> None:
        logger.info("Clear History requested by user.")
        if self.ctx and self.ui:
            try:
                self.ctx.clear_context()
                self.ui.clear_history_display()
                self.ui.log("Conversation history and vector index cleared.", tag="info")
                self.ui.update_status("Ready (History Cleared)")
            except Exception as e:
                logger.error(f"Error clearing context: {e}", exc_info=True)
                self.ui.log(f"Error clearing context: {e}", tag="error")
        else: logger.warning("ContextManager or UIManager not available for clear history.")

    def _handle_about(self) -> None:
        if self.ui: self.ui.show_about_dialog(self.APP_NAME, self.APP_VERSION)

    # <<< METHOD DEFINITION IS NOW CORRECTLY HERE >>>
    def _handle_text_input_submit(self, text: str):
        """Callback triggered by UIManager when text is submitted."""
        logger.info(f"Text input received: '{text[:60]}...'")
        if self._shutting_down or not self._backend_ready:
            logger.warning("Ignoring text input during shutdown or if backend not ready.")
            return
        if not text:
            logger.warning("Received empty text input from UI callback.")
            return

        # Display the user's typed message FIRST
        if self.ui:
            self.ui.append_history("user", text)
        else:
             logger.error("UI not available to display submitted text.")
             return

        # Use the common submission logic
        self._submit_user_input(text)


    # ────────────────── SHUTDOWN ─────────────────────────────────────────────
    def _handle_close_request(self) -> None:
        if self._shutting_down: return
        self._shutting_down = True
        logger.info("Shutdown requested. Initiating graceful shutdown...")
        if self.ui:
            try: self.ui.update_status("Shutting down…")
            except Exception: pass
        if self.ctx:
            try:
                logger.info("Saving full conversation history before shutdown...")
                self.ctx.save_context()
                logger.info("Conversation history saved.")
                self.ctx.shutdown()
            except Exception as e: logger.error(f"Failed to save/shutdown context: {e}", exc_info=True)
        self._stop_backend_managers()
        if self.ui:
            try: logger.info("Destroying UI window..."); self.ui.destroy(); logger.info("UI window destroyed.")
            except Exception as e: logger.error(f"Error destroying UI window: {e}", exc_info=True)
        logger.info("MiraiAssist shutdown complete.")


    def _stop_backend_managers(self) -> None:
        logger.info("Stopping backend managers...")
        if self.audio:
            try: logger.debug("Stopping AudioManager..."); self.audio.stop(); logger.debug("AudioManager stopped.")
            except Exception as e: logger.warning(f"Error stopping AudioManager: {e}", exc_info=True)


    # ────────────────── MAIN LOOP ─────────────────────────────────────────────
    def run(self) -> None:
        if not self._backend_ready: logger.critical("Backend not ready. Cannot start UI main loop."); sys.exit(1)
        if self.ui:
            try: logger.info("Entering UI main loop."); self.ui.run(); logger.info("UI main loop exited normally.")
            except Exception as e: logger.critical(f"UI main loop crashed: {e}", exc_info=True); self._handle_close_request(); sys.exit(1)
        else: logger.critical("Cannot run: UI Manager not initialized."); sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL HANDLER & ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
_controller_instance: Optional[MiraiAppController] = None

def _signal_handler(sig, frame) -> None:
    logger.warning(f"Received signal {sig}; requesting graceful shutdown.")
    global _controller_instance
    if _controller_instance and not _controller_instance._shutting_down:
        try: gui_queue.put_nowait({"type": "shutdown_request"})
        except queue.Full: logger.error("GUI queue full during signal handling. Forcing exit."); sys.exit(1)
        except Exception as e: logger.error(f"Error putting shutdown request on queue: {e}. Forcing exit."); sys.exit(1)
    elif _controller_instance and _controller_instance._shutting_down: logger.warning("Shutdown already in progress. Signal ignored.")
    else: logger.info("Controller instance not found during signal handling. Exiting."); sys.exit(0)

def main() -> None:
    global _controller_instance
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    controller = None
    try:
        controller = MiraiAppController()
        _controller_instance = controller
        controller.run()
    except SystemExit: raise # Allow clean exits
    except Exception as e:
        logger.critical(f"Unhandled top-level exception: {e}", exc_info=True)
        try:
            root = Tk(); root.withdraw()
            messagebox.showerror("Critical Error", f"A critical unhandled error occurred:\n\n{e}\n\nPlease check logs.")
        except Exception: print(f"\nCRITICAL UNHANDLED ERROR (Dialog Failed): {e}\n", file=sys.stderr); traceback.print_exc()
        finally:
            if controller and hasattr(controller, '_handle_close_request'):
                 logger.info("Attempting cleanup after top-level error...")
                 try: controller._handle_close_request()
                 except Exception as cleanup_e: logger.error(f"Error during final cleanup attempt: {cleanup_e}")
            sys.exit(1)

if __name__ == "__main__":
    main()