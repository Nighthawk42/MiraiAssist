# ================================================
# FILE: modules/ui_manager.py
# ================================================
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import messagebox, Event
import logging
from typing import TYPE_CHECKING, Optional, Callable, Dict, Any, Tuple, List

if TYPE_CHECKING:
    from .config_manager import ConfigManager
    import queue

try:
    import customtkinter
except ImportError as exc:
    print("FATAL ERROR: customtkinter library not found.", file=sys.stderr)
    raise

logger = logging.getLogger(__name__)

class UIManager:
    APP_NAME = "MiraiAssist"
    APP_VERSION = "0.0.1-RAG"

    HISTORY_FONT = ("Segoe UI", 16)
    LOGBOX_FONT = ("Consolas", 12)
    INPUT_FONT = ("Segoe UI", 14)

    RecordCallback = Callable[[], None]
    PTTStartCallback = Callable[[], None]
    PTTStopCallback = Callable[[], None]
    ThemeChangeCallback = Callable[[str], None]
    ClearHistoryCallback = Callable[[], None]
    CloseCallback = Callable[[], None]
    AboutCallback = Callable[[], None]
    TextSubmitCallback = Callable[[str], None]

    def __init__(self, gui_queue: 'queue.Queue[Dict[str, Any]]', config: Optional['ConfigManager'] = None):
        self.gui_queue = gui_queue
        self.config = config
        self.root: Optional[customtkinter.CTk] = None
        self.history_textbox: Optional[customtkinter.CTkTextbox] = None
        self.log_textbox: Optional[customtkinter.CTkTextbox] = None
        self.record_button: Optional[customtkinter.CTkButton] = None
        self.status_label: Optional[customtkinter.CTkLabel] = None
        # --- Changed from CTkEntry to CTkTextbox ---
        self.input_textbox: Optional[customtkinter.CTkTextbox] = None
        self.send_button: Optional[customtkinter.CTkButton] = None
        # --- Callback References ---
        self._record_callback: Optional[UIManager.RecordCallback] = None
        self._ptt_start_callback: Optional[UIManager.PTTStartCallback] = None
        self._ptt_stop_callback: Optional[UIManager.PTTStopCallback] = None
        self._theme_change_callback: Optional[UIManager.ThemeChangeCallback] = None
        self._clear_history_callback: Optional[UIManager.ClearHistoryCallback] = None
        self._close_callback: Optional[UIManager.CloseCallback] = None
        self._about_callback: Optional[UIManager.AboutCallback] = None
        self._text_submit_callback: Optional[UIManager.TextSubmitCallback] = None
        self._assistant_streaming = False

        initial_theme = "System"
        if self.config: initial_theme = self.config.get("gui", "theme_preference", "system")
        try: customtkinter.set_appearance_mode(initial_theme.lower()); logger.info(f"UI Theme: {initial_theme}")
        except ValueError: logger.warning(f"Invalid theme '{initial_theme}'. Defaulting to System."); customtkinter.set_appearance_mode("System")

    def build_window(self, on_close_callback: CloseCallback, record_callback: RecordCallback,
                     theme_change_callback: ThemeChangeCallback, clear_history_callback: ClearHistoryCallback,
                     about_callback: AboutCallback, text_submit_callback: TextSubmitCallback) -> None:
        if self.root and self.root.winfo_exists(): logger.warning("build_window called but window exists."); return

        self._close_callback = on_close_callback
        self._record_callback = record_callback
        self._theme_change_callback = theme_change_callback
        self._clear_history_callback = clear_history_callback
        self._about_callback = about_callback
        self._text_submit_callback = text_submit_callback

        self.root = customtkinter.CTk()
        self.root.title(f"{self.APP_NAME} v{self.APP_VERSION}")
        self.root.protocol("WM_DELETE_WINDOW", self._handle_close)
        self.root.minsize(650, 550)
        self.root.geometry("800x650")

        # --- Menu Bar --- (No changes)
        menubar = tk.Menu(self.root); self.root.configure(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0); file_menu.add_command(label="Exit", command=self._handle_close); menubar.add_cascade(label="File", menu=file_menu)
        options_menu = tk.Menu(menubar, tearoff=0); theme_menu = tk.Menu(options_menu, tearoff=0)
        for mode in ["Light", "Dark", "System"]: theme_menu.add_command(label=mode, command=lambda m=mode: self._handle_theme_change(m))
        options_menu.add_cascade(label="Theme", menu=theme_menu); options_menu.add_separator(); options_menu.add_command(label="Clear Conversation History", command=self._handle_clear_history); menubar.add_cascade(label="Options", menu=options_menu)
        help_menu = tk.Menu(menubar, tearoff=0); help_menu.add_command(label="About", command=self._handle_about); menubar.add_cascade(label="Help", menu=help_menu)

        # --- Main Layout ---
        self.root.grid_columnconfigure(0, weight=1); self.root.grid_rowconfigure(0, weight=1); self.root.grid_rowconfigure(1, weight=0)
        main_frame = customtkinter.CTkFrame(self.root, fg_color="transparent"); main_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1); main_frame.grid_rowconfigure(0, weight=3); main_frame.grid_rowconfigure(1, weight=1); main_frame.grid_rowconfigure(2, weight=0); main_frame.grid_rowconfigure(3, weight=0)

        # History & Log Textboxes
        self.history_textbox = customtkinter.CTkTextbox(main_frame, wrap=tk.WORD, state=tk.DISABLED, border_width=1, corner_radius=4, font=self.HISTORY_FONT); self.history_textbox.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        self.log_textbox = customtkinter.CTkTextbox(main_frame, wrap=tk.WORD, state=tk.DISABLED, height=100, border_width=1, corner_radius=4, font=self.LOGBOX_FONT); self.log_textbox.grid(row=1, column=0, sticky="nsew", pady=5)
        self._configure_history_tags(); self._configure_log_tags() # Configure after creation

        # Record Button
        self.record_button = customtkinter.CTkButton(main_frame, text="Record", height=35, command=self._handle_record_click, state=tk.DISABLED); self.record_button.grid(row=2, column=0, pady=(5, 5), sticky="ew")

        # --- Input Frame ---
        input_frame = customtkinter.CTkFrame(main_frame, fg_color="transparent"); input_frame.grid(row=3, column=0, pady=(5, 0), sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1); input_frame.grid_columnconfigure(1, weight=0)

        # --- Use CTkTextbox for input ---
        self.input_textbox = customtkinter.CTkTextbox(
            input_frame,
            font=self.INPUT_FONT,
            height=70, # Approx 3 lines height initially
            border_width=1,
            corner_radius=4,
            wrap=tk.WORD # Enable word wrapping
            # placeholder_text="Type your message (Shift+Enter for newline)..." # Placeholder not directly supported in CTkTextbox
        )
        self.input_textbox.grid(row=0, column=0, padx=(0, 5), sticky="nsew") # Use nsew to allow vertical expansion if needed later
        # --- Bind Enter for submit, Shift+Enter for newline ---
        self.input_textbox.bind("<Return>", self._handle_textbox_enter)
        self.input_textbox.bind("<Shift-Return>", self._handle_textbox_shift_enter)
        # Add placeholder text manually (requires focus handling)
        self._add_placeholder()
        self.input_textbox.bind("<FocusIn>", self._remove_placeholder)
        self.input_textbox.bind("<FocusOut>", self._add_placeholder)

        # --- Send Button ---
        self.send_button = customtkinter.CTkButton(input_frame, text="Send", width=70, height=35, command=self._handle_text_submit); self.send_button.grid(row=0, column=1, padx=(0, 0), sticky="e") # Keep button aligned

        # Status Bar
        self.status_label = customtkinter.CTkLabel(self.root, text="Status: Initializing...", anchor="w", padx=10); self.status_label.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        logger.info("UIManager: Window and widgets built.")

    # --- Placeholder Text Handling for CTkTextbox ---
    PLACEHOLDER_TEXT = "Type message (Shift+Enter for newline)"
    PLACEHOLDER_COLOR = "gray50" # Example, adjust if needed

    def _add_placeholder(self, event=None):
        if not self.input_textbox: return
        # Add placeholder only if textbox is empty
        current_text = self.input_textbox.get("1.0", tk.END).strip()
        if not current_text:
            # Use helper to get placeholder color safely
            ph_color = self._get_theme_color(["CTkEntry", "placeholder_text_color"], fallback_light="gray70", fallback_dark="gray30")
            self.input_textbox.configure(text_color=ph_color)
            self.input_textbox.insert("1.0", self.PLACEHOLDER_TEXT)

    def _remove_placeholder(self, event=None):
        if not self.input_textbox: return
        # Remove placeholder only if it's currently displayed
        current_text = self.input_textbox.get("1.0", tk.END).strip()
        # Use helper to get normal text color
        normal_color = self._get_theme_color(["CTkTextbox", "text_color"])
        if current_text == self.PLACEHOLDER_TEXT:
            self.input_textbox.delete("1.0", tk.END)
            self.input_textbox.configure(text_color=normal_color)


    def _get_theme_color(self, theme_key_path: List[str], fallback_light: str = "black", fallback_dark: str = "white") -> str:
        try:
            color_val = customtkinter.ThemeManager.theme
            for key in theme_key_path: color_val = color_val[key]
            if isinstance(color_val, (list, tuple)) and len(color_val) == 2:
                return color_val[1] if customtkinter.get_appearance_mode() == "Dark" else color_val[0]
            elif isinstance(color_val, str): return color_val
            else: logger.warning(f"Unexpected theme color format: {theme_key_path}. Using fallback."); return fallback_dark if customtkinter.get_appearance_mode()=="Dark" else fallback_light
        except Exception as e: logger.error(f"Error getting theme color: {theme_key_path}: {e}. Using fallback.", exc_info=True); return fallback_dark if customtkinter.get_appearance_mode()=="Dark" else fallback_light

    def _configure_history_tags(self) -> None:
        if not self.history_textbox: return
        try:
            self.history_textbox.tag_config("user_prefix", foreground="#007AFF")
            self.history_textbox.tag_config("user", foreground="#007AFF")
            self.history_textbox.tag_config("assistant_prefix", foreground="#34C759")
            self.history_textbox.tag_config("assistant", foreground="#34C759")
            event_color = self._get_theme_color(["CTkEntry", "placeholder_text_color"], fallback_light="gray50", fallback_dark="gray60")
            self.history_textbox.tag_config("event", foreground=event_color)
            self.history_textbox.tag_config("system_prefix", foreground="#FF9500")
            self.history_textbox.tag_config("system", foreground="#FF9500")
        except Exception as e: logger.error(f"Error configuring history tags: {e}", exc_info=True)

    def _configure_log_tags(self) -> None:
        if not self.log_textbox: return
        try:
            default_text_color = self._get_theme_color(["CTkTextbox", "text_color"])
            placeholder_color = self._get_theme_color(["CTkEntry", "placeholder_text_color"], fallback_light="gray50", fallback_dark="gray60")
            self.log_textbox.tag_config("debug", foreground=placeholder_color)
            self.log_textbox.tag_config("info", foreground=default_text_color)
            self.log_textbox.tag_config("warning", foreground="#FFA500")
            self.log_textbox.tag_config("error", foreground="#FF4500")
            self.log_textbox.tag_config("critical", foreground="#DC143C")
        except Exception as e:
            logger.error(f"Error configuring log tags: {e}", exc_info=True)
            try: # Basic fallbacks
                self.log_textbox.tag_config("info", foreground="black" if customtkinter.get_appearance_mode() == "Light" else "white"); self.log_textbox.tag_config("debug", foreground="gray"); self.log_textbox.tag_config("warning", foreground="orange"); self.log_textbox.tag_config("error", foreground="red"); self.log_textbox.tag_config("critical", foreground="red")
            except Exception as fallback_e: logger.error(f"Error applying fallback log tag colors: {fallback_e}")

    # --- Event Handlers ---
    def _handle_close(self) -> None: logger.debug("Close triggered."); self._close_callback and self._close_callback()

    def _handle_record_click(self) -> None: logger.debug("Record clicked."); self._record_callback and self._record_callback()

    def _handle_theme_change(self, mode: str) -> None:
        logger.debug(f"Theme change to {mode}.")
        try:
            customtkinter.set_appearance_mode(mode.lower())
            self._configure_log_tags(); self._configure_history_tags() # Reconfigure tags
            logger.debug("Tags reconfigured.")
        except Exception as e: logger.error(f"Error applying theme '{mode}': {e}", exc_info=True)
        self._theme_change_callback and self._theme_change_callback(mode)

    def _handle_clear_history(self) -> None:
         logger.debug("Clear history clicked.")
         if messagebox.askyesno("Confirm Clear", "Clear conversation history? Cannot be undone."):
            self._clear_history_callback and self._clear_history_callback()

    def _handle_about(self) -> None:
         logger.debug("About clicked.")
         if self._about_callback: self._about_callback()
         else: self.show_about_dialog(self.APP_NAME, self.APP_VERSION)

    def _handle_text_submit(self) -> None:
        if not self.input_textbox: return
        text = self.input_textbox.get("1.0", tk.END).strip()
        # Ignore submission if it's just the placeholder text
        if text and text != self.PLACEHOLDER_TEXT:
            logger.debug(f"Text submitted: '{text[:50]}...'")
            if self._text_submit_callback:
                self.input_textbox.delete("1.0", tk.END)
                self._add_placeholder() # Re-add placeholder after clearing
                self._text_submit_callback(text)
            else: logger.warning("Text submit callback not set!")
        else:
            logger.debug("Empty or placeholder text submission ignored.")
            if not text: self._add_placeholder() # Ensure placeholder is back if empty

    def _handle_textbox_enter(self, event: Event) -> str:
        """Handles Enter key press in input textbox - submits text."""
        self._handle_text_submit()
        return "break" # Prevents default newline insertion

    def _handle_textbox_shift_enter(self, event: Event) -> None:
        """Handles Shift+Enter key press - allows default newline insertion."""
        # We don't return "break" here, allowing the default behavior
        pass

    # --- Public API Methods ---
    def schedule_task(self, callback: Callable[[], Any]) -> None:
        if self.root and hasattr(self.root, 'winfo_exists') and self.root.winfo_exists():
            try: self.root.after(0, callback)
            except Exception as e: logger.error(f"Error scheduling task: {e}", exc_info=True)
        else: logger.warning("Cannot schedule task, root window doesn't exist.")

    def update_status(self, text: str) -> None:
        def _task():
             if self.status_label and hasattr(self.status_label, 'winfo_exists') and self.status_label.winfo_exists():
                 try: self.status_label.configure(text=f"Status: {text}")
                 except Exception: pass
        self.schedule_task(_task)

    def log(self, text: str, tag: str = "info") -> None:
        def _task():
            if not self.log_textbox or not hasattr(self.log_textbox, 'winfo_exists') or not self.log_textbox.winfo_exists(): return
            try:
                self.log_textbox.configure(state=tk.NORMAL)
                actual_tag = tag if tag in self.log_textbox.tag_names() else "info"
                self.log_textbox.insert(tk.END, f"{text}\n", (actual_tag,))
                self.log_textbox.configure(state=tk.DISABLED)
                self.log_textbox.see(tk.END)
            except Exception as e: logger.error(f"Error logging to GUI: {e}", exc_info=True)
        self.schedule_task(_task)

    def append_history(self, role: str, text: str) -> None:
        def _task():
            if not self.history_textbox or not hasattr(self.history_textbox, 'winfo_exists') or not self.history_textbox.winfo_exists(): return
            try:
                self.history_textbox.configure(state=tk.NORMAL)
                prefix, prefix_tag, content_tag = role.capitalize(), "event", "event" # Default
                if role == "user": prefix, prefix_tag, content_tag = "You", "user_prefix", "user"
                elif role == "assistant": prefix, prefix_tag, content_tag = "Assistant", "assistant_prefix", "assistant"
                elif role == "system": prefix, prefix_tag, content_tag = "System", "system_prefix", "system"
                self.history_textbox.insert(tk.END, f"{prefix}: ", (prefix_tag,))
                self.history_textbox.insert(tk.END, f"{text}\n\n", (content_tag,))
                self.history_textbox.configure(state=tk.DISABLED)
                self.history_textbox.see(tk.END)
            except Exception as e: logger.error(f"Error appending history: {e}", exc_info=True)
        self.schedule_task(_task)

    def append_history_event(self, event_text: str) -> None:
         def _task():
             if not self.history_textbox or not hasattr(self.history_textbox, 'winfo_exists') or not self.history_textbox.winfo_exists(): return
             try:
                 self.history_textbox.configure(state=tk.NORMAL); self.history_textbox.insert(tk.END, f"{event_text}\n\n", ("event",)); self.history_textbox.configure(state=tk.DISABLED); self.history_textbox.see(tk.END)
             except Exception as e: logger.error(f"Error appending history event: {e}", exc_info=True)
         self.schedule_task(_task)

    def start_assistant_stream(self) -> None:
        def _task():
            if self._assistant_streaming: return
            if not self.history_textbox or not hasattr(self.history_textbox, 'winfo_exists') or not self.history_textbox.winfo_exists(): return
            try:
                self.history_textbox.configure(state=tk.NORMAL); self.history_textbox.insert(tk.END, "Assistant: ", ("assistant_prefix",)); self.history_textbox.configure(state=tk.DISABLED); self._assistant_streaming = True
            except Exception as e: logger.error(f"Error starting stream display: {e}", exc_info=True)
        self.schedule_task(_task)

    def append_stream_chunk(self, delta: str) -> None:
        def _task():
            if not self._assistant_streaming: self.start_assistant_stream(); self.schedule_task(lambda d=delta: self.append_stream_chunk(d)); return
            if not self.history_textbox or not hasattr(self.history_textbox, 'winfo_exists') or not self.history_textbox.winfo_exists(): return
            try:
                self.history_textbox.configure(state=tk.NORMAL); self.history_textbox.insert(tk.END, delta, ("assistant",)); self.history_textbox.configure(state=tk.DISABLED); self.history_textbox.see(tk.END)
            except Exception as e: logger.error(f"Error appending stream chunk: {e}", exc_info=True)
        self.schedule_task(_task)

    def finish_assistant_stream(self) -> None:
        def _task():
            if not self._assistant_streaming: return
            self._assistant_streaming = False
            if not self.history_textbox or not hasattr(self.history_textbox, 'winfo_exists') or not self.history_textbox.winfo_exists(): return
            try:
                self.history_textbox.configure(state=tk.NORMAL); self.history_textbox.insert(tk.END, "\n\n", ("assistant",)); self.history_textbox.configure(state=tk.DISABLED); self.history_textbox.see(tk.END)
            except Exception as e: logger.error(f"Error finishing stream display: {e}", exc_info=True)
        self.schedule_task(_task)

    def clear_history_display(self) -> None:
         def _task():
             if self.history_textbox and hasattr(self.history_textbox, 'winfo_exists') and self.history_textbox.winfo_exists():
                 try: self.history_textbox.configure(state=tk.NORMAL); self.history_textbox.delete("1.0", tk.END); self.history_textbox.configure(state=tk.DISABLED)
                 except Exception as e: logger.error(f"Failed to clear GUI history: {e}")
         self.schedule_task(_task)

    def set_record_button_state(self, text: str, color: Optional[str] = None, enabled: bool = True) -> None:
        def _task():
             if not self.record_button or not hasattr(self.record_button, 'winfo_exists') or not self.record_button.winfo_exists(): return
             try:
                 state = tk.NORMAL if enabled else tk.DISABLED
                 fg_color = color if color else self._get_theme_color(["CTkButton", "fg_color"])
                 self.record_button.configure(text=text, state=state, fg_color=fg_color)
             except Exception as e: logger.error(f"Failed to configure record button: {e}", exc_info=True)
        self.schedule_task(_task)

    def bind_ptt(self, press_event: str, release_event: str, ptt_start_callback: PTTStartCallback, ptt_stop_callback: PTTStopCallback) -> bool:
        self._ptt_start_callback = ptt_start_callback; self._ptt_stop_callback = ptt_stop_callback
        def _task():
            if not self.root or not hasattr(self.root, 'winfo_exists') or not self.root.winfo_exists(): return
            try:
                 self.root.bind_all(press_event, self._handle_ptt_press, add="+"); self.root.bind_all(release_event, self._handle_ptt_release, add="+")
                 logger.info(f"PTT Bindings enabled: Press='{press_event}', Release='{release_event}'")
            except Exception as e: logger.error(f"Error setting up PTT bindings: {e}", exc_info=True)
        self.schedule_task(_task)
        return True

    def _handle_ptt_press(self, event: Event) -> Optional[str]:
        try:
            focused = self.root.focus_get() if self.root else None
            # Check focus against CTkTextbox and the input_textbox
            if isinstance(focused, (customtkinter.CTkTextbox, tk.Text)) or focused == self.input_textbox:
                logger.debug(f"PTT ignored: Focus on text input widget.")
                return None
        except Exception: pass
        if self._ptt_start_callback: self._ptt_start_callback(); return "break"
        return None

    def _handle_ptt_release(self, event: Event) -> Optional[str]:
         if self._ptt_stop_callback: self._ptt_stop_callback(); return "break"
         return None

    def show_about_dialog(self, app_name: str, version: str) -> None:
        def _task():
             try: messagebox.showinfo(f"About {app_name}", f"{app_name} v{version}\n\nA modular voice assistant.\nPowered by many open source libraries.")
             except Exception: pass
        self.schedule_task(_task)

    def show_error_dialog(self, title: str, message: str) -> None:
         def _task():
              try: messagebox.showerror(title, message)
              except Exception: pass
         self.schedule_task(_task)

    def run(self) -> None:
        if not self.root: logger.error("UIManager: Cannot run, window not built."); return
        logger.info("UIManager: Starting main loop.")
        try: self.root.mainloop()
        except Exception as e: logger.critical(f"UI mainloop crash: {e}", exc_info=True); self._close_callback and self._close_callback(); raise

    def destroy(self) -> None:
        def _task():
            if self.root and hasattr(self.root, 'winfo_exists') and self.root.winfo_exists():
                logger.info("UIManager: Destroying main window.")
                try: self.root.destroy()
                except Exception: pass
            self.root = None
        if self.root: self.schedule_task(_task)
        else: logger.debug("UIManager: Destroy called but window gone.")