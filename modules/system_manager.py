# ================================================
# FILE: modules/system_manager.py (Corrected Checks)
# ================================================
"""
SystemManager for MiraiAssist.

Handles:
- Application-wide logging setup (console + file), optionally using Rich for console.
- System information logging (OS, Python, hardware).
- Critical runtime requirement validation.
"""
from __future__ import annotations

import sys
import os
import platform
import subprocess
import socket
import logging
import logging.handlers
import importlib
import shutil
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

# Use relative import for ConfigManager and other local modules
from .config_manager import ConfigManager
# Corrected import path if context_manager is directly under modules
from .context_manager import ContextManager

# --- Rich Integration ---
try:
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    class RichHandler: pass # Dummy class
# ------------------------

# Optional PyTorch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class torch: # Dummy class
        @staticmethod
        def cuda_is_available(): return False
        @staticmethod
        def cuda_device_count(): return 0
        @staticmethod
        def cuda_get_device_name(i): return ""
        __version__ = "Not Installed"
        class version: cuda = "N/A"

logger = logging.getLogger(__name__)

class LoggingSetupError(Exception):
    """Custom exception for errors during logging configuration."""
    pass

class RequirementError(Exception):
    """Custom exception raised when a critical runtime requirement is not met."""
    pass


class SystemManager:
    """
    Manages system-level tasks like logging setup, information gathering,
    and requirement verification for the MiraiAssist application.
    """

    MIN_PYTHON_VERSION: Tuple[int, int] = (3, 9) # Minimum required Python version

    # Default logging settings (used if config is missing keys)
    DEFAULT_LOG_FORMAT = '%(asctime)s [%(levelname)-8s] %(name)-25s %(message)s'
    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    DEFAULT_CONSOLE_LEVEL = "INFO"
    DEFAULT_APP_LOG_LEVEL = "DEBUG"
    DEFAULT_ERROR_LOG_LEVEL = "WARNING"
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_APP_LOG_FILE = "mirai_assist.log"
    DEFAULT_ERROR_LOG_FILE = "errors.log"
    DEFAULT_LOG_MAX_BYTES = 5 * 1024 * 1024 # 5 MB
    DEFAULT_LOG_BACKUP_COUNT = 3

    def __init__(self, cfg: ConfigManager) -> None:
        """
        Initializes the SystemManager.
        """
        if not cfg.is_loaded:
            raise ValueError("ConfigManager must be loaded before initializing SystemManager.")
        self.cfg = cfg
        self._sysinfo_logged = False
        self.console_handler: Optional[logging.Handler] = None
        logger.debug("SystemManager initialized.")

    # --------------------------------------------------------------------- #
    # Logging Setup
    # --------------------------------------------------------------------- #

    def _rotate_log_file(self, file_path: Path) -> None:
        """Rotates a single log file if it exists and is non-empty."""
        try:
            if file_path.exists() and file_path.stat().st_size > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = file_path.with_name(f"{file_path.stem}_{timestamp}{file_path.suffix}")
                shutil.move(str(file_path), str(backup_path))
                logger.debug(f"Rotated existing log file: '{file_path.name}' -> '{backup_path.name}'")
        except OSError as e:
            logger.error(f"Error rotating log file '{file_path}': {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error rotating log file '{file_path}': {e}", exc_info=True)

    def setup_logging(self) -> None:
        """
        Configures application-wide logging based on settings in ConfigManager.
        Sets up console (optionally with Rich) and rotating file handlers.
        """
        logger.info("Setting up application logging...")
        try:
            log_cfg = self.cfg.get("logging", default={})

            # Basic Logger Setup
            root_logger = logging.getLogger()
            if root_logger.hasHandlers():
                 logger.debug("Clearing existing logging handlers.")
                 for handler in root_logger.handlers[:]:
                      root_logger.removeHandler(handler)
                      handler.close()
            root_logger.setLevel(logging.DEBUG)

            # Standard Formatter (for file logs primarily)
            log_format = log_cfg.get("format", self.DEFAULT_LOG_FORMAT)
            date_format = log_cfg.get("date_format", self.DEFAULT_DATE_FORMAT)
            standard_formatter = logging.Formatter(log_format, datefmt=date_format)

            # Console Handler (Rich or Standard)
            console_level_str = log_cfg.get("console_log_level", self.DEFAULT_CONSOLE_LEVEL).upper()
            console_level = getattr(logging, console_level_str, logging.INFO)
            use_rich = log_cfg.get("rich_console_logging", False)

            if use_rich and RICH_AVAILABLE:
                logger.info("Configuring Rich console handler.")
                rich_keywords = log_cfg.get("rich_keywords", [])
                if not isinstance(rich_keywords, list) or not rich_keywords: rich_keywords = None

                rich_handler = RichHandler(
                    level=console_level,
                    show_time=log_cfg.get("rich_show_time", True),
                    show_level=log_cfg.get("rich_show_level", True),
                    show_path=log_cfg.get("rich_show_path", False),
                    markup=log_cfg.get("rich_markup", True),
                    rich_tracebacks=log_cfg.get("rich_tracebacks", True),
                    tracebacks_show_locals=log_cfg.get("rich_tracebacks_show_locals", False),
                    keywords=rich_keywords,
                )
                root_logger.addHandler(rich_handler)
                self.console_handler = rich_handler
                logger.info(f"Rich console logging enabled at level: {console_level_str}")

            elif use_rich and not RICH_AVAILABLE:
                logger.warning("Rich console logging enabled in config, but 'rich' library not found. Falling back to standard handler.")
                # Fall through to standard handler setup
                use_rich = False # Ensure we proceed with standard handler

            if not use_rich: # Standard StreamHandler
                logger.info("Configuring standard console handler.")
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(standard_formatter)
                stream_handler.setLevel(console_level)
                root_logger.addHandler(stream_handler)
                self.console_handler = stream_handler
                logger.info(f"Standard console logging enabled at level: {console_level_str}")


            # File Logging
            if log_cfg.get("file_logging_enabled", True):
                log_dir_path_str = log_cfg.get("log_directory", self.DEFAULT_LOG_DIR)
                log_directory = Path(log_dir_path_str).resolve()
                try:
                    log_directory.mkdir(parents=True, exist_ok=True)
                    logger.info(f"File logging enabled. Log directory: {log_directory}")
                except OSError as e:
                    logger.error(f"Failed to create log directory '{log_directory}': {e}. File logging disabled.", exc_info=True)
                    return

                # Application Log File Handler
                app_log_filename = log_cfg.get("application_log_file", self.DEFAULT_APP_LOG_FILE)
                app_log_path = log_directory / app_log_filename
                app_log_level_str = log_cfg.get("application_log_level", self.DEFAULT_APP_LOG_LEVEL).upper()
                app_log_level = getattr(logging, app_log_level_str, logging.DEBUG)
                max_bytes = int(log_cfg.get("log_max_bytes", self.DEFAULT_LOG_MAX_BYTES))
                backup_count = int(log_cfg.get("log_backup_count", self.DEFAULT_LOG_BACKUP_COUNT))
                self._rotate_log_file(app_log_path)
                try:
                    app_file_handler = logging.handlers.RotatingFileHandler(
                        app_log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
                    )
                    app_file_handler.setFormatter(standard_formatter)
                    app_file_handler.setLevel(app_log_level)
                    root_logger.addHandler(app_file_handler)
                    logger.info(f"Application file logging configured: '{app_log_path.name}' at level {app_log_level_str}")
                except Exception as e: logger.error(f"Failed to setup application file logger '{app_log_path}': {e}", exc_info=True)

                # Error Log File Handler
                error_log_filename = log_cfg.get("error_log_file", self.DEFAULT_ERROR_LOG_FILE)
                error_log_path = log_directory / error_log_filename
                error_log_level_str = log_cfg.get("error_log_level", self.DEFAULT_ERROR_LOG_LEVEL).upper()
                error_log_level = getattr(logging, error_log_level_str, logging.WARNING)
                self._rotate_log_file(error_log_path)
                try:
                    error_file_handler = logging.handlers.RotatingFileHandler(
                        error_log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
                    )
                    error_file_handler.setFormatter(standard_formatter)
                    error_file_handler.setLevel(error_log_level)
                    root_logger.addHandler(error_file_handler)
                    logger.info(f"Error file logging configured: '{error_log_path.name}' at level {error_log_level_str}")
                except Exception as e: logger.error(f"Failed to setup error file logger '{error_log_path}': {e}", exc_info=True)
            else:
                logger.info("File logging is disabled via configuration.")

            logger.info("Logging setup complete.")

        except Exception as e:
            logger.exception("An critical error occurred during logging setup.")
            raise LoggingSetupError(f"Failed to configure logging: {e}") from e

    # --------------------------------------------------------------------- #
    # System Information Logging
    # --------------------------------------------------------------------- #

    def log_system_info(self) -> None:
        """Logs key system and environment details."""
        if self._sysinfo_logged:
            logger.debug("System info already logged. Skipping.")
            return

        logger.info("----- System Information -----")
        try:
            logger.info(f"OS            : {platform.system()} {platform.release()} ({platform.machine()})")
            logger.info(f"Python        : {platform.python_version()} ({platform.python_implementation()})")
            logger.info(f"Python Path   : {sys.executable}")
            # Use project_root derived in main.py if available via cfg perhaps, or re-derive
            project_root_path = Path(__file__).resolve().parents[1] # Assuming modules/system_manager.py
            logger.info(f"Project Root  : {project_root_path}") # Adjust if structure differs
            logger.info(f"Config File   : {self.cfg.config_path}")

            # uv Version (Best Effort)
            try:
                uv_path = shutil.which("uv")
                if uv_path:
                    result = subprocess.run([uv_path, "--version"], capture_output=True, text=True, timeout=3, check=False, encoding='utf-8')
                    uv_version = result.stdout.strip() if result.returncode == 0 else f"Error ({result.returncode})"
                else: uv_version = "<'uv' command not found in PATH>"
                logger.info(f"uv Version    : {uv_version}")
            except Exception as e: logger.info(f"uv Version    : <Error checking: {e}>")

            # PyTorch & CUDA
            logger.info(f"PyTorch       : Version {torch.__version__}" if TORCH_AVAILABLE else "PyTorch       : Not Installed")
            if TORCH_AVAILABLE:
                try:
                    cuda_available = torch.cuda.is_available()
                    logger.info(f"  CUDA Status : {'Available' if cuda_available else 'Not Available or Not Setup'}")
                    if cuda_available:
                        cuda_version = getattr(torch.version, "cuda", "Unknown")
                        logger.info(f"  CUDA Version: {cuda_version}")
                        device_count = torch.cuda.device_count()
                        logger.info(f"  GPU Count   : {device_count}")
                        for i in range(device_count):
                            try: gpu_name = torch.cuda.get_device_name(i); logger.info(f"  GPU {i}       : {gpu_name}")
                            except Exception as e: logger.warning(f"  GPU {i}       : <Error getting name: {e}>")
                    else:
                         # Check config vs reality
                         stt_device = self.cfg.get("stt", "device", "cpu").lower()
                         tts_device = self.cfg.get("tts", "device", "cpu").lower()
                         if 'cuda' in [stt_device, tts_device]:
                              logger.warning("  Configuration requests CUDA, but torch.cuda.is_available() is False.")
                except Exception as e: logger.error(f"  Error checking PyTorch/CUDA details: {e}", exc_info=True)

            # Context Store Path (RAG version)
            context_cfg = self.cfg.get("context_manager", default={})
            storage_path = context_cfg.get("storage_path", ContextManager.DEFAULT_STORAGE_PATH)
            vector_db_path = context_cfg.get("vector_db_path", ContextManager.DEFAULT_VECTOR_DB_PATH)
            logger.info(f"Context Store : History='{storage_path}', Vector DB='{vector_db_path}'")
            logger.info("------------------------------")
            self._sysinfo_logged = True

        except Exception as e:
            logger.error(f"Error gathering system information: {e}", exc_info=True)
            logger.info("----- System Information End (incomplete) -----")


    # --------------------------------------------------------------------- #
    # Requirement Verification
    # --------------------------------------------------------------------- #

    def _check_python_version(self) -> bool:
        """Checks if the current Python version meets the minimum requirement."""
        if sys.version_info < self.MIN_PYTHON_VERSION:
            logger.critical(f"CRITICAL: Python version {self.MIN_PYTHON_VERSION[0]}.{self.MIN_PYTHON_VERSION[1]}+ required. Found: {platform.python_version()}")
            return False
        logger.info(f"✓ Python version check passed ({platform.python_version()})")
        return True # <<< Added return True

    def _check_import(self, module_name: str, package_name: Optional[str] = None, purpose: str = "") -> bool:
        """Checks if a module can be imported."""
        install_name = package_name if package_name else module_name
        purpose_str = f" ({purpose})" if purpose else ""
        try:
            importlib.import_module(module_name)
            logger.info(f"✓ Dependency check passed: {module_name}{purpose_str}")
            return True # <<< Added return True
        except ImportError:
            logger.critical(f"CRITICAL: Missing required module '{module_name}'{purpose_str}. Install with: uv add {install_name}")
            return False
        except Exception as e:
             logger.critical(f"CRITICAL: Error importing module '{module_name}'{purpose_str}: {e}", exc_info=True)
             return False

    def _check_cuda_availability(self) -> bool:
        """Checks if CUDA is available via PyTorch, if configured to be used."""
        stt_device = self.cfg.get("stt", "device", "cpu").lower()
        tts_device = self.cfg.get("tts", "device", "cpu").lower() # Check TTS too if it might use GPU
        needs_cuda = 'cuda' in [stt_device, tts_device]

        if not needs_cuda:
            logger.info("✓ CUDA check skipped (not configured for use in STT/TTS).")
            return True # <<< Added return True

        if not TORCH_AVAILABLE:
            logger.critical("CRITICAL: CUDA requested (device='cuda'), but PyTorch is not installed.")
            return False

        if not torch.cuda.is_available():
            logger.critical("CRITICAL: CUDA requested (device='cuda'), but torch.cuda.is_available() returned False. Check drivers and PyTorch CUDA build.")
            return False

        logger.info("✓ CUDA availability check passed.")
        return True # <<< Added return True

    def _check_llm_endpoint(self) -> bool:
        """Checks basic network connectivity to the configured LLM API endpoint."""
        base_url: Optional[str] = self.cfg.get("llm", "api_base_url")
        if not base_url:
            logger.critical("CRITICAL: LLM endpoint URL (llm.api_base_url) is not configured.")
            return False

        try:
            parsed_url = urlparse(base_url)
            hostname = parsed_url.hostname
            port = parsed_url.port

            if not hostname:
                 logger.critical(f"CRITICAL: Could not parse hostname from LLM endpoint URL: {base_url}")
                 return False

            if port is None: port = 443 if parsed_url.scheme == "https" else 80

            logger.info(f"Checking LLM endpoint connectivity: {hostname}:{port} (from {base_url})")
            with socket.create_connection((hostname, port), timeout=5.0):
                logger.info(f"✓ LLM endpoint check passed: Successfully connected to {hostname}:{port}")
            return True # <<< Added return True (after successful connection)

        except socket.timeout:
            logger.critical(f"CRITICAL: Cannot reach LLM endpoint: Connection to {hostname}:{port} timed out.")
            return False
        except socket.gaierror as e:
             logger.critical(f"CRITICAL: Cannot reach LLM endpoint: DNS resolution failed for {hostname}. Error: {e}")
             return False
        except OSError as e:
            logger.critical(f"CRITICAL: Cannot reach LLM endpoint {hostname}:{port}. Error: {e}")
            return False
        except Exception as e:
             logger.critical(f"CRITICAL: Unexpected error checking LLM endpoint {base_url}: {e}", exc_info=True)
             return False

    def _check_context_store_writability(self) -> bool:
        """Checks if the context storage file path and vector DB path are writable."""
        context_cfg = self.cfg.get("context_manager", default={})
        storage_path_str = context_cfg.get("storage_path", ContextManager.DEFAULT_STORAGE_PATH)
        vector_db_path_str = context_cfg.get("vector_db_path", ContextManager.DEFAULT_VECTOR_DB_PATH)
        store_path = Path(storage_path_str).resolve()
        vector_path = Path(vector_db_path_str).resolve()

        paths_to_check = {
            "History": store_path.parent, # Check parent dir for JSON file
            "Vector DB": vector_path      # Check ChromaDB dir itself
        }
        all_writable = True

        for name, dir_path in paths_to_check.items():
            try:
                # Check if directory exists, try to create if not
                if not dir_path.exists():
                    logger.info(f"Context store ({name}) directory does not exist, attempting to create: {dir_path}")
                    dir_path.mkdir(parents=True, exist_ok=True)
                elif not dir_path.is_dir():
                     logger.critical(f"CRITICAL: Context store ({name}) path exists but is not a directory: {dir_path}")
                     all_writable = False
                     continue # Stop checking this path, move to next

                # Check directory write permissions
                temp_file_path = dir_path / f".writetest_{os.getpid()}_{datetime.now().timestamp()}"
                try:
                    with open(temp_file_path, 'w') as f: f.write('test')
                    temp_file_path.unlink() # Clean up the temporary file
                    logger.info(f"✓ Context store ({name}) writability check passed (directory: {dir_path})")
                    # Don't return True here yet, need to check all paths
                except OSError as e:
                     logger.critical(f"CRITICAL: Cannot write to context store ({name}) directory '{dir_path}'. Check permissions. Error: {e}")
                     all_writable = False
                finally:
                     temp_file_path.unlink(missing_ok=True) # Ensure cleanup

            except OSError as e:
                logger.critical(f"CRITICAL: Error accessing or creating context store ({name}) directory '{dir_path}': {e}")
                all_writable = False
            except Exception as e:
                 logger.critical(f"CRITICAL: Unexpected error checking context store ({name}) writability ({dir_path}): {e}", exc_info=True)
                 all_writable = False

        # Return the overall result after checking all paths
        return all_writable

    # Helper for Rich dependency check called by verify_requirements
    def _check_rich_dependency(self) -> bool:
         """Checks Rich install only if enabled in config. Returns True if disabled or installed."""
         if not self.cfg.get("logging", "rich_console_logging", False):
              logger.info("✓ Dependency check skipped: Rich (Console Logging disabled in config)")
              return True # Pass if not enabled
         # If enabled, check import
         if not RICH_AVAILABLE:
             logger.critical("CRITICAL: Missing optional module 'rich' required for rich_console_logging. Install with: uv add rich")
             return False
         logger.info("✓ Dependency check passed: Rich (for Console Logging)")
         return True # <<< Added return True


    # --- Main Verification Method ---

    def verify_requirements(self) -> None:
        """
        Runs all critical pre-flight checks for the application.
        """
        logger.info("Running critical requirement checks...")
        failures: List[str] = []

        # Define Checks
        checks_to_run = [
            (self._check_python_version, "Python Version"),
            # Core Dependencies
            (lambda: self._check_import("yaml", "pyyaml", "Config parsing"), "PyYAML"),
            (lambda: self._check_import("customtkinter", purpose="GUI Toolkit"), "CustomTkinter"),
            (lambda: self._check_import("pyaudio", purpose="Audio I/O"), "PyAudio"),
            (lambda: self._check_import("numpy", purpose="Audio/Numeric Processing"), "NumPy"),
            (lambda: self._check_import("soundfile", purpose="Audio File I/O"), "SoundFile"),
             # Rich (optional, check if enabled via _check_rich_dependency)
            (self._check_rich_dependency, "Rich (Console Logging)"),
            # AI Modules
            (lambda: self._check_import("faster_whisper", package_name="faster-whisper", purpose="STT"), "Faster Whisper"),
            (lambda: self._check_import("openai", purpose="LLM Client"), "OpenAI Client"),
            # (lambda: self._check_import("tiktoken", purpose="LLM Tokenizer"), "TikToken"), # Still optional
            (lambda: self._check_import("kokoro", purpose="TTS"), "Kokoro TTS"),
            # RAG Dependencies
            (lambda: self._check_import("sentence_transformers", purpose="RAG Embeddings"), "Sentence Transformers"),
            (lambda: self._check_import("chromadb", purpose="RAG Vector Store"), "ChromaDB"),
            # Hardware/Connectivity
            (self._check_cuda_availability, "CUDA Availability"),
            (self._check_llm_endpoint, "LLM Endpoint Connectivity"),
            (self._check_context_store_writability, "Context Store Writability"),
        ]

        # Run Checks
        all_passed = True
        for check_func, check_name in checks_to_run:
            try:
                # Execute the check function and store the boolean result
                result = check_func()
                # Explicitly check if the result is False (covers None or other non-True values implicitly)
                if result is False: # Check specifically for False
                    # Avoid adding Rich failure here if it passed because it was disabled
                    if not (check_name == "Rich (Console Logging)" and self.cfg.get("logging", "rich_console_logging", False) is False):
                         failures.append(check_name)
                         all_passed = False
                elif result is not True:
                     # Log if a check didn't return True or False explicitly (potential bug)
                     logger.warning(f"Requirement check '{check_name}' returned non-boolean value: {result}. Treating as failure.")
                     failures.append(f"{check_name} (Bad Return)")
                     all_passed = False

            except Exception as e:
                 # Catch unexpected errors within the check function itself
                 logger.critical(f"CRITICAL: Unexpected error during requirement check '{check_name}': {e}", exc_info=True)
                 failures.append(f"{check_name} (Error)")
                 all_passed = False


        # Report Results
        if not all_passed:
            failure_summary = ", ".join(failures)
            logger.critical("*** APPLICATION STARTUP BLOCKED ***")
            logger.critical(f"Failed requirement checks: {failure_summary}")
            logger.critical("Please resolve the issues listed above and restart the application.")
            raise RequirementError(f"Failed checks: {failure_summary}")
        else:
            logger.info("✓ All critical system requirement checks passed.")