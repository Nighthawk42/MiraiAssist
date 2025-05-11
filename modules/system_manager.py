# modules/system_manager.py

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
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from urllib.parse import urlparse

from .config_manager import ConfigManager
from .context_manager import ContextManager # For default paths

try:
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    class RichHandler: # type: ignore
        def __init__(self, *args, **kwargs): pass # Dummy for type hints

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class torch: # type: ignore # Dummy class
        @staticmethod
        def cuda_is_available(): return False
        @staticmethod
        def cuda_device_count(): return 0
        @staticmethod
        def cuda_get_device_name(i): return ""
        __version__ = "Not Installed"
        class version: cuda = "N/A" # type: ignore

logger = logging.getLogger(__name__)

class LoggingSetupError(Exception):
    """Custom exception for errors during logging configuration."""
    pass

class RequirementError(Exception):
    """Custom exception raised when a critical runtime requirement is not met."""
    pass


class SystemManager:
    MIN_PYTHON_VERSION: Tuple[int, int] = (3, 9)
    DEFAULT_LOG_FORMAT = '%(asctime)s [%(levelname)-8s] %(name)-25s %(message)s'
    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    DEFAULT_CONSOLE_LEVEL = "INFO"
    DEFAULT_APP_LOG_LEVEL = "DEBUG"
    DEFAULT_ERROR_LOG_LEVEL = "WARNING"
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_APP_LOG_FILE = "mirai_assist.log"
    DEFAULT_ERROR_LOG_FILE = "errors.log"
    DEFAULT_LOG_MAX_BYTES = 5 * 1024 * 1024
    DEFAULT_LOG_BACKUP_COUNT = 3

    def __init__(self, cfg: ConfigManager) -> None:
        if not cfg.is_loaded:
            raise ValueError("ConfigManager must be loaded before initializing SystemManager.")
        self.cfg = cfg
        self._sysinfo_logged = False
        self.console_handler: Optional[logging.Handler] = None
        logger.debug("SystemManager initialized.")

    def _rotate_log_file(self, file_path: Path) -> None:
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
        logger.info("Setting up application logging...")
        try:
            log_cfg = self.cfg.get("logging", default={})
            root_logger = logging.getLogger()
            if root_logger.hasHandlers():
                 logger.debug("Clearing existing logging handlers.")
                 for handler in root_logger.handlers[:]:
                      root_logger.removeHandler(handler); handler.close()
            root_logger.setLevel(logging.DEBUG)

            log_format = log_cfg.get("format", self.DEFAULT_LOG_FORMAT)
            date_format = log_cfg.get("date_format", self.DEFAULT_DATE_FORMAT)
            standard_formatter = logging.Formatter(log_format, datefmt=date_format)

            console_level_str = log_cfg.get("console_log_level", self.DEFAULT_CONSOLE_LEVEL).upper()
            console_level = getattr(logging, console_level_str, logging.INFO)
            use_rich = log_cfg.get("rich_console_logging", False)

            if use_rich and RICH_AVAILABLE:
                logger.info("Configuring Rich console handler.")
                rich_keywords = log_cfg.get("rich_keywords", [])
                if not isinstance(rich_keywords, list) or not rich_keywords: rich_keywords = None
                rich_handler = RichHandler(
                    level=console_level, show_time=log_cfg.get("rich_show_time", True),
                    show_level=log_cfg.get("rich_show_level", True), show_path=log_cfg.get("rich_show_path", False),
                    markup=log_cfg.get("rich_markup", True), rich_tracebacks=log_cfg.get("rich_tracebacks", True),
                    tracebacks_show_locals=log_cfg.get("rich_tracebacks_show_locals", False), keywords=rich_keywords,
                )
                root_logger.addHandler(rich_handler); self.console_handler = rich_handler
                logger.info(f"Rich console logging enabled at level: {console_level_str}")
            elif use_rich and not RICH_AVAILABLE:
                logger.warning("Rich console logging enabled, but 'rich' not found. Falling back.")
                use_rich = False
            if not use_rich:
                logger.info("Configuring standard console handler.")
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(standard_formatter); stream_handler.setLevel(console_level)
                root_logger.addHandler(stream_handler); self.console_handler = stream_handler
                logger.info(f"Standard console logging enabled at level: {console_level_str}")

            if log_cfg.get("file_logging_enabled", True):
                log_dir_path_str = log_cfg.get("log_directory", self.DEFAULT_LOG_DIR)
                log_directory = Path(log_dir_path_str).resolve()
                try:
                    log_directory.mkdir(parents=True, exist_ok=True)
                    logger.info(f"File logging enabled. Log directory: {log_directory}")
                except OSError as e:
                    logger.error(f"Failed to create log directory '{log_directory}': {e}. File logging disabled.", exc_info=True)
                    return

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
                    app_file_handler.setFormatter(standard_formatter); app_file_handler.setLevel(app_log_level)
                    root_logger.addHandler(app_file_handler)
                    logger.info(f"Application file logging: '{app_log_path.name}' at level {app_log_level_str}")
                except Exception as e: logger.error(f"Failed to setup app file logger '{app_log_path}': {e}", exc_info=True)

                error_log_filename = log_cfg.get("error_log_file", self.DEFAULT_ERROR_LOG_FILE)
                error_log_path = log_directory / error_log_filename
                error_log_level_str = log_cfg.get("error_log_level", self.DEFAULT_ERROR_LOG_LEVEL).upper()
                error_log_level = getattr(logging, error_log_level_str, logging.WARNING)
                self._rotate_log_file(error_log_path)
                try:
                    error_file_handler = logging.handlers.RotatingFileHandler(
                        error_log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
                    )
                    error_file_handler.setFormatter(standard_formatter); error_file_handler.setLevel(error_log_level)
                    root_logger.addHandler(error_file_handler)
                    logger.info(f"Error file logging: '{error_log_path.name}' at level {error_log_level_str}")
                except Exception as e: logger.error(f"Failed to setup error file logger '{error_log_path}': {e}", exc_info=True)
            else:
                logger.info("File logging is disabled via configuration.")
            logger.info("Logging setup complete.")
        except Exception as e:
            logger.exception("Critical error during logging setup.") # Use logger.exception for auto exc_info
            raise LoggingSetupError(f"Failed to configure logging: {e}") from e

    def log_system_info(self) -> None:
        if self._sysinfo_logged: logger.debug("System info already logged."); return
        logger.info("----- System Information -----")
        try:
            logger.info(f"OS            : {platform.system()} {platform.release()} ({platform.machine()})")
            logger.info(f"Python        : {platform.python_version()} ({platform.python_implementation()})")
            logger.info(f"Python Path   : {sys.executable}")
            project_root_path = Path(__file__).resolve().parents[1]
            logger.info(f"Project Root  : {project_root_path}")
            logger.info(f"Config File   : {self.cfg.config_path}")
            try:
                uv_path = shutil.which("uv")
                if uv_path:
                    result = subprocess.run([uv_path, "--version"], capture_output=True, text=True, timeout=3, check=False, encoding='utf-8')
                    uv_version = result.stdout.strip() if result.returncode == 0 else f"Error ({result.returncode})"
                else: uv_version = "<'uv' command not found>"
                logger.info(f"uv Version    : {uv_version}")
            except Exception as e: logger.info(f"uv Version    : <Error checking: {e}>")

            logger.info(f"PyTorch       : Version {torch.__version__}" if TORCH_AVAILABLE else "PyTorch       : Not Installed")
            if TORCH_AVAILABLE:
                try:
                    cuda_available = torch.cuda.is_available()
                    logger.info(f"  CUDA Status : {'Available' if cuda_available else 'Not Available'}")
                    if cuda_available:
                        cuda_version = getattr(torch.version, "cuda", "Unknown") # type: ignore
                        logger.info(f"  CUDA Version: {cuda_version}")
                        device_count = torch.cuda.device_count()
                        logger.info(f"  GPU Count   : {device_count}")
                        for i in range(device_count):
                            try: gpu_name = torch.cuda.get_device_name(i); logger.info(f"  GPU {i}       : {gpu_name}")
                            except Exception as e_gpu: logger.warning(f"  GPU {i}       : <Error name: {e_gpu}>")
                    else:
                         stt_dev = self.cfg.get("stt", "device", "cpu").lower()
                         tts_dev = self.cfg.get("tts", "device", "cpu").lower() # Placeholder if TTS had GPU option
                         if 'cuda' in [stt_dev, tts_dev]: logger.warning("  Config requests CUDA, but torch.cuda.is_available() is False.")
                except Exception as e_torch: logger.error(f"  Error PyTorch/CUDA details: {e_torch}", exc_info=True)

            context_cfg = self.cfg.get("context_manager", default={})
            storage_path = context_cfg.get("storage_path", ContextManager.DEFAULT_STORAGE_PATH)
            vector_db_path = context_cfg.get("vector_db_path", ContextManager.DEFAULT_VECTOR_DB_PATH)
            logger.info(f"Context Store : History='{storage_path}', Vector DB='{vector_db_path}'")
            logger.info("------------------------------")
            self._sysinfo_logged = True
        except Exception as e:
            logger.error(f"Error gathering system information: {e}", exc_info=True)
            logger.info("----- System Information End (incomplete) -----")

    def _check_python_version(self) -> bool:
        if sys.version_info < self.MIN_PYTHON_VERSION:
            logger.critical(f"CRITICAL: Python {self.MIN_PYTHON_VERSION[0]}.{self.MIN_PYTHON_VERSION[1]}+ required. Found: {platform.python_version()}")
            return False
        logger.info(f"✓ Python version check passed ({platform.python_version()})")
        return True

    def _check_import(self, module_name: str, package_name: Optional[str] = None, purpose: str = "") -> bool:
        install_name = package_name if package_name else module_name
        purpose_str = f" ({purpose})" if purpose else ""
        try:
            importlib.import_module(module_name)
            logger.info(f"✓ Dependency check passed: {module_name}{purpose_str}")
            return True
        except ImportError:
            logger.critical(f"CRITICAL: Missing module '{module_name}'{purpose_str}. Install: uv add {install_name}")
            return False
        except Exception as e:
             logger.critical(f"CRITICAL: Error importing '{module_name}'{purpose_str}: {e}", exc_info=True)
             return False

    def _check_cuda_availability(self) -> bool:
        stt_device = self.cfg.get("stt", "device", "cpu").lower()
        # Example: if embedding model could also use GPU via context_manager config
        embedding_device_cfg = self.cfg.get("context_manager", "embedding_device", "cpu").lower()
        needs_cuda = 'cuda' in [stt_device, embedding_device_cfg]

        if not needs_cuda:
            logger.info("✓ CUDA check skipped (not configured for STT/Embeddings).")
            return True
        if not TORCH_AVAILABLE:
            logger.critical("CRITICAL: CUDA requested, but PyTorch is not installed.")
            return False
        if not torch.cuda.is_available():
            logger.critical("CRITICAL: CUDA requested, but torch.cuda.is_available() is False. Check drivers/PyTorch CUDA build.")
            return False
        logger.info("✓ CUDA availability check passed.")
        return True

    def _check_llm_endpoint(self) -> bool:
        base_url: Optional[str] = self.cfg.get("llm", "api_base_url")
        if not base_url:
            logger.critical("CRITICAL: LLM endpoint URL (llm.api_base_url) not configured.")
            return False
        try:
            parsed_url = urlparse(base_url)
            hostname, port = parsed_url.hostname, parsed_url.port
            if not hostname: logger.critical(f"CRITICAL: Cannot parse hostname from LLM URL: {base_url}"); return False
            if port is None: port = 443 if parsed_url.scheme == "https" else 80
            logger.info(f"Checking LLM endpoint: {hostname}:{port} (from {base_url})")
            with socket.create_connection((hostname, port), timeout=5.0):
                logger.info(f"✓ LLM endpoint check passed: Connected to {hostname}:{port}")
            return True
        except socket.timeout: logger.critical(f"CRITICAL: LLM endpoint timeout: {hostname}:{port}."); return False
        except socket.gaierror as e: logger.critical(f"CRITICAL: LLM endpoint DNS fail: {hostname}. Error: {e}"); return False
        except OSError as e: logger.critical(f"CRITICAL: LLM endpoint OSError {hostname}:{port}. Error: {e}"); return False
        except Exception as e: logger.critical(f"CRITICAL: LLM endpoint error {base_url}: {e}", exc_info=True); return False

    def _check_context_store_writability(self) -> bool:
        context_cfg = self.cfg.get("context_manager", default={})
        store_path = Path(context_cfg.get("storage_path", ContextManager.DEFAULT_STORAGE_PATH)).resolve()
        vector_path = Path(context_cfg.get("vector_db_path", ContextManager.DEFAULT_VECTOR_DB_PATH)).resolve()
        paths_to_check = {"History JSON Dir": store_path.parent, "Vector DB Dir": vector_path}
        all_writable = True
        for name, dir_path in paths_to_check.items():
            try:
                if not dir_path.exists(): dir_path.mkdir(parents=True, exist_ok=True); logger.info(f"Created context dir ({name}): {dir_path}")
                elif not dir_path.is_dir(): logger.critical(f"CRITICAL: Context path ({name}) not a dir: {dir_path}"); all_writable = False; continue
                temp_file = dir_path / f".writetest_{os.getpid()}_{datetime.now().timestamp()}"
                try:
                    with open(temp_file, 'w') as f: f.write('test')
                    logger.info(f"✓ Context store ({name}) writability passed (directory: {dir_path})")
                except OSError as e: logger.critical(f"CRITICAL: Cannot write to context dir ({name}) '{dir_path}': {e}"); all_writable = False
                finally: temp_file.unlink(missing_ok=True)
            except Exception as e: logger.critical(f"CRITICAL: Error checking context dir ({name}) '{dir_path}': {e}", exc_info=True); all_writable = False
        return all_writable

    def _check_rich_dependency(self) -> bool:
         if not self.cfg.get("logging", "rich_console_logging", False):
              logger.info("✓ Dependency check skipped: Rich (Console Logging disabled)")
              return True
         if not RICH_AVAILABLE:
             logger.critical("CRITICAL: Missing 'rich' for rich_console_logging. Install: uv add rich")
             return False
         logger.info("✓ Dependency check passed: Rich (for Console Logging)")
         return True

    def verify_requirements(self) -> None:
        logger.info("Running critical requirement checks...")
        failures: List[str] = []
        checks_to_run: List[Tuple[Callable[[], bool], str]] = [
            (self._check_python_version, "Python Version"),
            (lambda: self._check_import("yaml", "pyyaml", "Config parsing"), "PyYAML"),
            (lambda: self._check_import("customtkinter", purpose="GUI Toolkit"), "CustomTkinter"),
            (lambda: self._check_import("pyaudio", purpose="Audio I/O"), "PyAudio"),
            (lambda: self._check_import("numpy", purpose="Audio/Numeric Processing"), "NumPy"),
            (lambda: self._check_import("soundfile", purpose="Audio File I/O"), "SoundFile"),
            (self._check_rich_dependency, "Rich (Console Logging)"),
            (lambda: self._check_import("faster_whisper", package_name="faster-whisper", purpose="STT"), "Faster Whisper"),
            (lambda: self._check_import("openai", purpose="LLM Client"), "OpenAI Client"),
            (lambda: self._check_import("kokoro", purpose="TTS"), "Kokoro TTS"),
            (lambda: self._check_import("sentence_transformers", purpose="RAG Embeddings"), "Sentence Transformers"),
            (lambda: self._check_import("chromadb", purpose="RAG Vector Store"), "ChromaDB"),
            # Added Transformers and Tokenizers library checks
            (lambda: self._check_import("transformers", purpose="Universal Tokenizer"), "Transformers Library"),
            (lambda: self._check_import("tokenizers", purpose="Core Tokenizer for Transformers"), "Tokenizers (HF)"),
            (self._check_cuda_availability, "CUDA Availability"),
            (self._check_llm_endpoint, "LLM Endpoint Connectivity"),
            (self._check_context_store_writability, "Context Store Writability"),
        ]
        all_passed = True
        for check_func, check_name in checks_to_run:
            try:
                result = check_func()
                if result is False:
                    if not (check_name == "Rich (Console Logging)" and not self.cfg.get("logging", "rich_console_logging", False)):
                         failures.append(check_name)
                    all_passed = False
                elif result is not True:
                     logger.warning(f"Req check '{check_name}' returned non-boolean: {result}. Treating as fail.")
                     failures.append(f"{check_name} (Bad Return)"); all_passed = False
            except Exception as e:
                 logger.critical(f"CRITICAL: Error during req check '{check_name}': {e}", exc_info=True)
                 failures.append(f"{check_name} (Error)"); all_passed = False

        if not all_passed:
            failure_summary = ", ".join(failures)
            logger.critical("*** APPLICATION STARTUP BLOCKED ***")
            logger.critical(f"Failed requirement checks: {failure_summary}")
            logger.critical("Please resolve the issues and restart.")
            raise RequirementError(f"Failed checks: {failure_summary}")
        else:
            logger.info("✓ All critical system requirement checks passed.")