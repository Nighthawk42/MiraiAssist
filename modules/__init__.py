# modules/__init__.py

# This file marks the 'modules' directory as a Python package.

# For easier imports if needed, e.g., from modules import ConfigManager
from .audio_manager import AudioManager, AudioManagerError
from .config_manager import ConfigManager, ConfigError
from .context_manager import ContextManager, ContextManagerError
from .llm_manager import LLMManager, LLMManagerError
from .memory_manager import MemoryManager, MemoryManagerError # ADDED MemoryManager
from .stt_manager import STTManager, STTManagerError
from .system_manager import SystemManager, LoggingSetupError, RequirementError
from .tts_manager import TTSManager, TTSManagerError
from .ui_manager import UIManager

__all__ = [
    "AudioManager", "AudioManagerError",
    "ConfigManager", "ConfigError",
    "ContextManager", "ContextManagerError",
    "LLMManager", "LLMManagerError",
    "MemoryManager", "MemoryManagerError", # ADDED MemoryManager
    "STTManager", "STTManagerError",
    "SystemManager", "LoggingSetupError", "RequirementError",
    "TTSManager", "TTSManagerError",
    "UIManager",
]