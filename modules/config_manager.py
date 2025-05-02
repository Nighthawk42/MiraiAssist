# ================================================
# FILE: modules/config_manager.py
# ================================================

import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Type, Union
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")

class ConfigError(Exception):
    """Raised when the configuration file cannot be loaded / parsed."""


class ConfigManager:
    """
    Singleton that loads `config.yaml` and provides access to settings.

    Usage:
    ------
    >>> cfg = ConfigManager()      # Get the singleton instance
    >>> cfg.load()                 # Load once at startup (optional path)
    >>> sr = cfg.get("audio", "sample_rate", default=16000)
    >>> audio_settings = cfg.get("audio", default={}) # Get whole section
    """

    # Singleton instance storage
    _instance: Optional["ConfigManager"] = None

    def __new__(cls, config_path: Optional[Union[str, Path]] = None) -> "ConfigManager":
        """Enforces the singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize attributes only once
            cls._instance._config_path = Path(config_path or "config.yaml").resolve()
            cls._instance._data = {}
            cls._instance._loaded = False
            logger.debug(f"ConfigManager singleton created. Path set to: {cls._instance._config_path}")
        elif config_path is not None:
            # Allow updating path if explicitly provided *after* creation, but before load
            # Though typically path is set once at creation or implicitly defaults.
            if not cls._instance._loaded:
                 new_path = Path(config_path).resolve()
                 if new_path != cls._instance._config_path:
                      logger.warning(f"ConfigManager path changed after instantiation to: {new_path}")
                      cls._instance._config_path = new_path
            else:
                 logger.warning("ConfigManager path cannot be changed after config is loaded.")

        return cls._instance

    # Public API ------------------------------------------------------------ #
    def load(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Reads and parses the YAML configuration file.

        Can be called multiple times, but only loads the file once unless
        `force_reload=True`.

        Args:
            config_path: Optional path to the config file. If provided,
                         it updates the path stored in the instance before loading.
        """
        if config_path:
            new_path = Path(config_path).resolve()
            if new_path != self._config_path:
                 if self._loaded:
                      logger.warning("Cannot change config path after initial load without force_reload.")
                      return
                 self._config_path = new_path
                 logger.info(f"Config file path updated to: {self._config_path}")

        if self._loaded:
            logger.debug("Configuration already loaded. Skipping.")
            return

        if not self._config_path.exists():
            msg = f"Configuration file not found: {self._config_path}"
            # Log critical error, GUI should handle shutdown based on exception
            logger.critical(msg)
            raise ConfigError(msg)

        try:
            logger.info(f"Loading configuration from: {self._config_path}")
            with self._config_path.open('r', encoding="utf-8") as fh:
                loaded_data = yaml.safe_load(fh)

            if loaded_data is None:
                logger.warning(f"Configuration file is empty: {self._config_path}")
                self._data = {}
            elif not isinstance(loaded_data, dict):
                msg = f"Configuration file root must be a dictionary (mapping), found {type(loaded_data)}."
                logger.critical(msg)
                raise ConfigError(msg)
            else:
                self._data = loaded_data

            self._loaded = True
            logger.info("Configuration loaded successfully.")

        except yaml.YAMLError as exc:
            msg = f"Error parsing YAML configuration file '{self._config_path}': {exc}"
            logger.critical(msg, exc_info=True)
            raise ConfigError(msg) from exc
        except IOError as exc:
            msg = f"Error reading configuration file '{self._config_path}': {exc}"
            logger.critical(msg, exc_info=True)
            raise ConfigError(msg) from exc
        except Exception as exc:
            msg = f"An unexpected error occurred while loading config '{self._config_path}': {exc}"
            logger.critical(msg, exc_info=True)
            raise ConfigError(msg) from exc

    def get(self, section: str, key: Optional[str] = None, default: Optional[T] = None) -> Any | T:
        """
        Retrieves a configuration value.

        Args:
            section: The top-level section key (e.g., "audio", "llm").
            key: The specific key within the section. If None, returns the
                 entire section dictionary.
            default: The value to return if the section or key is not found.

        Returns:
            The configuration value, the section dictionary, or the default.
        """
        if not self._loaded:
            # Allow access before load, but return default and log warning
            logger.warning("Configuration accessed before 'load()' was called. Returning default.")
            # If key is None, default should ideally be a dict or compatible type
            return default if key is not None else (default if default is not None else {})


        section_data = self._data.get(section)

        if section_data is None:
            # logger.debug(f"Config section '{section}' not found, returning default.")
            return default if key is not None else (default if default is not None else {})

        if key is None:
            # Return the entire section if key is not specified
            return section_data if isinstance(section_data, dict) else (default if default is not None else {})


        # Return the specific key value or default if key is not found within the section
        return section_data.get(key, default)

    def save(self) -> None:
        """Persists the current in-memory configuration back to the YAML file."""
        if not self._loaded:
            logger.error("Cannot save configuration - it was never loaded successfully.")
            return
        if not self._data:
             logger.warning("Configuration data is empty. Saving an empty file.")

        try:
            logger.info(f"Saving configuration to: {self._config_path}")
            # Ensure parent directory exists
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with self._config_path.open("w", encoding="utf-8") as fh:
                yaml.dump(self._data, fh, default_flow_style=False, sort_keys=False, indent=2, allow_unicode=True)
            logger.info("Configuration saved successfully.")
        except IOError as exc:
            logger.error(f"Could not save configuration file '{self._config_path}': {exc}", exc_info=True)
        except Exception as exc:
            logger.error(f"An unexpected error occurred while saving config: {exc}", exc_info=True)

    def update_value(self, section: str, key: str, value: Any) -> None:
        """
        Updates a specific configuration value in memory.

        Call `save()` afterwards to persist the change to the file.

        Args:
            section: The top-level section key.
            key: The specific key within the section.
            value: The new value to set.
        """
        if not self._loaded:
            logger.error("Cannot update value - configuration not loaded.")
            return

        if section not in self._data or not isinstance(self._data[section], dict):
            logger.debug(f"Creating new section '{section}' in config data.")
            self._data[section] = {}

        logger.debug(f"Updating config: [{section}][{key}] = {value}")
        self._data[section][key] = value

    # --- Convenience section getters (Optional) ---
    # These can be useful but also hide where config comes from.
    # Use cfg.get("audio", default={}) instead if preferred.

    def get_audio_config(self) -> Dict[str, Any]:
        """Returns the 'audio' configuration section."""
        return self.get("audio", default={})

    def get_llm_config(self) -> Dict[str, Any]:
        """Returns the 'llm' configuration section."""
        return self.get("llm", default={})

    def get_stt_config(self) -> Dict[str, Any]:
        """Returns the 'stt' configuration section."""
        return self.get("stt", default={})

    def get_tts_config(self) -> Dict[str, Any]:
        """Returns the 'tts' configuration section."""
        return self.get("tts", default={})

    def get_logging_config(self) -> Dict[str, Any]:
        """Returns the 'logging' configuration section."""
        return self.get("logging", default={})

    # --- Properties ---
    @property
    def is_loaded(self) -> bool:
        """Returns True if the configuration has been loaded, False otherwise."""
        return self._loaded

    @property
    def config_path(self) -> Path:
        """Returns the resolved path to the configuration file."""
        return self._config_path

    @property
    def data(self) -> Dict[str, Any]:
        """Returns a copy of the entire configuration data dictionary."""
        return self._data.copy() # Return a copy to prevent direct modification