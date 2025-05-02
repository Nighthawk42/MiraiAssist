# ================================================
# FILE: modules/stt_manager.py
# ================================================

import logging
from pathlib import Path
import threading
import time
from typing import Optional, Dict, Any, Tuple, List

try:
    # Attempt to import the core library
    from faster_whisper import WhisperModel
    # Attempt to import optional VAD dependency if filtering is enabled often
    # This helps catch missing deps early, though WhisperModel might load VAD lazily
    # from whisper_vad import VoiceActivityDetector # Example, check faster-whisper's actual VAD mechanism if used
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    # Define dummy class if library not installed for type hinting and basic checks
    class WhisperModel:
        def __init__(self, *args, **kwargs):
            logger.warning("Faster Whisper dummy model initialized (library not found).")
        def transcribe(self, *args, **kwargs) -> Tuple[List, Dict[str, Any]]:
            logger.error("Cannot transcribe: Faster Whisper library not installed.")
            return ([], {"error": "Faster Whisper not installed"})

# Use relative import for ConfigManager
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

# Custom Exception
class STTManagerError(Exception):
    """Custom exception for STTManager errors, particularly during initialization."""
    pass

class STTManager:
    """
    Handles Speech-to-Text (STT) transcription using the Faster Whisper library.

    Loads a specified Whisper model and provides a method to transcribe audio files.
    Handles model loading errors and transcription processing state.
    """

    # Default configuration values
    DEFAULT_MODEL_SIZE = "base.en" # Smaller, faster, English-only default
    DEFAULT_DEVICE = "cpu"
    DEFAULT_COMPUTE_TYPE_CPU = "int8" # Good balance for CPU
    DEFAULT_COMPUTE_TYPE_CUDA = "float16" # Good balance for CUDA
    DEFAULT_VAD_FILTER = True
    DEFAULT_BEAM_SIZE = 5

    def __init__(self, config: ConfigManager):
        """
        Initializes the STTManager and loads the Faster Whisper model based on config.

        Args:
            config: The application's ConfigManager instance.

        Raises:
            STTManagerError: If Faster Whisper is not available or model loading fails.
        """
        logger.info("Initializing STTManager...")
        if not FASTER_WHISPER_AVAILABLE:
            # Log the error clearly, SystemManager should ideally catch this earlier
            logger.critical("Faster Whisper library is not installed. Please run: uv add faster-whisper[all]") # or specific backend
            raise STTManagerError("Required library 'faster-whisper' not installed.")

        self.config = config
        # Use convenience getter with default {}
        self.stt_config = config.get_stt_config()

        # --- Model configuration ---
        self.model_size: str = self.stt_config.get("model_size", self.DEFAULT_MODEL_SIZE)
        self.device: str = self.stt_config.get("device", self.DEFAULT_DEVICE).lower() # Ensure lowercase

        # Determine compute type based on device
        if self.device == "cuda":
            default_compute = self.DEFAULT_COMPUTE_TYPE_CUDA
        else: # cpu or other devices
            default_compute = self.DEFAULT_COMPUTE_TYPE_CPU
        self.compute_type: str = self.stt_config.get("compute_type", default_compute)

        # --- VAD (Voice Activity Detection) configuration ---
        self.vad_filter: bool = self.stt_config.get("vad_filter", self.DEFAULT_VAD_FILTER)
        _vad_params_config = self.stt_config.get("vad_parameters")
        # Ensure vad_parameters is a dictionary, default to empty if not specified or None
        if _vad_params_config is None:
             self.vad_parameters: Dict[str, Any] = {} # Default empty if key exists but is None
        elif isinstance(_vad_params_config, dict):
             self.vad_parameters: Dict[str, Any] = _vad_params_config
        else:
             logger.warning(f"Invalid 'vad_parameters' type in config (expected dict, got {type(_vad_params_config)}). Using default VAD parameters.")
             self.vad_parameters = {} # Default empty on invalid type


        # --- Transcription options ---
        self.beam_size: int = int(self.stt_config.get("beam_size", self.DEFAULT_BEAM_SIZE))
        # Add other options from config as needed (e.g., language, initial_prompt)
        self.language: Optional[str] = self.stt_config.get("language") # e.g., "en", None for auto-detect
        self.initial_prompt: Optional[str] = self.stt_config.get("initial_prompt")


        # --- State ---
        self.model: Optional[WhisperModel] = None
        self._is_processing_lock = threading.Lock()
        self._is_processing: bool = False

        # --- Load Model ---
        # This can take time, so it's done during init
        try:
            self._load_model()
        except Exception as e:
             # Wrap loading errors in STTManagerError for clarity at startup
             raise STTManagerError(f"Failed to initialize STTManager: {e}") from e

        logger.info("STTManager initialized successfully.")

    def _load_model(self):
        """Loads the configured Faster Whisper model."""
        logger.info(f"Loading Faster Whisper model: '{self.model_size}' (Device: {self.device}, Compute: {self.compute_type})")
        # Log VAD settings
        if self.vad_filter:
            logger.info(f"VAD Filter Enabled. Parameters: {self.vad_parameters or 'default'}")
        else:
            logger.info("VAD Filter Disabled.")

        try:
            start_time = time.time()
            # Note: Faster Whisper might download the model here if not cached
            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                # Pass other relevant init options if available, e.g., cpu_threads, num_workers
                # cpu_threads=self.stt_config.get("cpu_threads", 4),
                # num_workers=self.stt_config.get("num_workers", 1)
            )
            load_time = time.time() - start_time
            logger.info(f"Whisper model '{self.model_size}' loaded successfully in {load_time:.2f} seconds.")

        except Exception as e:
            logger.critical(f"Failed to load Whisper model '{self.model_size}' with config (device={self.device}, compute={self.compute_type}): {e}", exc_info=True)
            # Provide more specific guidance based on common errors
            error_str = str(e).lower()
            if "cuda" in error_str or "cublas" in error_str or "cudnn" in error_str:
                 error_msg = (f"Failed to load STT model on GPU. Ensure CUDA toolkit and cuDNN are installed correctly, "
                              f"compatible with your PyTorch version, and meet faster-whisper requirements for "
                              f"device='{self.device}' / compute='{self.compute_type}'. Error: {e}")
            elif "not found" in error_str and self.model_size in error_str:
                error_msg = (f"Failed to load STT model: Model size '{self.model_size}' could not be found or downloaded. "
                             f"Check the model name spelling and your internet connection. "
                             f"Ensure model cache directory is writable. Error: {e}")
            elif "download" in error_str:
                 error_msg = f"Error occurred during model download for '{self.model_size}'. Check network connection and permissions. Error: {e}"
            else:
                error_msg = f"An unexpected error occurred while loading STT model '{self.model_size}'. Error: {e}"
            # Raise the wrapped error
            raise STTManagerError(error_msg) from e


    def transcribe(self, audio_path: str) -> Optional[str]:
        """
        Transcribes the audio file at the given path to text.

        Args:
            audio_path: The path to the audio file (e.g., WAV, MP3).

        Returns:
            The transcribed text as a string if successful.
            Returns an empty string ("") if transcription results in no text (e.g., silence).
            Returns None if an error occurs during transcription or if the model
            is unavailable or busy.
        """
        # Check processing state with lock
        if not self._is_processing_lock.acquire(blocking=False):
            logger.warning("Transcription requested, but STTManager is already processing another file. Request ignored.")
            return None # Indicate busy

        # Ensure model is loaded
        if not self.model:
            logger.error("Cannot transcribe: Whisper model is not loaded.")
            self._is_processing_lock.release()
            return None # Indicate error

        # Validate audio file path
        audio_file = Path(audio_path)
        if not audio_file.exists():
            logger.error(f"Cannot transcribe: Audio file not found at '{audio_path}'")
            self._is_processing_lock.release()
            return None # Indicate error
        if not audio_file.is_file():
            logger.error(f"Cannot transcribe: Path '{audio_path}' is not a file.")
            self._is_processing_lock.release()
            return None # Indicate error


        self._is_processing = True # Set flag after acquiring lock
        logger.info(f"Starting transcription for: {audio_path}")
        start_time = time.time()
        transcribed_text: Optional[str] = None # Default to None for errors

        try:
            # Prepare transcription options
            transcribe_options = {
                "beam_size": self.beam_size,
                "vad_filter": self.vad_filter,
                "vad_parameters": self.vad_parameters,
                "language": self.language, # Can be None for auto-detect
                "initial_prompt": self.initial_prompt,
                # Add other options here as needed: temperature, word_timestamps, etc.
                # "word_timestamps": self.stt_config.get("word_timestamps", False),
            }
            # Filter out None values from options
            transcribe_options = {k: v for k, v in transcribe_options.items() if v is not None}


            # Perform transcription - Faster Whisper handles file reading
            # The result is an iterator of Segment objects and an Info object
            segments_iterator, info = self.model.transcribe(
                str(audio_file), # Pass path as string
                **transcribe_options
            )

            # Process the segments iterator to build the full text
            # This is where the actual computation happens for the iterator
            segment_texts = [segment.text for segment in segments_iterator]
            full_text = " ".join(segment_texts).strip()

            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Transcription finished in {duration:.2f} seconds.")

            # Log detected language info if available
            if info:
                 detected_lang = getattr(info, 'language', 'N/A')
                 lang_prob = getattr(info, 'language_probability', 'N/A')
                 logger.info(f"Detected language: {detected_lang} (Probability: {lang_prob:.2f})" if isinstance(lang_prob, float) else f"Detected language: {detected_lang}")
                 # Log VAD duration if filter was enabled
                 if self.vad_filter:
                      vad_duration = getattr(info, 'duration_after_vad', None)
                      if vad_duration is not None:
                           logger.info(f"Audio duration after VAD: {vad_duration:.2f}s")


            # Handle empty transcription result (silence or no speech detected)
            if not full_text:
                logger.warning("Transcription resulted in empty text (possibly silence or no speech detected).")
                transcribed_text = "" # Return empty string, not None
            else:
                # Log a snippet of the result for confirmation
                logger.info(f"Transcription result: '{full_text[:80]}...'")
                transcribed_text = full_text

        except Exception as e:
            # Catch any unexpected errors during the transcription process
            logger.error(f"Error during transcription for '{audio_path}': {e}", exc_info=True)
            transcribed_text = None # Explicitly set to None on error
        finally:
            # Release the lock and reset the processing flag
            self._is_processing = False
            self._is_processing_lock.release()
            logger.debug("STT processing finished and lock released.")

        return transcribed_text

    @property
    def is_processing(self) -> bool:
         """Returns True if the manager is currently transcribing."""
         # Check the internal flag (lock status is harder to check reliably from outside)
         return self._is_processing