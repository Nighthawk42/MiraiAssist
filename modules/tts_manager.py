# ================================================
# FILE: modules/tts_manager.py
# ================================================

from __future__ import annotations  # For forward references

import warnings

# Suppress Kokoro repo_id notice
warnings.filterwarnings(
    "ignore",
    message="Defaulting repo_id to hexgrad/Kokoro-82M.*"
)
# Suppress that one RNN/dropout warning
warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout after all but last recurrent layer.*"
)
# Suppress the weight_norm deprecation
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*torch\\.nn\\.utils\\.weight_norm.*"
)

import threading
import time
import re
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional, Generator, Tuple, List, Union

import numpy as np

# Relative import for ConfigManager
from .config_manager import ConfigManager

# For type‐checking only (to avoid circular import)
if TYPE_CHECKING:
    import queue
    from .audio_manager import AudioManager

# Try to import the real Kokoro pipeline
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    # Dummy stub so imports elsewhere still work
    class KPipeline:
        def __init__(self, *args, **kwargs):
            logger.warning("Kokoro TTS dummy pipeline initialized (library not found).")
        def __call__(self, *args, **kwargs) -> Generator[Tuple[Any, Any, None], None, None]:
            logger.error("Cannot generate TTS: Kokoro library not installed.")
            if False:
                yield (None, None, None)

logger = logging.getLogger(__name__)

class TTSManagerError(Exception):
    """Raised for initialization or runtime failures in TTSManager."""
    pass

class TTSManager:
    """
    Manages text-to-speech using Kokoro’s KPipeline.

    - Sanitizes the text.
    - Streams audio chunks from Kokoro.
    - Concatenates them into a single NumPy array (or list).
    - Plays back via AudioManager.
    """

    DEFAULT_LANG_CODE = "a"        # American English
    DEFAULT_VOICE     = "af_heart" # One of Kokoro’s preset voices
    DEFAULT_SPEED     = 1.0

    def __init__(
        self,
        config: ConfigManager,
        gui_queue: 'queue.Queue[dict[str,Any]]',
        audio_manager: 'AudioManager'
    ):
        logger.info("Initializing TTSManager...")
        if not KOKORO_AVAILABLE:
            logger.critical("Kokoro TTS library is not installed. Run: pip install kokoro")
            raise TTSManagerError("Required library 'kokoro' not installed.")

        self.config = config
        self.gui_queue = gui_queue
        self.audio_manager = audio_manager

        # Load TTS settings from config
        tts_cfg = config.get_tts_config()
        self.lang_code = tts_cfg.get("lang_code", self.DEFAULT_LANG_CODE)
        self.voice     = tts_cfg.get("voice",     self.DEFAULT_VOICE)
        self.speed     = float(tts_cfg.get("speed", self.DEFAULT_SPEED))

        # Initialize the Kokoro pipeline
        try:
            logger.info(f"Creating Kokoro KPipeline (lang: {self.lang_code})...")
            start = time.time()
            self.pipeline = KPipeline(lang_code=self.lang_code)
            elapsed = time.time() - start
            logger.info(f"Kokoro pipeline ready in {elapsed:.2f}s.")
        except AssertionError as e:
            logger.critical(f"Invalid lang_code '{self.lang_code}'. Check config.", exc_info=False)
            raise TTSManagerError(f"Invalid TTS language code '{self.lang_code}'.") from e
        except Exception as e:
            logger.critical(f"Failed to init Kokoro pipeline: {e}", exc_info=True)
            raise TTSManagerError(f"Kokoro init failed: {e}") from e

        self._is_speaking_lock = threading.Lock()

    def _sanitize_text(self, text: str) -> str:
        """Strip markdown, emojis, and collapse whitespace."""
        no_md = re.sub(r"[*_~`#]", "", text or "")
        emoji_pat = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        no_emoji = emoji_pat.sub("", no_md)
        return re.sub(r"\s+", " ", no_emoji).strip()

    def speak_text(self, text: str):
        """
        Public API: generate and play TTS in a background thread.

        If called while already speaking, silently ignored.
        """
        clean = self._sanitize_text(text)
        if not clean:
            return
        if not self._is_speaking_lock.acquire(blocking=False):
            logger.warning("TTS already in progress; ignoring new request.")
            return

        logger.info("Starting TTS worker thread...")
        thread = threading.Thread(
            target=self._generate_and_play_audio,
            args=(clean,),
            daemon=True,
            name="TTSWorkerThread"
        )
        thread.start()

    def _generate_and_play_audio(self, clean_text: str):
        """
        Worker: stream Kokoro chunks, concatenate, and hand off to AudioManager.
        """
        try:
            all_chunks: List[np.ndarray] = []
            sample_rate: Optional[int] = None
            start = time.time()

            # Kokoro yields (gs, ps, audio), where `audio` may be a list or ndarray
            for idx, (gs, ps, audio_chunk) in enumerate(
                self.pipeline(clean_text, voice=self.voice, speed=self.speed)
            ):
                # Accept either list/tuple or ndarray
                if audio_chunk is None:
                    logger.debug(f"Skipping chunk #{idx}: no audio.")
                    continue

                # Coerce to ndarray
                arr = np.asarray(audio_chunk)
                if arr.size == 0:
                    logger.debug(f"Skipping chunk #{idx}: zero length.")
                    continue

                all_chunks.append(arr)
                if sample_rate is None:
                    # Use the pipeline’s reported sample rate (ps) if numeric, else default 24000
                    sample_rate = int(ps) if isinstance(ps, (int, float)) else 24000
                logger.debug(f"Appended chunk #{idx}: {arr.size} samples at {sample_rate} Hz.")

            gen_time = time.time() - start
            if not all_chunks or sample_rate is None:
                logger.warning("Kokoro TTS generated no valid audio data.")
                self.gui_queue.put({"type":"status","payload":"Ready (TTS No Audio)"})
                return

            # Concatenate into one array
            audio = np.concatenate(all_chunks)
            duration = len(audio) / sample_rate
            logger.info(f"TTS ready in {gen_time:.2f}s; total audio {duration:.2f}s.")

            # Play via AudioManager
            self.audio_manager.play_audio_data(audio, sample_rate)

        except Exception as e:
            logger.error(f"Error in TTS worker: {e}", exc_info=True)
            self.gui_queue.put({"type":"status","payload":"ERROR: TTS Failed"})
        finally:
            self._is_speaking_lock.release()

    def stop_speaking(self):
        """Request AudioManager to abort playback immediately."""
        if self.audio_manager and self.audio_manager.is_playing:
            logger.info("Stopping TTS playback.")
            self.audio_manager.stop_playback()
            if self._is_speaking_lock.locked():
                self._is_speaking_lock.release()
