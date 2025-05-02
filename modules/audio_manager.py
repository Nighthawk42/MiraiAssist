# ================================================
# FILE: modules/audio_manager.py
# ================================================

from __future__ import annotations

import logging
import queue
import tempfile
import threading
import time
import wave
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pyaudio

# Use relative import for ConfigManager
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

class AudioManagerError(Exception):
    """Custom exception for AudioManager specific errors."""
    pass

class AudioManager:
    """
    Handles audio input (recording via PTT) and output (playback).

    Uses PyAudio for cross-platform audio stream management. Can play
    audio from files or directly from NumPy arrays.
    Communicates status and results via a shared queue with the GUI.
    """

    def __init__(self, config: ConfigManager, gui_queue: queue.Queue):
        logger.info("Initializing AudioManager...")
        self.config = config
        self.gui_queue = gui_queue

        # Config shortcuts - use .get() for safety
        audio_cfg = config.get("audio", default={})
        self.sample_rate: int = int(audio_cfg.get("sample_rate", 16000))
        self.chunk_size: int   = int(audio_cfg.get("record_chunk_size", 1024))
        self.channels: int     = int(audio_cfg.get("channels", 1)) # Input channels
        self.format: int       = pyaudio.paInt16 # Input format
        self.format_np = np.int16 # Corresponding numpy type

        # State
        self._p: Optional[pyaudio.PyAudio] = None
        self._input_stream = None
        # Removed self._output_stream as streams are opened/closed per playback
        self._recording_thread: Optional[threading.Thread] = None
        self._playback_thread: Optional[threading.Thread] = None
        self._recording_frames: List[bytes] = []
        self._is_recording = False
        self._is_playing = False
        self._shutting_down = False
        self._stop_playback_event = threading.Event()

        try:
            self._p = pyaudio.PyAudio()
        except Exception as e:
            logger.critical(f"Failed to initialize PyAudio: {e}", exc_info=True)
            raise AudioManagerError(f"PyAudio initialization failed: {e}") from e

        # Select devices
        self.input_device_index = self._get_device_index("input_device", is_input=True)
        self.output_device_index = self._get_device_index("output_device", is_input=False)

        # Validate sample rate vs device default (optional, can cause issues)
        # self._validate_sample_rate() # Consider if needed, might override user pref

        self._log_device_info()
        logger.info("AudioManager initialized successfully.")

    def _validate_sample_rate(self):
        """If the requested sample_rate differs from the device default, log warning."""
        # This validation might be too strict, disabling for now. User config should be preferred.
        # Consider re-enabling if device compatibility becomes a major issue.
        # idx = self.input_device_index if self.input_device_index is not None else self.output_device_index
        # if idx is None or self._p is None:
        #     return
        # try:
        #     info = (self._p.get_device_info_by_index(idx))
        #     default_rate = int(info.get("defaultSampleRate", self.sample_rate))
        #     if self.sample_rate != default_rate:
        #         logger.warning(
        #             f"Requested sample_rate={self.sample_rate} differs from device default={default_rate}. "
        #             "Using requested rate, but check device compatibility if issues arise."
        #         )
        # except Exception as e:
        #     logger.debug(f"Could not validate sample rate against device default: {e}")
        pass


    def _get_device_index(self, key: str, is_input: bool) -> Optional[int]:
        if not self._p: return None
        audio_cfg = self.config.get("audio", default={})
        cfg_value = audio_cfg.get(key)
        device_type = "input" if is_input else "output"
        default_info_func = self._p.get_default_input_device_info if is_input else self._p.get_default_output_device_info
        max_channels_key = "maxInputChannels" if is_input else "maxOutputChannels"

        # 1) None or missing → Use default device
        if cfg_value is None:
            try:
                info = default_info_func()
                idx = info["index"]
                logger.info(f"Using default {device_type} device: {info['name']} (Index: {idx})")
                return idx
            except Exception as e:
                logger.warning(f"Could not get default {device_type} device: {e}. Trying first available.")
                return self._find_first_valid_device(is_input)

        # 2) Integer index
        if isinstance(cfg_value, int):
            try:
                info = self._p.get_device_info_by_index(cfg_value)
                channels = info.get(max_channels_key, 0)
                if channels > 0:
                    logger.info(f"Using configured {device_type} device index #{cfg_value}: {info['name']}")
                    return cfg_value
                else:
                     logger.warning(f"Configured {device_type} device index #{cfg_value} ({info['name']}) has 0 channels. Trying default.")
                     return self._get_device_index(key=key, is_input=is_input) # Retry logic with None
            except OSError as e:
                logger.warning(f"Invalid configured {device_type} device index #{cfg_value}: {e}. Trying default.")
                return self._get_device_index(key=None, is_input=is_input) # Retry logic with None

        # 3) Name substring (case-insensitive)
        if isinstance(cfg_value, str):
            name_lower = cfg_value.lower()
            for i in range(self._p.get_device_count()):
                try:
                    info = self._p.get_device_info_by_index(i)
                    dev_name = info.get("name", "").lower()
                    channels = info.get(max_channels_key, 0)
                    # Check if it's the correct type (input/output) and name matches
                    if channels > 0 and name_lower in dev_name:
                        logger.info(f"Matched configured {device_type} device name '{cfg_value}' to: {info['name']} (Index: {i})")
                        return i
                except OSError:
                    continue # Skip invalid device index
            logger.warning(f"Could not find {device_type} device matching name '{cfg_value}'. Trying default.")
            return self._get_device_index(key=None, is_input=is_input) # Retry logic with None

        # Fallback if cfg_value is invalid type
        logger.warning(f"Invalid type for audio config key '{key}': {type(cfg_value)}. Trying default.")
        return self._get_device_index(key=None, is_input=is_input)

    def _find_first_valid_device(self, is_input: bool) -> Optional[int]:
        if not self._p: return None
        max_channels_key = "maxInputChannels" if is_input else "maxOutputChannels"
        device_type = "input" if is_input else "output"
        for i in range(self._p.get_device_count()):
            try:
                info = self._p.get_device_info_by_index(i)
                channels = info.get(max_channels_key, 0)
                if channels > 0:
                    logger.info(f"Fallback: Found valid {device_type} device: {info['name']} (Index: {i})")
                    return i
            except OSError:
                continue # Skip invalid device index
        logger.error(f"Fatal: No valid {device_type} device found on the system.")
        self.gui_queue.put({"type": "status", "payload": f"ERROR: No {device_type.capitalize()} Device Found"})
        return None

    def _log_device_info(self):
        if not self._p: return
        try:
            if self.input_device_index is not None:
                info = self._p.get_device_info_by_index(self.input_device_index)
                logger.info(f"Selected Input Device [{self.input_device_index}]: {info['name']}, Rate: {info['defaultSampleRate']}, Max Ch: {info['maxInputChannels']}")
            else:
                logger.warning("No input device selected.")
            if self.output_device_index is not None:
                info = self._p.get_device_info_by_index(self.output_device_index)
                logger.info(f"Selected Output Device [{self.output_device_index}]: {info['name']}, Rate: {info['defaultSampleRate']}, Max Ch: {info['maxOutputChannels']}")
            else:
                logger.warning("No output device selected.")
        except Exception as e:
            logger.warning(f"Could not log full device info: {e}")

    # --- Recording ---

    def _recording_loop(self):
        stream = None
        if not self._p: return
        try:
            stream = self._p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.input_device_index
            )
            logger.info("Recording stream opened.")
            self._recording_frames.clear()
            while self._is_recording and not self._shutting_down:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self._recording_frames.append(data)
                except OSError as e:
                    # This can happen if the device is disconnected during recording
                    logger.error(f"Error reading from input stream: {e}", exc_info=True)
                    self.gui_queue.put({"type": "log", "payload": f"Mic read error: {e}", "tag": "error"})
                    self.gui_queue.put({"type": "status", "payload": "ERROR: Mic Disconnected?"})
                    self._is_recording = False # Stop the loop
                    break
            logger.info("Recording loop finished.")

        except Exception as e:
            logger.error(f"Recording error: {e}", exc_info=True)
            self.gui_queue.put({"type": "log", "payload": f"Recording failed: {e}", "tag": "error"})
            self.gui_queue.put({"type": "status", "payload": "ERROR: Recording Setup Failed"})
        finally:
            if stream:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
                    logger.info("Recording stream closed.")
                except Exception as e:
                    logger.warning(f"Error closing recording stream: {e}")
            self._is_recording = False # Ensure flag is false even on error


    def start_recording(self):
        if self._is_recording:
            logger.warning("Already recording.")
            return
        if self.input_device_index is None:
            logger.error("Cannot record: No input device selected.")
            self.gui_queue.put({"type": "status", "payload": "ERROR: No Mic Selected"})
            return
        if self._shutting_down:
            logger.warning("Cannot start recording, system is shutting down.")
            return

        self._is_recording = True
        self.gui_queue.put({"type": "status", "payload": "Recording..."})
        self._recording_thread = threading.Thread(
            target=self._recording_loop, daemon=True, name="RecordingThread"
        )
        self._recording_thread.start()

    def stop_recording(self):
        if not self._is_recording:
            # logger.debug("Stop recording called but not recording.")
            return
        logger.info("Stopping recording...")
        self._is_recording = False # Signal the loop to stop

        # Wait briefly for the recording thread to finish reading its last chunk
        if self._recording_thread and self._recording_thread.is_alive():
             self._recording_thread.join(timeout=0.5) # Shorter timeout
             if self._recording_thread.is_alive():
                  logger.warning("Recording thread did not stop gracefully.")

        # Check if we actually recorded anything
        if not self._recording_frames:
            logger.warning("No audio frames recorded.")
            self.gui_queue.put({"type": "status", "payload": "Ready (No audio recorded)"})
            return

        # Save the recorded data to a temporary WAV file
        audio_cfg = self.config.get("audio", default={})
        filename = audio_cfg.get("input_filename", "temp_input.wav")
        # Ensure temp files are in a standard location if possible
        temp_dir = Path(tempfile.gettempdir()) / "mirai_assist"
        temp_dir.mkdir(parents=True, exist_ok=True)
        out_path = temp_dir / f"{Path(filename).stem}_{int(time.time())}.wav"

        duration = len(self._recording_frames) * self.chunk_size / self.sample_rate
        logger.info(f"Recorded {duration:.2f} seconds of audio.")

        if self._p is None:
             logger.error("PyAudio not available, cannot save WAV.")
             self.gui_queue.put({"type": "status", "payload": "ERROR: PyAudio error"})
             return

        try:
            wf = wave.open(str(out_path), 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self._p.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self._recording_frames))
            wf.close()
            logger.info(f"Recorded audio saved to temporary file: {out_path}")
            self.gui_queue.put({
                "type": "audio_ready",
                "payload": {"filepath": str(out_path.resolve()), "duration": duration}
            })
            # Don't change status here, let the next step (STT) handle it
            # self.gui_queue.put({"type": "status", "payload": "Processing..."})
        except Exception as e:
            logger.error(f"Failed to save temporary WAV file: {e}", exc_info=True)
            self.gui_queue.put({"type": "log", "payload": f"Failed to save audio: {e}", "tag": "error"})
            self.gui_queue.put({"type": "status", "payload": "ERROR: Save WAV Failed"})
        finally:
            # Clear frames regardless of success/failure
            self._recording_frames.clear()
            logger.debug("Recording frames buffer cleared.")

    # --- Playback (File - Kept for potential other uses) ---

    def _playback_loop(self, filepath: str, cleanup: bool):
        """Plays audio from a WAV file."""
        stream = None
        wf = None
        if not self._p: return

        try:
            wf = wave.open(filepath, 'rb')
            playback_format = self._p.get_format_from_width(wf.getsampwidth())
            playback_channels = wf.getnchannels()
            playback_rate = wf.getframerate()

            logger.info(f"Opening playback stream for file: {filepath} ({playback_channels}ch, {playback_rate}Hz)")
            stream = self._p.open(
                format=playback_format,
                channels=playback_channels,
                rate=playback_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                output_device_index=self.output_device_index
            )

            data = wf.readframes(self.chunk_size)
            while data and not self._stop_playback_event.is_set():
                stream.write(data)
                data = wf.readframes(self.chunk_size)

            # Wait for stream to finish playing buffered data
            if not self._stop_playback_event.is_set():
                stream.stop_stream() # Waits until buffer is empty
            else:
                stream.abort() # Stops immediately if interrupted

            logger.info("Playback file stream finished.")

        except FileNotFoundError:
            logger.error(f"Playback error: File not found at {filepath}")
            self.gui_queue.put({"type": "status", "payload": "ERROR: Audio File Missing"})
        except Exception as e:
            logger.error(f"Playback file error: {e}", exc_info=True)
            self.gui_queue.put({"type": "log", "payload": f"Playback failed: {e}", "tag": "error"})
            self.gui_queue.put({"type": "status", "payload": "ERROR: Playback Failed"})
        finally:
            if stream:
                try:
                    if stream.is_active(): stream.stop_stream() # Ensure stopped
                    stream.close()
                    logger.info("Playback file stream closed.")
                except Exception as e:
                    logger.warning(f"Error closing playback file stream: {e}")
            if wf:
                wf.close()
            self._is_playing = False
            # self._stop_playback_event.clear() # Moved to stop_playback()
            # Don't reset status here - main loop should handle final status
            # self.gui_queue.put({"type": "status", "payload": "Ready"})
            if cleanup:
                try:
                    os.remove(filepath)
                    logger.debug(f"Deleted temporary playback file: {filepath}")
                except OSError as e:
                     logger.warning(f"Could not delete temp file {filepath}: {e}")

    def play_audio_file(self, filepath: str, cleanup: bool = False):
        """Starts playback of an audio file in a background thread."""
        if self._is_playing:
            logger.warning("Already playing audio, stopping previous playback first.")
            self.stop_playback()
            time.sleep(0.1) # Short pause to allow resources to release

        if not Path(filepath).exists():
            logger.error(f"Cannot play file: Not found at {filepath}")
            self.gui_queue.put({"type": "status", "payload": "ERROR: File Not Found"})
            return
        if self._shutting_down:
            logger.warning("Cannot start playback, system is shutting down.")
            return

        self._is_playing = True
        self._stop_playback_event.clear() # Ensure event is clear before starting
        # Status is usually set by the caller (e.g., "Speaking..." or "Playing...")
        # self.gui_queue.put({"type": "status", "payload": "Playing Audio..."})
        self._playback_thread = threading.Thread(
            target=self._playback_loop, args=(filepath, cleanup),
            daemon=True, name="PlaybackFileThread"
        )
        self._playback_thread.start()

    # --- Playback (Data - Preferred for TTS) ---

    def _playback_data_loop(self, audio_data: np.ndarray, sample_rate: int, pyaudio_format: int, num_channels: int):
        """Plays audio directly from a NumPy array."""
        stream = None
        if not self._p: return

        try:
            logger.info(f"Opening playback stream for data ({num_channels}ch, {sample_rate}Hz, Format: {pyaudio_format})")
            stream = self._p.open(
                format=pyaudio_format,
                channels=num_channels,
                rate=sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                output_device_index=self.output_device_index
            )

            # Convert numpy array to bytes efficiently
            audio_bytes = audio_data.tobytes()
            total_bytes = len(audio_bytes)
            sample_width = audio_data.dtype.itemsize
            bytes_per_frame = num_channels * sample_width
            bytes_per_buffer = self.chunk_size * bytes_per_frame
            current_pos = 0

            logger.debug(f"Starting data playback: {total_bytes} bytes")

            while current_pos < total_bytes and not self._stop_playback_event.is_set():
                end_pos = min(current_pos + bytes_per_buffer, total_bytes)
                chunk = audio_bytes[current_pos:end_pos]
                if not chunk: break # Should not happen with min() but safety first
                stream.write(chunk)
                current_pos = end_pos

            # Wait for stream to finish playing buffered data
            if not self._stop_playback_event.is_set():
                 stream.stop_stream() # Waits until buffer is empty
            else:
                 stream.abort() # Stops immediately if interrupted

            logger.info("Playback data stream finished.")

        except Exception as e:
            logger.error(f"Playback data error: {e}", exc_info=True)
            self.gui_queue.put({"type": "log", "payload": f"Playback failed: {e}", "tag": "error"})
            self.gui_queue.put({"type": "status", "payload": "ERROR: Playback Failed"})
        finally:
            if stream:
                try:
                    if stream.is_active(): stream.stop_stream() # Ensure stopped
                    stream.close()
                    logger.info("Playback data stream closed.")
                except Exception as e:
                    logger.warning(f"Error closing playback data stream: {e}")
            self._is_playing = False
            # self._stop_playback_event.clear() # Moved to stop_playback()
            # Don't reset status here - main loop should handle final status
            # self.gui_queue.put({"type": "tts_finished"}) # Idea for future enhancement
            self.gui_queue.put({"type": "status", "payload": "Ready"}) # TEMPORARY: Set ready after playback

    def play_audio_data(self, audio_data: np.ndarray, sample_rate: int):
        """Starts playback of NumPy audio data in a background thread."""
        if self._is_playing:
            logger.warning("Already playing audio, stopping previous playback first.")
            self.stop_playback()
            time.sleep(0.1) # Short pause

        if self.output_device_index is None:
             logger.error("Cannot play audio data: No output device selected.")
             self.gui_queue.put({"type": "status", "payload": "ERROR: No Speaker Selected"})
             return
        if self._shutting_down:
            logger.warning("Cannot start playback, system is shutting down.")
            return
        if not self._p:
             logger.error("Cannot play audio data: PyAudio not initialized.")
             return

        # Determine PyAudio format and channels from numpy array
        try:
            dtype_map = {
                np.dtype('int16'): pyaudio.paInt16,
                np.dtype('int32'): pyaudio.paInt32,
                np.dtype('float32'): pyaudio.paFloat32,
                np.dtype('uint8'): pyaudio.paUInt8,
            }
            pyaudio_format = dtype_map[audio_data.dtype]
            # sample_width = audio_data.dtype.itemsize # Calculated in loop
        except KeyError:
            logger.error(f"Unsupported NumPy dtype for playback: {audio_data.dtype}")
            self.gui_queue.put({"type": "log", "payload": f"Unsupported audio format: {audio_data.dtype}", "tag": "error"})
            self.gui_queue.put({"type": "status", "payload": "ERROR: Unsupported Audio Format"})
            return

        num_channels = 1
        if audio_data.ndim > 1:
            num_channels = audio_data.shape[1]
        elif audio_data.ndim == 0:
             logger.error("Cannot play audio data: Input array is zero-dimensional.")
             self.gui_queue.put({"type": "status", "payload": "ERROR: Invalid Audio Data"})
             return

        self._is_playing = True
        self._stop_playback_event.clear() # Ensure event is clear before starting
        # Status should be set by the caller (usually "Speaking...")
        # self.gui_queue.put({"type": "status", "payload": "Speaking..."})
        self._playback_thread = threading.Thread(
            target=self._playback_data_loop,
            args=(audio_data, sample_rate, pyaudio_format, num_channels),
            daemon=True, name="PlaybackDataThread"
        )
        self._playback_thread.start()

    def stop_playback(self):
        """Signals any active playback thread to stop."""
        if not self._is_playing:
            # logger.debug("Stop playback called but not playing.")
            return
        logger.info("Signalling playback to stop...")
        self._stop_playback_event.set() # Signal stop

        # Wait briefly for the thread to acknowledge the stop signal
        if self._playback_thread and self._playback_thread.is_alive():
             self._playback_thread.join(timeout=0.2) # Short timeout just to wait for acknowledgement
             if self._playback_thread.is_alive():
                  logger.warning("Playback thread did not stop quickly after signal.")
                  # The thread should eventually terminate when its loop condition checks the event

        # Reset state - the loop itself sets _is_playing to False on exit
        # self._is_playing = False # Let the loop handle this
        self._stop_playback_event.clear() # Reset event for next playback
        # Don't set status here, let the GUI decide based on context
        # self.gui_queue.put({"type": "status", "payload": "Stopped"})
        logger.info("Stop playback signal sent.")

    # --- Shutdown ---

    def stop(self):
        """Stops all audio activities and terminates PyAudio."""
        logger.info("Stopping AudioManager...")
        self._shutting_down = True

        # Signal recording to stop (if active)
        if self._is_recording:
            logger.debug("Stopping active recording during shutdown...")
            self.stop_recording() # This handles thread joining

        # Signal playback to stop (if active)
        if self._is_playing:
            logger.debug("Stopping active playback during shutdown...")
            self.stop_playback() # This handles thread joining

        # Terminate PyAudio instance
        if self._p:
            try:
                logger.debug("Terminating PyAudio instance...")
                self._p.terminate()
                logger.info("PyAudio terminated.")
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
        self._p = None # Ensure reference is cleared

        logger.info("AudioManager stopped.")

    # --- Properties ---
    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def is_playing(self) -> bool:
        return self._is_playing