#!/usr/bin/env python3
"""Optional voice input (VOSK), used by architecture C only.

This is a human-machine-interface layer, not part of the architecture
comparison: speech recognition errors would otherwise be mixed into the
success rate of the navigation experiment. It is therefore kept in its own
module, disabled by default, and imported only by ``c_client.py``.

State machine: WAKE -> COMMAND -> WAKE.
Only what is spoken after the wake word is delivered to the agent, and the
listener keeps running while a skill is active, so a new spoken command can
interrupt the robot (barge-in).

Dependencies (``vosk``, ``sounddevice``) are imported lazily: if they are
missing, the client keeps working with the keyboard and reports the reason.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import threading
import time
from typing import Any, Optional

VOICE_ENABLED = os.getenv("VOICE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "")
SAMPLE_RATE = int(os.getenv("VOSK_SAMPLE_RATE", "16000"))
BLOCKSIZE = int(os.getenv("VOSK_BLOCKSIZE", "2048"))
WAKE_WORD = os.getenv("WAKE_WORD", "hi robot").strip().lower()
COMMAND_TIMEOUT_SEC = float(os.getenv("VOICE_COMMAND_TIMEOUT_SEC", "8"))


class VoiceInput:
    """Wake-word listener that pushes recognised commands into an asyncio queue."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.commands: asyncio.Queue[str] = asyncio.Queue()
        self.audio: queue.Queue[bytes] = queue.Queue(maxsize=100)
        self.ready = False
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if not VOICE_ENABLED:
            self.error = "voice input disabled (VOICE_ENABLED=false)"
            return
        if not VOSK_MODEL_PATH or not os.path.isdir(VOSK_MODEL_PATH):
            self.error = f"VOSK model not found: {VOSK_MODEL_PATH or '(unset)'}"
            return
        try:
            import sounddevice  # noqa: F401
            import vosk  # noqa: F401
        except Exception as exc:
            self.error = f"voice dependencies missing: {exc}"
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        return self.error is None

    def _audio_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        try:
            self.audio.put_nowait(bytes(indata))
        except queue.Full:
            try:                       # drop the oldest block, keep the newest
                self.audio.get_nowait()
                self.audio.put_nowait(bytes(indata))
            except queue.Empty:
                pass

    @staticmethod
    def _text(result_json: str) -> str:
        try:
            text = json.loads(result_json).get("text", "")
        except Exception:
            return ""
        return re.sub(r"\s+", " ", str(text).lower()).strip()

    def _emit(self, command: str) -> None:
        self.loop.call_soon_threadsafe(self.commands.put_nowait, command)

    def _run(self) -> None:
        import sounddevice as sd
        from vosk import KaldiRecognizer, Model

        try:
            model = Model(VOSK_MODEL_PATH)
            wake = KaldiRecognizer(model, SAMPLE_RATE, json.dumps([WAKE_WORD]))
            command = KaldiRecognizer(model, SAMPLE_RATE)
            state, entered_command = "WAKE", 0.0

            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE,
                                   dtype="int16", channels=1,
                                   callback=self._audio_callback):
                self.ready = True
                print(f"Voice ready | say: {WAKE_WORD}", flush=True)
                while not self._stop.is_set():
                    try:
                        data = self.audio.get(timeout=0.1)
                    except queue.Empty:
                        # Do not stay in COMMAND forever if nothing is spoken.
                        if state == "COMMAND" and time.monotonic() - entered_command > COMMAND_TIMEOUT_SEC:
                            print(f"No command heard. Say '{WAKE_WORD}' again.", flush=True)
                            command.Reset()
                            wake.Reset()
                            state = "WAKE"
                        continue

                    if state == "WAKE":
                        if wake.AcceptWaveform(data) and self._text(wake.Result()) == WAKE_WORD:
                            print("Wake word detected. Speak the command.", flush=True)
                            wake.Reset()
                            command.Reset()
                            state, entered_command = "COMMAND", time.monotonic()
                        continue

                    if time.monotonic() - entered_command > COMMAND_TIMEOUT_SEC:
                        print(f"No command heard. Say '{WAKE_WORD}' again.", flush=True)
                        command.Reset()
                        wake.Reset()
                        state = "WAKE"
                        continue

                    if command.AcceptWaveform(data):
                        spoken = self._text(command.Result())
                        command.Reset()
                        wake.Reset()
                        state = "WAKE"
                        if spoken:
                            print(f"Voice command: {spoken}", flush=True)
                            self._emit(spoken)
                        else:
                            print(f"Nothing understood. Say '{WAKE_WORD}' again.", flush=True)
        except Exception as exc:
            self.error = str(exc)
            print(f"Voice input failed: {exc}", flush=True)

    async def read(self) -> str:
        """Wait for the next spoken command. Never returns if voice is off."""
        if not self.available:
            await asyncio.Event().wait()      # park forever; keyboard still works
        return await self.commands.get()

    def close(self) -> None:
        self._stop.set()
