"""Azure Speech Repository - Azure Speech SDK wrapper for data access."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import wave
from typing import AsyncGenerator, Dict, Any, Iterable, Sequence, AsyncIterable

import azure.cognitiveservices.speech as speechsdk

from core.exceptions import AzureSpeechError

logger = logging.getLogger(__name__)


DEFAULT_PHRASE_HINTS: Sequence[str] = (
    "capstone",
    "microservices",
    "micro service",
    "three layer",
    "3-layer",
    "web service",
    "rest api",
    "container",
    "kubernetes",
    "docker",
    "pipeline",
    "continuous integration",
    "continuous delivery",
    "domain driven design",
    "viết luận",
    "bảo vệ khóa luận",
    "cấu trúc dữ liệu",
    "trí tuệ nhân tạo",
    "máy học",
    "cloud",
    "azure",
    "machine learning",
    "deep learning",
    "model serving",
    "inference",
    "event driven",
    "message queue",
    "database schema",
    "three tier",
)


def _pcm_to_wav(pcm: bytes, sample_rate: int = 16000, sample_width: int = 2) -> bytes:
    """Convert raw PCM to WAV format."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buffer.getvalue()


class AzureSpeechRepository:
    """Repository for Azure Speech Service operations."""
    
    def __init__(
        self,
        subscription_key: str,
        region: str,
        language: str = "vi-VN",
        default_phrase_hints: Sequence[str] | None = None,
        sample_rate: int = 16000,
    ) -> None:
        """
        Initialize Azure Speech repository.
        
        Args:
            subscription_key: Azure Speech subscription key
            region: Azure region
            language: Speech recognition language
            default_phrase_hints: Default phrase hints for better recognition
            sample_rate: Audio sample rate (Hz)
        """
        self.sample_rate = sample_rate
        self.default_phrase_hints: Sequence[str] = (
            tuple(default_phrase_hints) if default_phrase_hints else DEFAULT_PHRASE_HINTS
        )
        
        # Speech config
        self.speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
        self.speech_config.speech_recognition_language = language
        self.speech_config.output_format = speechsdk.OutputFormat.Detailed
        
        # Segmentation & silence timeouts (optimized for Vietnamese)
        self.speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "800"
        )
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "5000"
        )
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "500"
        )
        
        # TrueText (capitalization + punctuation)
        try:
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption,
                "TrueText",
            )
        except AttributeError:
            # Fallback for older SDK
            self.speech_config.set_property_by_name(
                "SpeechServiceResponse_PostProcessingOption", "TrueText"
            )
        
        # Timestamps & stable partial results
        try:
            self.speech_config.request_word_level_timestamps()
        except Exception:
            pass
        
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceResponse_StablePartialResultThreshold, "3"
        )
        
        logger.info(
            f"Azure Speech Repository initialized | language={language} | region={region} | sr={sample_rate}Hz"
        )
    
    async def recognize_stream(
        self,
        audio_source: AsyncIterable[bytes],
        extra_phrases: Iterable[str] | None = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream speech recognition from an async audio source.
        
        Args:
            audio_source: Async iterable providing raw PCM audio chunks
            extra_phrases: Additional phrase hints for this session
        
        Yields:
            Recognition events as dictionaries
        
        Raises:
            AzureSpeechError: If recognition fails
        """
        # Setup audio format: 16kHz/16-bit/mono
        fmt = speechsdk.audio.AudioStreamFormat(
            samples_per_second=self.sample_rate,
            bits_per_sample=16,
            channels=1,
        )
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        
        # Create recognizer
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config,
        )
        
        # Apply phrase hints
        all_hints = list(self.default_phrase_hints)
        if extra_phrases:
            all_hints.extend(extra_phrases)
        
        if all_hints:
            phrase_list = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
            for phrase in all_hints:
                phrase_list.addPhrase(phrase)
        
        # Event queue for async handling
        event_queue: asyncio.Queue[Dict[str, Any] | None] = asyncio.Queue()
        
        # Event handlers
        def on_recognizing(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
            """Handle partial recognition results."""
            text = evt.result.text.strip()
            if text:
                event_queue.put_nowait({
                    "type": "partial",
                    "text": text,
                })
        
        def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
            """Handle final recognition results."""
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = evt.result.text.strip()
                if text:
                    event_queue.put_nowait({
                        "type": "result",
                        "text": text,
                    })
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                event_queue.put_nowait({"type": "nomatch"})
        
        def on_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
            """Handle cancellation."""
            if evt.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Azure Speech error: {evt.error_details}")
                event_queue.put_nowait({
                    "type": "error",
                    "error": evt.error_details,
                })
            event_queue.put_nowait(None)  # Signal end
        
        def on_session_stopped(evt: speechsdk.SessionEventArgs) -> None:
            """Handle session stop."""
            event_queue.put_nowait(None)  # Signal end
        
        # Connect event handlers
        recognizer.recognizing.connect(on_recognizing)
        recognizer.recognized.connect(on_recognized)
        recognizer.canceled.connect(on_canceled)
        recognizer.session_stopped.connect(on_session_stopped)
        
        # Start recognition
        recognizer.start_continuous_recognition_async()
        
        audio_task = asyncio.create_task(
            self._pump_audio(audio_source, push_stream)
        )
        
        try:
            while True:
                event = await event_queue.get()
                if event is None:
                    break
                yield event
        finally:
            recognizer.stop_continuous_recognition_async()
            audio_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await audio_task

    async def _pump_audio(
        self,
        audio_source: AsyncIterable[bytes],
        push_stream: speechsdk.audio.PushAudioInputStream,
    ) -> None:
        """Forward audio chunks from the provided source into Azure's push stream."""
        try:
            async for chunk in audio_source:
                if not chunk:
                    continue
                push_stream.write(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Audio streaming interrupted: {exc}")
        finally:
            with contextlib.suppress(Exception):
                push_stream.close()
    
    def recognize_once(self, audio_bytes: bytes) -> str:
        """
        Perform one-shot speech recognition on audio bytes.
        
        Args:
            audio_bytes: Audio data (WAV or PCM)
        
        Returns:
            Recognized text
        
        Raises:
            AzureSpeechError: If recognition fails
        """
        # Convert to WAV if raw PCM
        if not audio_bytes.startswith(b"RIFF"):
            audio_bytes = _pcm_to_wav(audio_bytes, self.sample_rate)
        
        # Create recognizer from bytes
        audio_stream = speechsdk.audio.PushAudioInputStream()
        audio_stream.write(audio_bytes)
        audio_stream.close()
        
        audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config,
        )
        
        # Recognize
        result = recognizer.recognize_once()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text.strip()
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return ""
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            raise AzureSpeechError(f"Recognition canceled: {cancellation.reason} - {cancellation.error_details}")
        else:
            raise AzureSpeechError(f"Unexpected result reason: {result.reason}")
