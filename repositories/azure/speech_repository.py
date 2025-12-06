"""Azure Speech Repository - Azure Speech SDK wrapper for speech recognition."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from typing import Any, AsyncGenerator, AsyncIterable, Dict, Iterable, List, Sequence

import azure.cognitiveservices.speech as speechsdk

from core.exceptions import AzureSpeechError
from repositories.interfaces.speech_repository import ISpeechRepository
from repositories.models.speech_config import SpeechRecognitionConfig
from services.audio.utils import pcm_to_wav

logger = logging.getLogger(__name__)


def _load_phrase_hints_from_file(file_path: str) -> List[str]:
    """Load phrase hints from JSON file."""
    if not file_path or not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        phrases: List[str] = []
        for key, value in data.items():
            if key.startswith("_"):
                continue
            if isinstance(value, list):
                phrases.extend([p for p in value if isinstance(p, str)])
        
        logger.info(f"Loaded {len(phrases)} phrase hints from {file_path}")
        return phrases
    except Exception as e:
        logger.error(f"Error loading phrase hints from {file_path}: {e}")
        return []


_FALLBACK_PHRASE_HINTS: Sequence[str] = (
    "xin chào", "cảm ơn", "vâng ạ", "dạ vâng", "được ạ", "không ạ",
    "bảo vệ khóa luận", "bảo vệ đồ án", "hội đồng bảo vệ",
    "REST API", "microservices", "database", "repository pattern",
)


def _get_default_phrase_hints() -> Sequence[str]:
    """Get default phrase hints from config file or fallback."""
    try:
        from app.config import Config
        if Config.PHRASE_HINTS_FILE:
            hints = _load_phrase_hints_from_file(Config.PHRASE_HINTS_FILE)
            if hints:
                return tuple(hints)
        default_path = os.path.join(Config.PROJECT_ROOT, "data", "phrase_hints.json")
        if os.path.exists(default_path):
            hints = _load_phrase_hints_from_file(default_path)
            if hints:
                return tuple(hints)
    except Exception as e:
        logger.warning(f"Failed to load phrase hints from config: {e}")
    
    return _FALLBACK_PHRASE_HINTS


class AzureSpeechRepository(ISpeechRepository):
    """Repository for Azure Speech Service operations."""
    
    _PROVIDER_NAME: str = "azure"
    _SUPPORTED_LANGUAGES: tuple[str, ...] = (
        "vi-VN", "en-US", "en-GB", "ja-JP", "ko-KR",
        "zh-CN", "zh-TW", "th-TH", "fr-FR", "de-DE",
    )
    
    def get_provider_name(self) -> str:
        return self._PROVIDER_NAME
    
    def get_supported_languages(self) -> list[str]:
        return list(self._SUPPORTED_LANGUAGES)
    
    def configure(self, config: SpeechRecognitionConfig) -> None:
        """Apply provider-agnostic configuration to Azure Speech."""
        if config.language and config.language in self._SUPPORTED_LANGUAGES:
            self.language = config.language
            self.speech_config.speech_recognition_language = config.language
        
        if config.sample_rate:
            self.sample_rate = config.sample_rate
        
        if config.provider_options:
            opts = config.provider_options
            if "segmentation_silence_ms" in opts:
                self.speech_config.set_property(
                    speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
                    str(opts["segmentation_silence_ms"])
                )
            if "initial_silence_ms" in opts:
                self.speech_config.set_property(
                    speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
                    str(opts["initial_silence_ms"])
                )
            if "end_silence_ms" in opts:
                self.speech_config.set_property(
                    speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
                    str(opts["end_silence_ms"])
                )
            if "stable_partial_threshold" in opts:
                self.speech_config.set_property(
                    speechsdk.PropertyId.SpeechServiceResponse_StablePartialResultThreshold,
                    str(opts["stable_partial_threshold"])
                )
        
        if config.phrase_hints:
            self.default_phrase_hints = tuple(config.phrase_hints)
        
        logger.info(
            f"Azure Speech reconfigured | language={self.language} | "
            f"sample_rate={self.sample_rate} | hints={len(self.default_phrase_hints)}"
        )
    
    def __init__(
        self,
        subscription_key: str,
        region: str,
        language: str = "vi-VN",
        default_phrase_hints: Sequence[str] | None = None,
        sample_rate: int = 16000,
        segmentation_silence_ms: int | None = None,
        initial_silence_ms: int | None = None,
        end_silence_ms: int | None = None,
        stable_partial_threshold: int | None = None,
    ) -> None:
        """Initialize Azure Speech repository."""
        try:
            from app.config import Config
            _segmentation_silence = segmentation_silence_ms or Config.AZURE_SPEECH_SEGMENTATION_SILENCE_MS
            _initial_silence = initial_silence_ms or Config.AZURE_SPEECH_INITIAL_SILENCE_MS
            _end_silence = end_silence_ms or Config.AZURE_SPEECH_END_SILENCE_MS
            _stable_threshold = stable_partial_threshold or Config.AZURE_SPEECH_STABLE_PARTIAL_THRESHOLD
        except Exception:
            _segmentation_silence = segmentation_silence_ms or 600
            _initial_silence = initial_silence_ms or 5000
            _end_silence = end_silence_ms or 400
            _stable_threshold = stable_partial_threshold or 4
        
        self.speech_key = subscription_key
        self.subscription_key = subscription_key
        self.region = region
        self.language = language
        self.sample_rate = sample_rate
        self.default_phrase_hints: Sequence[str] = (
            tuple(default_phrase_hints) if default_phrase_hints else _get_default_phrase_hints()
        )
        
        # Check for Custom Speech endpoint FIRST
        custom_endpoint_id = None
        try:
            from app.config import Config
            custom_endpoint_id = getattr(Config, 'AZURE_SPEECH_CUSTOM_ENDPOINT_ID', '') or ''
            custom_endpoint_id = custom_endpoint_id.strip() if custom_endpoint_id else ''
        except Exception:
            custom_endpoint_id = ''
        
        # Create SpeechConfig - use Custom Speech endpoint if available
        if custom_endpoint_id:
            # For Custom Speech, we need to set endpoint_id property
            # The endpoint URL format: wss://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?cid={endpoint_id}
            self.speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
            self.speech_config.endpoint_id = custom_endpoint_id
            logger.info(f"✓ Using Custom Speech model: {custom_endpoint_id}")
        else:
            self.speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
            logger.info("Using default Azure Speech model")
        
        self.speech_config.speech_recognition_language = language
        self.speech_config.output_format = speechsdk.OutputFormat.Detailed
        
        # Use CONVERSATION mode for Vietnamese - better for real-time streaming
        # DICTATION mode can cause delayed results and over-buffering
        # INTERACTIVE is too aggressive with end-pointing
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_RecoMode,
            "CONVERSATION"
        )
        
        # Enable audio logging for debugging (disable in production)
        # self.speech_config.set_property(
        #     speechsdk.PropertyId.Speech_LogFilename, "speech_log.txt"
        # )
        
        # Request detailed JSON with NBest alternatives for better accuracy
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceResponse_RequestDetailedResultTrueFalse,
            "true"
        )
        
        # Segmentation silence: time to wait before ending a phrase
        # Higher = longer phrases, fewer fragments
        # For Vietnamese, use 1000-1500ms to avoid cutting mid-sentence
        self.speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
            str(max(_segmentation_silence, 1000))  # At least 1000ms for Vietnamese
        )
        
        # Initial silence: how long to wait for speech to start
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            str(_initial_silence)
        )
        
        # End silence: time after speech ends to finalize result
        # For Vietnamese, use 600-800ms to allow natural pauses
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
            str(max(_end_silence, 600))  # At least 600ms
        )
        
        # For Vietnamese: Don't use TrueText (English-optimized)
        # Request word-level timestamps for confidence analysis
        try:
            self.speech_config.request_word_level_timestamps()
        except Exception:
            pass
        
        self.speech_config.set_profanity(speechsdk.ProfanityOption.Raw)  # Don't mask - academic context
        
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceResponse_StablePartialResultThreshold,
            str(_stable_threshold)
        )
        
        endpoint_info = f"custom={custom_endpoint_id[:8]}..." if custom_endpoint_id else "default"
        logger.info(
            f"Azure Speech Repository initialized | language={language} | region={region} | "
            f"sr={sample_rate}Hz | hints={len(self.default_phrase_hints)} | endpoint={endpoint_info}"
        )
    
    async def recognize_continuous_async(
        self,
        audio_stream: Any,
        speaker: str,
        extra_phrases: Sequence[str] | None = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Implement interface method for continuous recognition."""
        if hasattr(audio_stream, "__aiter__") and not isinstance(audio_stream, speechsdk.audio.PushAudioInputStream):
            async for evt in self.recognize_stream(audio_stream, extra_phrases=extra_phrases):
                yield evt
            return

        fmt = speechsdk.audio.AudioStreamFormat(
            samples_per_second=self.sample_rate,
            bits_per_sample=16,
            channels=1,
        )
        try:
            if not isinstance(audio_stream, speechsdk.audio.PushAudioInputStream):
                push_stream = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
                if hasattr(audio_stream, "read"):
                    data = audio_stream.read()
                    if data:
                        push_stream.write(data)
                audio_stream = push_stream
        except Exception as exc:
            logger.warning(f"Failed to adapt audio_stream: {exc}")
            if hasattr(audio_stream, "__aiter__"):
                async for evt in self.recognize_stream(audio_stream, extra_phrases=extra_phrases):
                    yield evt
                return
            raise

        audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config,
        )

        all_hints = list(self.default_phrase_hints)
        if extra_phrases:
            all_hints.extend(extra_phrases)
        if speaker:
            lowered = speaker.lower()
            if lowered not in {"khách", "guest", "unknown"}:
                all_hints.append(speaker)
        if all_hints:
            phrase_list = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
            for phrase in all_hints:
                with contextlib.suppress(Exception):
                    phrase_list.addPhrase(phrase)

        event_queue: asyncio.Queue[Dict[str, Any] | None] = asyncio.Queue()

        def _extract_best_text(result: speechsdk.SpeechRecognitionResult) -> str:
            """Extract best text from recognition result.
            
            With Detailed output format, analyze NBest alternatives and pick
            the one with highest confidence. This helps Vietnamese recognition
            by choosing the most likely transcription.
            """
            import json
            
            fallback_text = result.text.strip() if result.text else ""
            
            try:
                props_json = result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult, ""
                )
                if not props_json:
                    return fallback_text
                    
                data = json.loads(props_json)
                
                # Check NBest alternatives for highest confidence
                nbest = data.get("NBest", [])
                if nbest:
                    # Sort by confidence descending
                    sorted_alternatives = sorted(
                        nbest, 
                        key=lambda x: x.get("Confidence", 0), 
                        reverse=True
                    )
                    
                    best_alt = sorted_alternatives[0]
                    best_confidence = best_alt.get("Confidence", 0)
                    
                    # For Vietnamese: prefer Display text (with ITN applied)
                    # over Lexical (raw form) or regular text
                    best_text = (
                        best_alt.get("Display", "") or 
                        best_alt.get("Lexical", "") or
                        best_alt.get("ITN", "")
                    ).strip()
                    
                    if best_text and best_confidence > 0.5:
                        # Log low confidence for debugging
                        if best_confidence < 0.7:
                            logger.debug(
                                f"Low confidence ({best_confidence:.2f}): '{best_text[:50]}...'"
                            )
                        return best_text
                
                # Fallback to DisplayText from main result
                display_text = data.get("DisplayText", "").strip()
                if display_text:
                    return display_text
                    
            except Exception as e:
                logger.debug(f"JSON parse error: {e}")
            
            return fallback_text

        def on_recognizing(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
            text = _extract_best_text(evt.result)
            if text:
                event_queue.put_nowait({"type": "partial", "text": text})

        def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = _extract_best_text(evt.result)
                if text:
                    event_queue.put_nowait({"type": "result", "text": text})
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                event_queue.put_nowait({"type": "nomatch"})

        def on_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
            if evt.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Azure Speech error: {evt.error_details}")
                event_queue.put_nowait({"type": "error", "error": evt.error_details})
            event_queue.put_nowait(None)

        def on_session_stopped(evt: speechsdk.SessionEventArgs) -> None:
            event_queue.put_nowait(None)

        recognizer.recognizing.connect(on_recognizing)
        recognizer.recognized.connect(on_recognized)
        recognizer.canceled.connect(on_canceled)
        recognizer.session_stopped.connect(on_session_stopped)

        recognizer.start_continuous_recognition_async()
        try:
            while True:
                event = await event_queue.get()
                if event is None:
                    break
                yield event
        finally:
            with contextlib.suppress(Exception):
                recognizer.stop_continuous_recognition_async()
    
    async def recognize_stream(
        self,
        audio_source: AsyncIterable[bytes],
        extra_phrases: Iterable[str] | None = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream speech recognition from an async audio source."""
        print(f"🔵 [AZURE] recognize_stream START | region={self.region} | language={self.language}")
        print(f"🔵 [AZURE] speech_key exists: {bool(self.speech_key)} | Custom endpoint: {getattr(self.speech_config, 'endpoint_id', 'None')}")
        fmt = speechsdk.audio.AudioStreamFormat(
            samples_per_second=self.sample_rate,
            bits_per_sample=16,
            channels=1,
        )
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config,
        )
        
        all_hints = list(self.default_phrase_hints)
        if extra_phrases:
            all_hints.extend(extra_phrases)
        
        if all_hints:
            phrase_list = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
            for phrase in all_hints:
                phrase_list.addPhrase(phrase)
        
        event_queue: asyncio.Queue[Dict[str, Any] | None] = asyncio.Queue()
        
        def _extract_best_text(result: speechsdk.SpeechRecognitionResult) -> str:
            """Extract best text from recognition result with NBest analysis."""
            import json
            
            fallback_text = result.text.strip() if result.text else ""
            
            try:
                props_json = result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult, ""
                )
                if not props_json:
                    return fallback_text
                    
                data = json.loads(props_json)
                
                # Check NBest alternatives for highest confidence
                nbest = data.get("NBest", [])
                if nbest:
                    sorted_alternatives = sorted(
                        nbest, 
                        key=lambda x: x.get("Confidence", 0), 
                        reverse=True
                    )
                    best_alt = sorted_alternatives[0]
                    best_text = (
                        best_alt.get("Display", "") or 
                        best_alt.get("Lexical", "")
                    ).strip()
                    if best_text:
                        return best_text
                
                display_text = data.get("DisplayText", "").strip()
                if display_text:
                    return display_text
                    
            except Exception:
                pass
            
            return fallback_text
        
        recognition_count = {"partial": 0, "result": 0}
        
        def on_recognizing(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
            text = _extract_best_text(evt.result)
            if text:
                recognition_count["partial"] += 1
                if recognition_count["partial"] == 1:
                    print(f"🟢 [AZURE] First partial! text='{text[:50]}...'")
                event_queue.put_nowait({"type": "partial", "text": text})
        
        def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = _extract_best_text(evt.result)
                if text:
                    recognition_count["result"] += 1
                    print(f"🟢 [AZURE] Result #{recognition_count['result']}: '{text}'")
                    event_queue.put_nowait({"type": "result", "text": text})
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                event_queue.put_nowait({"type": "nomatch"})
        
        def on_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
            print(f"🔴 [AZURE] on_canceled | reason={evt.reason} | error={evt.error_details if hasattr(evt, 'error_details') else 'N/A'}")
            if evt.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Azure Speech error: {evt.error_details}")
                event_queue.put_nowait({"type": "error", "error": evt.error_details})
            event_queue.put_nowait(None)
        
        def on_session_stopped(evt: speechsdk.SessionEventArgs) -> None:
            print(f"🔵 [AZURE] on_session_stopped")
            event_queue.put_nowait(None)
        
        recognizer.recognizing.connect(on_recognizing)
        recognizer.recognized.connect(on_recognized)
        recognizer.canceled.connect(on_canceled)
        recognizer.session_stopped.connect(on_session_stopped)
        
        print(f"🔵 [AZURE] Starting continuous recognition...")
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
        """Forward audio chunks into Azure's push stream."""
        chunk_count = 0
        total_bytes = 0
        print(f"🔵 [AZURE] _pump_audio START")
        try:
            async for chunk in audio_source:
                if not chunk:
                    continue
                chunk_count += 1
                total_bytes += len(chunk)
                if chunk_count == 1:
                    print(f"🔵 [AZURE] First audio chunk pushed to Azure | size={len(chunk)} bytes")
                elif chunk_count % 50 == 0:
                    print(f"🔵 [AZURE] Audio chunks pushed: {chunk_count} | total_bytes={total_bytes}")
                push_stream.write(chunk)
        except asyncio.CancelledError:
            print(f"🔵 [AZURE] _pump_audio cancelled | chunks={chunk_count} | bytes={total_bytes}")
            raise
        except Exception as exc:
            print(f"🔴 [AZURE] _pump_audio error: {exc}")
            logger.warning(f"Audio streaming interrupted: {exc}")
        finally:
            print(f"🔵 [AZURE] _pump_audio END | total_chunks={chunk_count} | total_bytes={total_bytes}")
            with contextlib.suppress(Exception):
                push_stream.close()
    
    def recognize_once(self, audio_bytes: bytes) -> str:
        """Perform one-shot speech recognition on audio bytes."""
        if not audio_bytes.startswith(b"RIFF"):
            audio_bytes = pcm_to_wav(audio_bytes, self.sample_rate)
        
        audio_stream = speechsdk.audio.PushAudioInputStream()
        audio_stream.write(audio_bytes)
        audio_stream.close()
        
        audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config,
        )
        
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
