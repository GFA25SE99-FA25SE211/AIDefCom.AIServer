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
from services.audio_processing.audio_utils import pcm_to_wav

logger = logging.getLogger(__name__)


DEFAULT_PHRASE_HINTS: Sequence[str] = (
    # --- Academic & Defense context / Học thuật & bảo vệ khóa luận ---
    "capstone project",
    "graduation thesis",
    "thesis defense",
    "bảo vệ khóa luận",
    "đề tài tốt nghiệp",
    "hội đồng bảo vệ",
    "ủy viên phản biện",
    "chủ tịch hội đồng",
    "thư ký hội đồng",
    "giảng viên hướng dẫn",
    "student presentation",
    "slide trình bày",
    "evaluation form",
    "phiếu chấm điểm",
    "assessment rubric",
    "project report",
    "bản báo cáo",
    "final defense",
    "final presentation",
    "thuyết trình",
    "feedback",
    "nhận xét của giảng viên",
    "điểm trung bình",
    "overall score",
    "grading system",
    "rubric criteria",

    # --- Architecture & Software design / Kiến trúc & thiết kế phần mềm ---
    "three layer architecture",
    "three tier architecture",
    "multi layer system",
    "clean architecture",
    "domain driven design",
    "DDD pattern",
    "repository pattern",
    "unit of work",
    "dependency injection",
    "application layer",
    "infrastructure layer",
    "domain layer",
    "API layer",
    "REST API",
    "web service",
    "microservices",
    "micro service",
    "event driven architecture",
    "message queue",
    "event bus",
    "signalR",
    "real time communication",
    "communication protocol",
    "socket connection",
    "websocket endpoint",
    "event handler",
    "pipeline processing",
    "data flow diagram",
    "system architecture diagram",
    "use case diagram",
    "sequence diagram",
    "activity diagram",
    "biểu đồ hệ thống",
    "kiến trúc phần mềm",
    "thiết kế hệ thống",
    "mô hình 3 lớp",

    # --- Backend & Database / Xử lý phía server & cơ sở dữ liệu ---
    "C Sharp",
    "dot net core",
    "ASP dot net API",
    "entity framework core",
    "ef core",
    "SQL Server",
    "database schema",
    "stored procedure",
    "foreign key",
    "migration",
    "seeding data",
    "auto mapper",
    "data transfer object",
    "DTO",
    "repository",
    "controller",
    "service layer",
    "dependency injection container",
    "jwt authentication",
    "authorization policy",
    "role based access control",
    "RBAC",
    "middleware",
    "background service",
    "worker queue",
    "caching layer",
    "redis cache",
    "memory cache",
    "azure redis cache",
    "azure sql database",
    "connection string",
    "app settings",
    "environment variable",

    # --- AI / Speech / NLP Integration / Tích hợp trí tuệ nhân tạo ---
    "speech recognition",
    "speech to text",
    "text to speech",
    "voice authentication",
    "speaker diarization",
    "real time transcription",
    "true text punctuation",
    "azure speech service",
    "speech brain",
    "ecapa tdnn",
    "xvector model",
    "mfcc feature extraction",
    "voice embedding",
    "speaker verification",
    "machine learning",
    "deep learning",
    "neural network",
    "inference model",
    "model serving",
    "AI inference",
    "predictive model",
    "ai microservice",
    "service for inference",
    "tách giọng nói",
    "xác thực giọng nói",
    "chuyển giọng nói thành văn bản",
    "tự động chấm điểm",
    "phân tích nội dung bài nói",
    "phân loại vai trò người nói",
    "giọng giảng viên",
    "giọng sinh viên",

    # --- DevOps & Cloud / Triển khai & đám mây ---
    "docker",
    "docker compose",
    "containerization",
    "container registry",
    "azure app service",
    "azure blob storage",
    "azure api management",
    "azure static web app",
    "azure devops",
    "github actions",
    "CI CD pipeline",
    "continuous integration",
    "continuous deployment",
    "cloud infrastructure",
    "cloud deployment",
    "virtual machine",
    "load balancer",
    "reverse proxy",
    "nginx configuration",
    "api gateway",
    "gateway routing",
    "environment configuration",
    "monitoring and logging",
    "application insight",
    "telemetry data",

    # --- Frontend & UX / Giao diện người dùng ---
    "react",
    "react native",
    "next js",
    "frontend application",
    "state management",
    "redux toolkit",
    "component",
    "tailwind css",
    "ui design",
    "responsive layout",
    "authentication flow",
    "api integration",
    "frontend to backend",
    "jwt token",
    "form validation",
    "real time update",
    "websocket client",
    "user dashboard",
    "admin dashboard",
    "notification center",
    "chat interface",
    "speech input",
    "voice chat",
    "AI chatbot",

    # --- Project management / Quản lý dự án ---
    "project timeline",
    "work breakdown structure",
    "WBS",
    "kanban board",
    "Jira issue",
    "GitHub repository",
    "commit history",
    "version control",
    "release note",
    "deployment schedule",
    "demo day",
    "milestone",
    "project budget",
    "team collaboration",
    "phân công công việc",
    "tiến độ dự án",
    "kế hoạch bảo vệ",
    "deadline",
    "cập nhật tiến độ",
    "báo cáo hàng tuần",

    # --- Evaluation & Communication / Đánh giá & tương tác ---
    "question and answer",
    "Q and A session",
    "phần hỏi đáp",
    "thắc mắc",
    "ý kiến phản biện",
    "câu hỏi của giảng viên",
    "phần trả lời của sinh viên",
    "clarification",
    "feedback",
    "recommendation",
    "điểm số",
    "chấm điểm",
    "nhận xét",
    "đề xuất cải thiện",
    "final score",
    "grade sheet",
)


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
        
        # Segmentation & silence timeouts (optimized for Vietnamese with faster response)
        # Reduced from 800ms to 500ms for quicker speaker change detection
        self.speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "500"
        )
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "5000"
        )
        # Reduced from 500ms to 300ms for faster finalization
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "300"
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
            audio_bytes = pcm_to_wav(audio_bytes, self.sample_rate)
        
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
