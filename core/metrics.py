"""Prometheus metrics for observability."""

from prometheus_client import Counter, Histogram, Gauge, Info

# Speaker Identification Metrics
speaker_id_latency = Histogram(
    'speaker_identification_latency_seconds',
    'Time to identify speaker from audio',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

speaker_id_total = Counter(
    'speaker_identification_total',
    'Total speaker identification attempts',
    ['status']  # success, failed, cached
)

# Voice Embedding Metrics
embedding_extraction_time = Histogram(
    'embedding_extraction_seconds',
    'Time to extract voice embedding from audio',
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

# Cache Metrics
cache_operations = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['cache_type', 'operation', 'status']  # cache_type: L1/L2/L3, operation: get/set/delete
)

cache_hit_ratio = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio',
    ['cache_type']
)

# WebSocket Metrics
websocket_connections = Gauge(
    'websocket_connections_active',
    'Number of active WebSocket connections',
    ['endpoint']
)

websocket_messages = Counter(
    'websocket_messages_total',
    'Total WebSocket messages',
    ['endpoint', 'message_type']  # partial, result, error
)

# Transcript Metrics
transcript_save_total = Counter(
    'transcript_save_total',
    'Total transcript save attempts',
    ['status']  # success, failed, retry_success, queued
)

transcript_failed_queue_size = Gauge(
    'transcript_failed_queue_size',
    'Number of transcripts in failed queue'
)

# Database Metrics
db_query_duration = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['query_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

db_connection_pool_size = Gauge(
    'db_connection_pool_size',
    'Database connection pool size',
    ['pool_type']  # active, idle
)

# External API Metrics
external_api_call_duration = Histogram(
    'external_api_call_duration_seconds',
    'External API call duration',
    ['service', 'endpoint'],  # service: auth_service, defense_session
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

external_api_errors = Counter(
    'external_api_errors_total',
    'External API errors',
    ['service', 'error_type']  # timeout, connection, http_error
)

# Circuit Breaker Metrics
circuit_breaker_state = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half_open)',
    ['circuit_name']
)

circuit_breaker_failures = Counter(
    'circuit_breaker_failures_total',
    'Circuit breaker failures',
    ['circuit_name']
)

# Audio Processing Metrics
audio_quality_score = Histogram(
    'audio_quality_score',
    'Audio quality scores',
    ['metric_type'],  # rms, snr, voiced_ratio
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

filler_words_filtered = Counter(
    'filler_words_filtered_total',
    'Number of filler words filtered',
    ['language']  # vi, en
)

# System Info
app_info = Info('app', 'Application information')
app_info.info({
    'version': '3.0.0',
    'service': 'AIDefCom.AIServer',
    'features': 'stt,voice_auth,multi_speaker'
})
