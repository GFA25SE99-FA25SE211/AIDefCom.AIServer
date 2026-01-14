# AIDefCom AI Service

<div align="center">

**Real-time Speech-to-Text + Voice Authentication + Question Duplicate Detection**

[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/AIDefCom/AIServer)
[![Python](https://img.shields.io/badge/python-3.11-green.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Architecture](#-architecture)
3. [Features](#-features)
4. [Technology Stack](#-technology-stack)
5. [Installation](#-installation)
6. [Configuration](#-configuration)
7. [API Reference](#-api-reference)
   - [Voice Authentication APIs](#-voice-authentication-apis)
   - [Speech-to-Text WebSocket](#-speech-to-text-websocket)
   - [Question Management APIs](#-question-management-apis)
   - [System Endpoints](#-system-endpoints)
8. [Integration Examples](#-integration-examples)
9. [Deployment](#-deployment)
10. [Development](#-development)

---

## 🎯 Overview

**AIDefCom AI Service** is an AI microservice providing 3 main features:

| Feature | Description |
|---------|-------------|
| **🎤 Voice Authentication** | Register and authenticate users via voice (Pyannote/WeSpeaker) |
| **🎙️ Speech-to-Text** | Real-time streaming STT with Azure Cognitive Services + auto speaker identification |
| **❓ Question Detection** | Detect duplicate questions using fuzzy matching + semantic similarity |

**Base URL:** `https://<your-app>.azurewebsites.net`  
**Swagger UI:** `https://<your-app>.azurewebsites.net/docs`  
**ReDoc:** `https://<your-app>.azurewebsites.net/redoc`

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AIDefCom AI Service                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │   FastAPI   │  │  WebSocket  │  │  Prometheus │                │
│  │   Router    │  │   Handler   │  │   Metrics   │                │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘                │
│         │                │                                         │
│  ┌──────┴────────────────┴──────┐                                 │
│  │         Service Layer        │                                 │
│  │  ┌────────────────────────┐  │                                 │
│  │  │ VoiceService           │  │  Pyannote/WeSpeaker Embeddings │
│  │  │ SpeechService          │  │  Azure Speech SDK              │
│  │  │ QuestionService        │  │  Sentence-Transformers         │
│  │  │ RedisService           │  │  Caching & Session State       │
│  │  └────────────────────────┘  │                                 │
│  └──────────────────────────────┘                                 │
│                │                                                   │
│  ┌─────────────┴─────────────┐                                    │
│  │      Repository Layer     │                                    │
│  │  ┌──────────────────────┐ │                                    │
│  │  │ AzureBlobRepository  │ │  Voice Profiles Storage           │
│  │  │ SQLServerRepository  │ │  User Data                        │
│  │  │ AzureSpeechRepository│ │  Speech Recognition               │
│  │  └──────────────────────┘ │                                    │
│  └───────────────────────────┘                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Azure Blob   │   │ Azure Cache   │   │  SQL Server   │
│   Storage     │   │  for Redis    │   │               │
└───────────────┘   └───────────────┘   └───────────────┘
```

---

## ✨ Features

### 🔐 Voice Authentication
- **Enrollment**: Register voice samples (minimum 3 samples required)
- **Identification**: 1:N speaker identification (compare with all enrolled users)
- **Verification**: 1:1 voice verification (check match with specific user)
- **Auto Speaker Detection**: Automatic speaker recognition during streaming STT

### 🎙️ Speech-to-Text
- **Real-time Streaming**: WebSocket with Azure Cognitive Services
- **Vietnamese Optimized**: Tuned timeouts for Vietnamese language
- **Custom Speech Model**: Support for Azure Custom Speech endpoint
- **Multi-speaker Support**: Automatic speaker detection and labeling
- **Transcript Caching**: Redis-backed transcript with auto-resume

### ❓ Question Detection
- **Fuzzy Matching**: RapidFuzz (ratio, token_sort, token_set)
- **Semantic Similarity**: Sentence-Transformers embeddings
- **Vietnamese Support**: Vietnamese opposite keywords detection
- **Session-based**: Questions grouped by session ID

### ⚡ Performance & Scalability
- **Lazy Model Loading**: Background warmup cho ACA health probes
- **LRU Caching**: Memory-efficient embedding cache với TTL
- **Connection Pooling**: SQL Server, Redis connection pools
- **Rate Limiting**: SlowAPI với per-IP limits
- **Prometheus Metrics**: Built-in observability

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| **Framework** | FastAPI 0.100+, Uvicorn |
| **Speech** | Azure Cognitive Services Speech SDK |
| **Voice AI** | Pyannote.audio, WeSpeaker, PyTorch |
| **NLP** | Sentence-Transformers, RapidFuzz |
| **Cache** | Azure Cache for Redis (async) |
| **Storage** | Azure Blob Storage, SQL Server |
| **Monitoring** | Prometheus, Structlog |
| **Container** | Docker (Python 3.11-slim) |

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- Redis (optional, for caching)
- Azure Cognitive Services subscription
- Azure Blob Storage account

### Local Development

```bash
# Clone repository
git clone https://github.com/AIDefCom/AIServer.git
cd AIServer

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy environment template
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac

# Edit .env with your credentials
notepad .env

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build image
docker build -t aidefcom-ai-service .

# Run container
docker run -p 8000:8000 --env-file .env aidefcom-ai-service
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

#### Azure Speech Service (Required)
```env
AZURE_SPEECH_KEY=your_speech_key
AZURE_SPEECH_REGION=southeastasia
AZURE_SPEECH_CUSTOM_ENDPOINT_ID=  # Optional: Custom Speech model
```

#### Azure Cache for Redis (Optional)
```env
REDIS_HOST=your-cache.redis.cache.windows.net
REDIS_PORT=6380
REDIS_PASSWORD=your_redis_key
REDIS_SSL=true
REDIS_DB=0
REDIS_TTL_SECONDS=3600
```

#### Azure Blob Storage (Required for voice profiles)
```env
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
AZURE_BLOB_CONTAINER_NAME=voice-sample
```

#### SQL Server (Optional)
```env
# Option 1: Connection string
SQL_SERVER_CONNECTION_STRING=Server=...;Database=...;

# Option 2: Individual parameters
SQL_SERVER_HOST=your-server.database.windows.net
SQL_SERVER_PORT=1433
SQL_SERVER_DATABASE=your_database
SQL_SERVER_USERNAME=your_user
SQL_SERVER_PASSWORD=your_password
```

#### Auth Service Integration (Optional)
```env
AUTH_SERVICE_BASE_URL=https://your-auth-service.com/api
AUTH_SERVICE_VERIFY_SSL=true
AUTH_SERVICE_TIMEOUT=10
```

#### Voice Authentication Tuning
```env
# Thresholds
VOICE_COSINE_THRESHOLD=0.50          # Main identification threshold
VOICE_SPEAKER_LOCK_DECAY_SECONDS=8.0 # Speaker lock duration
VOICE_SPEAKER_SWITCH_MARGIN=0.10     # Margin to switch speaker
VOICE_SPEAKER_SWITCH_HITS_REQUIRED=4 # Confirmations before switch

# Audio quality
VOICE_MIN_DURATION=1.5               # Min audio duration (seconds)
VOICE_MIN_ENROLL_DURATION=10.0       # Min enrollment audio
VOICE_RMS_FLOOR=0.005                # Min RMS level
VOICE_SNR_FLOOR_DB=8.0               # Min SNR (dB)
```

#### Azure Speech Timeouts (Vietnamese Optimized)
```env
AZURE_SPEECH_SEGMENTATION_SILENCE_MS=1200  # Segmentation pause
AZURE_SPEECH_INITIAL_SILENCE_MS=8000       # Wait for speech start
AZURE_SPEECH_END_SILENCE_MS=800            # End of utterance
```

---

## 📚 API Reference

### 🔐 Voice Authentication APIs

#### 1. Enroll Voice Sample

**Endpoint:** `POST /voice/users/{user_id}/enroll`

Register a voice sample. **Minimum 3 samples** are required to complete enrollment.

**Request:**
```http
POST /voice/users/USR001/enroll
Content-Type: multipart/form-data

audio_file: <binary audio file>
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | path | ✅ | User ID to enroll |
| `audio_file` | file | ✅ | Audio file (WAV/MP3/FLAC, max 6MB, ~3-5s) |

**Response:**
```json
{
  "type": "enrollment",
  "success": true,
  "user_id": "USR001",
  "enrollment_count": 2,
  "min_required": 3,
  "is_complete": false,
  "message": "Enrollment sample 2/3 saved successfully"
}
```

| Status | Description |
|--------|-------------|
| `200` | Enrollment successful |
| `400` | Invalid audio, quality too low |
| `504` | Processing timeout |

---

#### 2. Identify Speaker

**Endpoint:** `POST /voice/identify`

Identify speaker from all enrolled users (1:N comparison).

**Request:**
```http
POST /voice/identify
Content-Type: multipart/form-data

audio_file: <binary audio file>
```

**Response (Match Found):**
```json
{
  "type": "identification",
  "success": true,
  "identified": true,
  "speaker_id": "USR001",
  "speaker_name": "Nguyễn Văn A",
  "score": 0.88,
  "confidence": 0.90,
  "message": "Speaker identified successfully"
}
```

**Response (No Match):**
```json
{
  "type": "identification",
  "success": true,
  "identified": false,
  "speaker_id": null,
  "speaker_name": null,
  "score": 0.45,
  "confidence": 0.0,
  "message": "No matching speaker found"
}
```

| Field | Description |
|-------|-------------|
| `identified` | Whether speaker was found |
| `score` | Cosine similarity (0-1) |
| `confidence` | Derived confidence level |

---

#### 3. Verify Voice

**Endpoint:** `POST /voice/users/{user_id}/verify`

Verify if audio matches a specific user (1:1 verification).

**Request:**
```http
POST /voice/users/USR001/verify
Content-Type: multipart/form-data

audio_file: <binary audio file>
```

**Response:**
```json
{
  "type": "verification",
  "success": true,
  "verified": true,
  "claimed_id": "USR001",
  "speaker_id": "USR001",
  "match": true,
  "score": 0.91,
  "confidence": 0.94,
  "message": "Voice verified successfully"
}
```

---

#### 4. Get Enrollment Status

**Endpoint:** `GET /voice/users/{user_id}/enrollment-status`

Check enrollment status of a user.

**Response:**
```json
{
  "user_id": "USR001",
  "name": "Nguyễn Văn A",
  "enrollment_status": "partial",
  "enrollment_count": 2,
  "min_required": 3,
  "is_complete": false,
  "message": "User has 2/3 enrollment samples"
}
```

| Status | Description |
|--------|-------------|
| `not_enrolled` | No samples registered yet |
| `partial` | Has 1-2 samples |
| `enrolled` | Has ≥3 samples (complete) |

---

#### 5. Reset Enrollment

**Endpoint:** `DELETE /voice/users/{user_id}/enrollment`

Delete all enrollment data for a user (requires re-enrollment from scratch).

**Response:**
```json
{
  "success": true,
  "user_id": "USR001",
  "message": "Enrollment reset successful for user USR001",
  "details": {
    "blob_deleted": true,
    "db_cleared": true,
    "cache_cleared": true
  }
}
```

---

### 🎙️ Speech-to-Text WebSocket

#### WebSocket Endpoint

**URL:** `wss://<host>/ws/stt`

Real-time streaming speech-to-text với tự động speaker identification.

#### Connection

```javascript
// Connect with optional defense session
const defenseSessionId = "550e8400-e29b-41d4-a716-446655440000";
const ws = new WebSocket(`wss://your-app.azurewebsites.net/ws/stt?defense_session_id=${defenseSessionId}`);

ws.onopen = () => {
  console.log('Connected to STT WebSocket');
};
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `speaker` | string | Initial speaker label (default: "Identifying...") |
| `phrases` | string | Phrase hints (comma or pipe separated) |
| `defense_session_id` | string | Filter speakers to this session's users |

> **⚠️ Note:** No need to send `user_id` - backend automatically identifies speaker from audio.

#### Sending Audio

```javascript
// Send audio chunks as binary
const audioChunk = new Uint8Array(audioBuffer);
ws.send(audioChunk);

// Audio requirements:
// - Format: PCM 16-bit, mono
// - Sample rate: 16000 Hz
// - Chunk size: 3200-6400 bytes (0.1-0.2s)
```

#### Commands

```javascript
// End session (triggers transcript save)
ws.send("stop");

// Start question capture mode
ws.send("q:start");

// End question capture (triggers duplicate check)
ws.send("q:end");
```

#### Event Types

**1. connected** - Initial connection confirmation
```json
{
  "type": "connected",
  "session_id": "abc123",
  "defense_session_id": "550e8400-...",
  "room_size": 2,
  "message": "WebSocket connected, starting recognition..."
}
```

**2. partial** - Real-time interim result
```json
{
  "type": "partial",
  "text": "Xin chào các",
  "speaker": "Nguyễn Văn A",
  "display": "<span style=\"color:#3498db\">Xin chào các</span>"
}
```

**3. result** - Final recognized segment
```json
{
  "type": "result",
  "text": "Xin chào các bạn, hôm nay chúng ta sẽ học về AI.",
  "speaker": "Nguyễn Văn A",
  "user_id": "USR001",
  "display": "<span style=\"color:#2ecc71\">...</span>"
}
```

**4. question_mode_started**
```json
{
  "type": "question_mode_started",
  "session_id": "abc123"
}
```

**5. question_mode_result** - After q:end
```json
{
  "type": "question_mode_result",
  "session_id": "abc123",
  "is_duplicate": false,
  "question_text": "AI là gì?",
  "similar_questions": []
}
```

**6. cached_transcript** - On reconnect
```json
{
  "type": "cached_transcript",
  "defense_session_id": "550e8400-...",
  "lines": [...],
  "message": "Loaded 15 lines"
}
```

**7. ping** - Keepalive (every 25s)
```json
{
  "type": "ping"
}
```

**8. error**
```json
{
  "type": "error",
  "error": "Audio stream interrupted"
}
```

---

### ❓ Question Management APIs

#### 1. Check Duplicate

**Endpoint:** `POST /questions/check-duplicate`

Check if a question is duplicate (threshold 0.85).

**Request:**
```json
{
  "session_id": "session_123",
  "question_text": "AI là gì?"
}
```

**Response:**
```json
{
  "is_duplicate": true,
  "question_text": "AI là gì?",
  "similar_questions": [
    {
      "text": "Trí tuệ nhân tạo là gì?",
      "score": 0.92,
      "fuzzy_score": 0.85,
      "semantic_score": 0.92
    }
  ],
  "message": "⚠️ Duplicate question! Found 1 similar question."
}
```

#### 2. Register Question

**Endpoint:** `POST /questions/register`

Register a new question (without duplicate check).

**Request:**
```json
{
  "session_id": "session_123",
  "question_text": "Machine Learning hoạt động thế nào?"
}
```

**Response:**
```json
{
  "success": true,
  "question_id": 5,
  "total_questions": 5,
  "message": "✅ Question saved. Total: 5"
}
```

#### 3. Check and Register (Combo)

**Endpoint:** `POST /questions/check-and-register`

Check duplicate + register nếu không trùng.

**Request:**
```json
{
  "session_id": "session_123",
  "question_text": "Deep Learning khác gì ML?"
}
```

#### 4. Get Session Questions

**Endpoint:** `GET /questions/session/{session_id}`

**Response:**
```json
{
  "session_id": "session_123",
  "questions": ["AI là gì?", "ML hoạt động thế nào?"],
  "total": 2
}
```

#### 5. Clear Session Questions

**Endpoint:** `DELETE /questions/session/{session_id}`

**Response:**
```json
{
  "success": true,
  "session_id": "session_123",
  "deleted": 5,
  "message": "✅ Deleted 5 questions."
}
```

---

### 🔧 System Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Service info + warmup status |
| `GET /health` | Liveness probe (always returns 200 for ACA) |
| `GET /ready` | Readiness probe (returns 200 after warmup) |
| `GET /healthz` | Legacy health check |
| `GET /docs` | Swagger UI |
| `GET /redoc` | ReDoc documentation |
| `GET /openapi.json` | OpenAPI schema |
| `GET /metrics` | Prometheus metrics |
| `GET /memory` | Memory statistics |
| `POST /gc` | Force garbage collection |

#### Health Check Response

```json
{
  "status": "healthy",
  "warmup": {
    "stage": "complete",
    "progress": 100,
    "error": null
  },
  "redis": { "status": "healthy" },
  "sql": { "status": "healthy" },
  "blob": { "status": "healthy" },
  "speech": { "status": "healthy" },
  "database_pool": {
    "pool_size": 5,
    "checked_out": 1,
    "overflow": 0
  }
}
```

---

## 📝 Integration Examples

### Voice Authentication Flow

```javascript
const API_BASE = 'https://your-app.azurewebsites.net';

// 1. Enroll 3 samples
async function enrollUser(userId, audioBlobs) {
  for (let i = 0; i < audioBlobs.length; i++) {
    const formData = new FormData();
    formData.append('audio_file', audioBlobs[i]);
    
    const response = await fetch(`${API_BASE}/voice/users/${userId}/enroll`, {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    console.log(`Sample ${i + 1}/3:`, result.message);
    
    if (result.is_complete) {
      console.log('Enrollment complete!');
      break;
    }
  }
}

// 2. Verify user identity
async function verifyUser(userId, audioBlob) {
  const formData = new FormData();
  formData.append('audio_file', audioBlob);
  
  const response = await fetch(`${API_BASE}/voice/users/${userId}/verify`, {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  
  if (result.verified) {
    console.log('✅ Voice verified! Confidence:', result.confidence);
    return true;
  } else {
    console.log('❌ Verification failed:', result.message);
    return false;
  }
}
```

### Real-time STT with Speaker Identification

```javascript
class STTClient {
  constructor(defenseSessionId) {
    this.defenseSessionId = defenseSessionId;
    this.ws = null;
    this.transcriptLines = [];
  }
  
  connect() {
    const url = `wss://your-app.azurewebsites.net/ws/stt?defense_session_id=${this.defenseSessionId}`;
    this.ws = new WebSocket(url);
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'connected':
          console.log('Connected to STT');
          break;
          
        case 'partial':
          // Update UI with interim result
          this.updatePartial(data.text, data.speaker);
          break;
          
        case 'result':
          // Final result - add to transcript
          this.transcriptLines.push({
            text: data.text,
            speaker: data.speaker,
            user_id: data.user_id
          });
          this.updateFinal(data.text, data.speaker);
          break;
          
        case 'cached_transcript':
          // Resume from cache
          this.transcriptLines = data.lines;
          this.restoreTranscript(data.lines);
          break;
          
        case 'ping':
          // Keepalive - ignore
          break;
      }
    };
    
    this.ws.onerror = (error) => console.error('WebSocket error:', error);
    this.ws.onclose = () => console.log('WebSocket closed');
  }
  
  sendAudio(audioChunk) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(audioChunk);
    }
  }
  
  startQuestion() {
    this.ws.send('q:start');
  }
  
  endQuestion() {
    this.ws.send('q:end');
  }
  
  stop() {
    this.ws.send('stop');
  }
}

// Usage
const stt = new STTClient('defense-session-uuid');
stt.connect();

// When recording audio
mediaRecorder.ondataavailable = (event) => {
  stt.sendAudio(event.data);
};
```

### Question Duplicate Detection

```javascript
async function checkQuestion(sessionId, questionText) {
  const response = await fetch(`${API_BASE}/questions/check-and-register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      question_text: questionText
    })
  });
  
  const result = await response.json();
  
  if (result.is_duplicate) {
    console.log('⚠️ Duplicate found!');
    console.log('Similar questions:', result.similar_questions);
    return false;
  }
  
  console.log('✅ Question registered');
  return true;
}
```

---

## 🚀 Deployment

### Azure Container Apps

```yaml
# azure-container-apps.yaml
name: aidefcom-ai-service
properties:
  configuration:
    ingress:
      external: true
      targetPort: 8000
      transport: http
    secrets:
      - name: azure-speech-key
        value: ${AZURE_SPEECH_KEY}
  template:
    containers:
      - name: ai-service
        image: your-registry.azurecr.io/aidefcom-ai-service:latest
        resources:
          cpu: 2.0
          memory: 4Gi
        env:
          - name: AZURE_SPEECH_KEY
            secretRef: azure-speech-key
        probes:
          - type: Liveness
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
          - type: Readiness
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 10
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  ai-service:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

---

## 💻 Development

### Project Structure

```
AIDefCom.AIServer/
├── main.py                 # Entry point (delegates to app.main)
├── app/
│   ├── main.py            # FastAPI app with lifespan, routers
│   └── config.py          # Configuration from environment
├── api/
│   ├── dependencies.py    # Dependency injection
│   ├── routers/
│   │   ├── voice_router.py     # Voice auth endpoints
│   │   ├── speech_router.py    # WebSocket STT
│   │   └── question_router.py  # Question management
│   └── schemas/           # Pydantic models
├── services/
│   ├── voice_service.py   # Voice authentication logic
│   ├── speech_service.py  # STT with speaker identification
│   ├── question_service.py # Duplicate detection
│   └── redis_service.py   # Redis client
├── repositories/
│   ├── azure/             # Azure Blob, Speech
│   ├── sql/               # SQL Server
│   └── voice/             # Voice profile storage
├── core/
│   ├── health.py          # Health check utilities
│   ├── metrics.py         # Prometheus metrics
│   └── exceptions.py      # Custom exceptions
├── data/
│   ├── opposite_keywords.json  # Vietnamese opposite keywords
│   └── phrase_hints.json       # Speech recognition hints
├── tests/                 # Test files
├── Dockerfile
└── requirements.txt
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_voice_auth.py -v

# With coverage
pytest tests/ --cov=services --cov-report=html
```

### Local Development Tips

```bash
# Watch logs
uvicorn main:app --reload --log-level debug

# Test voice enrollment
curl -X POST "http://localhost:8000/voice/users/test_user/enroll" \
  -F "audio_file=@sample.wav"

# Test STT WebSocket (using wscat)
wscat -c "ws://localhost:8000/ws/stt?defense_session_id=test"
```

---

## ⚠️ Error Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `400` | Bad Request (invalid audio, missing params) |
| `404` | Not Found |
| `408` | Request Timeout |
| `429` | Rate Limited |
| `500` | Internal Server Error |
| `503` | Service Unavailable (warmup incomplete) |
| `504` | Gateway Timeout |

### Common Error Messages

| Error | Cause |
|-------|-------|
| `"Empty audio data"` | Audio file is empty |
| `"Audio too large"` | Exceeds 6MB limit |
| `"User not enrolled or insufficient samples"` | User doesn't have 3 required samples |
| `"No enrolled users found"` | No users available for identification |
| `"Audio quality too low"` | RMS/SNR too low |

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 📞 Support

- **Swagger UI:** `/docs`
- **Health Check:** `/health`
- **Metrics:** `/metrics`

---

<div align="center">
Made with ❤️ by AIDefCom Team
</div>
