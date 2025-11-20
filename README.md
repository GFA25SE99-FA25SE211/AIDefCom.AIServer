# API Documentation - AIDefCom AI Service

**Base URL:** `https://<your-app>.azurewebsites.net`  
**Version:** 2.3.3  
**Swagger UI:** `https://<your-app>.azurewebsites.net/docs`

---

## 📋 Table of Contents
1. [Voice Authentication APIs](#voice-authentication-apis)
2. [Speech-to-Text WebSocket](#speech-to-text-websocket)
3. [Question Management APIs](#question-management-apis)
4. [Health Check](#health-check)
5. [Response Format Standards](#response-format-standards)
6. [Error Codes](#error-codes)

---

## 🎤 Voice Authentication APIs

### 1. Enroll Voice Sample
**Endpoint:** `POST /voice/users/{user_id}/enroll`

Đăng ký mẫu giọng nói cho một user. Cần tối thiểu **3 mẫu** để hoàn tất enrollment.

#### Request
```http
POST /voice/users/{user_id}/enroll
Content-Type: multipart/form-data

audio_file: <binary audio file>
```

**Parameters:**
- `user_id` (path, required): User ID cần enroll
- `audio_file` (form-data, required): File audio (WAV/MP3/FLAC, max 10MB, khuyến nghị 3-5 giây)

#### Response
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

**Response Fields:**
- `success` (boolean): Thành công hay không
- `user_id` (string): User ID đã enroll
- `enrollment_count` (int): Số mẫu đã có
- `min_required` (int): Số mẫu tối thiểu cần (3)
- `is_complete` (boolean): Đã đủ 3 mẫu chưa
- `message` (string): Thông báo chi tiết

#### Status Codes
- `200`: Enrollment thành công
- `400`: Audio không hợp lệ hoặc chất lượng kém
- `500`: Lỗi server

---

### 2. Identify Speaker
**Endpoint:** `POST /voice/identify`

Nhận diện người nói từ mẫu giọng nói (so sánh với tất cả users đã enroll).

#### Request
```http
POST /voice/identify
Content-Type: multipart/form-data

audio_file: <binary audio file>
```

**Parameters:**
- `audio_file` (form-data, required): File audio cần nhận diện (WAV/MP3/FLAC, max 10MB)

#### Response (Success - Identified)
```json
{
  "type": "identification",
  "success": true,
  "identified": true,
  "speaker_id": "USR001",
  "speaker_name": "Nguyen Van A",
  "confidence": 0.92,
  "score": 0.92,
  "message": "Speaker identified successfully"
}
```

#### Response (No Match)
```json
{
  "type": "identification",
  "success": true,
  "identified": false,
  "speaker_id": null,
  "speaker_name": null,
  "confidence": 0.0,
  "score": 0.58,
  "message": "No matching speaker found"
}
```

**Response Fields:**
- `identified` (boolean): Có nhận diện được hay không
- `speaker_id` (string|null): User ID của người được nhận diện
- `speaker_name` (string|null): Tên hiển thị
- `confidence` (float): Độ tin cậy (0-1)
- `score` (float): Điểm tương đồng thực tế (0-1)
- `message` (string): Thông báo

**Threshold:** Score >= 0.7 mới được coi là match

#### Status Codes
- `200`: Process thành công (check `identified` field)
- `400`: Audio không hợp lệ hoặc không có users nào đã enroll
- `500`: Lỗi server

---

### 3. Verify Voice
**Endpoint:** `POST /voice/users/{user_id}/verify`

Xác thực xem mẫu giọng có khớp với user ID đã claim hay không (1:1 verification).

#### Request
```http
POST /voice/users/{user_id}/verify
Content-Type: multipart/form-data

audio_file: <binary audio file>
```

**Parameters:**
- `user_id` (path, required): User ID cần verify
- `audio_file` (form-data, required): File audio để verify (WAV/MP3/FLAC, max 10MB)

#### Response (Verified - Match)
```json
{
  "type": "verification",
  "success": true,
  "verified": true,
  "claimed_id": "USR001",
  "speaker_id": "USR001",
  "match": true,
  "confidence": 0.89,
  "score": 0.89,
  "message": "Voice verified successfully"
}
```

#### Response (Not Verified - No Match)
```json
{
  "type": "verification",
  "success": false,
  "verified": false,
  "claimed_id": "USR001",
  "speaker_id": "USR002",
  "match": false,
  "confidence": 0.65,
  "score": 0.65,
  "message": "Voice verification failed - speaker mismatch"
}
```

#### Response (User Not Enrolled)
```json
{
  "type": "verification",
  "success": false,
  "verified": false,
  "claimed_id": "USR999",
  "speaker_id": null,
  "match": false,
  "confidence": 0.0,
  "score": 0.0,
  "message": "User not enrolled or insufficient samples"
}
```

**Response Fields:**
- `verified` (boolean): Có verify thành công không
- `claimed_id` (string): User ID được claim
- `speaker_id` (string|null): User ID thực sự nhận diện được
- `match` (boolean): `claimed_id == speaker_id`
- `confidence` (float): Độ tin cậy (0-1)
- `score` (float): Điểm tương đồng (0-1)
- `message` (string): Thông báo

**Use Cases:**
- Authentication: Xác thực user qua giọng nói
- Access Control: Cấp quyền truy cập nếu voice khớp
- Security: Phát hiện giả mạo giọng nói

#### Status Codes
- `200`: Process thành công (check `verified` field)
- `400`: Audio không hợp lệ hoặc user chưa enroll đủ
- `500`: Lỗi server

---

## 🎙️ Speech-to-Text WebSocket

### WebSocket Endpoint
**Endpoint:** `ws://<host>/ws/stt` (hoặc `wss://` cho HTTPS)

Real-time speech-to-text streaming với Azure Cognitive Services.

#### Connection
```javascript
const ws = new WebSocket('wss://<your-app>.azurewebsites.net/ws/stt');

ws.onopen = () => {
  console.log('Connected to STT WebSocket');
  
  // Send initialization (optional)
  ws.send(JSON.stringify({
    session_id: "session_123",
    lang: "vi-VN"
  }));
  
  // Backend sẽ TỰ ĐỘNG nhận diện người nói từ audio
  // Không cần gửi user_id hay speaker name
};
```

#### Initialization Message (Optional)
Sau khi connect, FE có thể gửi JSON message để config:

```json
{
  "session_id": "session_123",
  "lang": "vi-VN",
  "phrases": "AI,Machine Learning,Deep Learning"
}
```

**Fields:**
- `session_id` (string, optional): Session ID để group transcripts
- `lang` (string, optional): Language code (default: "vi-VN")
- `phrases` (string, optional): Phrase hints phân cách bằng dấu phẩy

**⚠️ Lưu ý quan trọng:**
- **KHÔNG cần gửi `user_id` hay `speaker`** - Backend sẽ tự động nhận diện người nói từ audio bằng voice identification
- Speaker name và user_id sẽ được trả về trong events `recognized`

#### Sending Audio
```javascript
// Send audio chunks as binary data
const audioBlob = new Blob([audioData], { type: 'audio/wav' });
ws.send(audioBlob);

// Or send raw audio buffer
ws.send(audioBuffer);
```

#### Automatic Speaker Identification
**Backend tự động nhận diện người nói:**

1. **Trong quá trình stream**, backend sẽ:
   - Thu thập audio chunks từ FE
   - Định kỳ (mỗi 0.6s) chạy voice identification
   - So sánh với database users đã enroll (≥3 samples)
   - Tự động gán `speaker` và `user_id` vào events

2. **FE nhận kết quả qua events:**
   ```json
   {
     "event": "recognized",
     "text": "Xin chào các bạn",
     "speaker": "Nguyen Van A",
     "user_id": "USR001",
     "timestamp": "2025-11-16T10:30:05Z"
   }
   ```

3. **Nếu không nhận diện được:**
   ```json
   {
     "event": "recognized",
     "text": "Xin chào các bạn",
     "speaker": "Khách",
     "timestamp": "2025-11-16T10:30:05Z"
   }
   ```

**Ưu điểm:**
- FE không cần biết trước user_id
- Tự động phát hiện khi người nói thay đổi
- Support multi-speaker trong cùng session

#### Receiving Events
```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.event) {
    case 'recognizing':
      // Interim result (real-time)
      console.log('Recognizing:', data.text);
      break;
      
    case 'recognized':
      // Final result for segment
      console.log('Recognized:', data.text);
      console.log('Speaker:', data.speaker);
      break;
      
    case 'session_started':
      console.log('Session ID:', data.session_id);
      break;
      
    case 'session_stopped':
      console.log('Total lines:', data.total_lines);
      break;
      
    case 'error':
      console.error('Error:', data.message);
      break;
  }
};
```

#### Event Types

**1. session_started**
```json
{
  "event": "session_started",
  "session_id": "abc123",
  "timestamp": "2025-11-16T10:30:00Z"
}
```

**2. recognizing** (real-time interim results)
```json
{
  "event": "recognizing",
  "text": "Xin chào các bạn",
  "speaker": "Nguyen Van A",
  "timestamp": "2025-11-16T10:30:05Z"
}
```

**3. recognized** (final segment result)
```json
{
  "event": "recognized",
  "text": "Xin chào các bạn, hôm nay chúng ta sẽ học về AI.",
  "speaker": "Nguyen Van A",
  "user_id": "USR001",
  "timestamp": "2025-11-16T10:30:08Z"
}
```

**Fields:**
- `speaker` (string): Tên người nói (tự động identify)
- `user_id` (string, optional): User ID nếu nhận diện được
- Nếu không nhận diện được: `speaker="Khách"`, không có `user_id`

**4. session_stopped**
```json
{
  "event": "session_stopped",
  "session_id": "abc123",
  "total_lines": 15,
  "message": "Session ended and transcript saved"
}
```

**5. error**
```json
{
  "event": "error",
  "message": "Audio stream interrupted",
  "code": "AUDIO_ERROR"
}
```

#### Ending Session
```javascript
// Send "stop" command
ws.send("stop");

// Or close connection
ws.close();
```

**Note:** Khi kết thúc session, transcript sẽ tự động được lưu vào external API `/api/transcripts`.

#### Audio Requirements
- **Format:** PCM 16-bit, mono
- **Sample Rate:** 16000 Hz
- **Chunk Size:** 3200-6400 bytes (0.1-0.2s)
- **Max Total Size:** No limit (streaming)

---

## ❓ Question Management APIs

### 1. Check Duplicate Question
**Endpoint:** `POST /questions/check-duplicate`

Kiểm tra xem câu hỏi có bị trùng lặp trong session hay không.

#### Request
```json
{
  "session_id": "session_123",
  "question_text": "AI là gì?",
  "threshold": 0.85
}
```

**Fields:**
- `session_id` (string, required): Session ID
- `question_text` (string, required): Nội dung câu hỏi
- `threshold` (float, optional): Ngưỡng tương đồng (default: 0.85)

#### Response (Not Duplicate)
```json
{
  "is_duplicate": false,
  "question_text": "AI là gì?",
  "similar_questions": [],
  "message": "✅ Câu hỏi hợp lệ, chưa bị trùng."
}
```

#### Response (Duplicate Found)
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
  "message": "⚠️ Câu hỏi trùng lặp! Tìm thấy 1 câu tương tự."
}
```

---

### 2. Register Question
**Endpoint:** `POST /questions/register`

Đăng ký câu hỏi mới vào session (không check duplicate).

#### Request
```json
{
  "session_id": "session_123",
  "question_text": "Machine Learning hoạt động thế nào?",
  "speaker": "Nguyen Van A",
  "timestamp": "2025-11-16T10:30:00Z"
}
```

**Fields:**
- `session_id` (string, required): Session ID
- `question_text` (string, required): Nội dung câu hỏi
- `speaker` (string, optional): Người hỏi
- `timestamp` (string, optional): Thời gian hỏi (ISO format)

#### Response
```json
{
  "success": true,
  "question_id": "q_abc123",
  "total_questions": 5,
  "message": "✅ Câu hỏi đã được lưu. Tổng: 5"
}
```

---

### 3. Check and Register (Combo)
**Endpoint:** `POST /questions/check-and-register`

Check duplicate + register nếu không trùng (một bước).

#### Request
```json
{
  "session_id": "session_123",
  "question_text": "Deep Learning khác gì Machine Learning?",
  "speaker": "Tran Thi B",
  "timestamp": "2025-11-16T10:35:00Z"
}
```

#### Response (Registered)
```json
{
  "is_duplicate": false,
  "question_text": "Deep Learning khác gì Machine Learning?",
  "similar_questions": [],
  "message": "✅ Câu hỏi đã được lưu. Tổng: 6"
}
```

#### Response (Duplicate - Not Registered)
```json
{
  "is_duplicate": true,
  "question_text": "Deep Learning khác gì Machine Learning?",
  "similar_questions": [
    {
      "text": "Sự khác biệt giữa Deep Learning và ML?",
      "score": 0.89,
      "fuzzy_score": 0.82,
      "semantic_score": 0.89
    }
  ],
  "message": "⚠️ Câu hỏi trùng lặp! Không thể đăng ký."
}
```

---

### 4. Get Session Questions
**Endpoint:** `GET /questions/session/{session_id}`

Lấy tất cả câu hỏi trong một session.

#### Request
```http
GET /questions/session/session_123
```

#### Response
```json
{
  "session_id": "session_123",
  "questions": [
    {
      "id": "q_001",
      "text": "AI là gì?",
      "speaker": "Nguyen Van A",
      "timestamp": "2025-11-16T10:30:00Z"
    },
    {
      "id": "q_002",
      "text": "Machine Learning hoạt động thế nào?",
      "speaker": "Nguyen Van A",
      "timestamp": "2025-11-16T10:32:00Z"
    }
  ],
  "total": 2
}
```

---

### 5. Clear Session Questions
**Endpoint:** `DELETE /questions/session/{session_id}`

Xóa tất cả câu hỏi trong session.

#### Request
```http
DELETE /questions/session/session_123
```

#### Response
```json
{
  "success": true,
  "session_id": "session_123",
  "deleted": 5,
  "message": "✅ Đã xóa 5 câu hỏi."
}
```

---

## ❤️ Health Check

### Health Endpoint
**Endpoint:** `GET /health`

Kiểm tra trạng thái server.

#### Request
```http
GET /health
```

#### Response
```json
{
  "status": "ok"
}
```

**Legacy Endpoint:** `GET /healthz` (giữ để tương thích)

---

## 📦 Response Format Standards

### Success Response (Voice Auth)
```json
{
  "type": "enrollment|identification|verification",
  "success": true,
  "user_id": "USR001",
  "speaker_id": "USR001",
  "confidence": 0.92,
  "score": 0.92,
  "message": "Success message"
}
```

### Error Response
```json
{
  "error": "Error message description",
  "detail": "Technical detail (optional)"
}
```

### Question Response
```json
{
  "is_duplicate": false,
  "question_text": "Question content",
  "similar_questions": [],
  "message": "Status message"
}
```

---

## ⚠️ Error Codes

| Status Code | Description | Common Causes |
|-------------|-------------|---------------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid audio, missing parameters, validation failed |
| 404 | Not Found | Endpoint không tồn tại |
| 500 | Internal Server Error | Server error, service unavailable |

### Common Error Messages

#### Voice Authentication
- `"Empty audio data"` - File audio rỗng
- `"Audio too large (>10MB)"` - File quá lớn
- `"User not enrolled or insufficient samples"` - User chưa enroll đủ 3 mẫu
- `"No enrolled users found"` - Không có user nào đã enroll (identify)
- `"Audio quality too low"` - Chất lượng audio không đủ (quá nhỏ, nhiễu, v.v.)

#### WebSocket
- `"Audio stream interrupted"` - Kết nối audio bị gián đoạn
- `"Session initialization failed"` - Không thể khởi tạo session
- `"Recognition error"` - Lỗi nhận dạng giọng nói

#### Questions
- `"Invalid session_id"` - Session ID không hợp lệ
- `"Question text is required"` - Thiếu nội dung câu hỏi
- `"Service unavailable"` - Redis hoặc semantic service không khả dụng

---

## 🔐 Authentication & Security

**Current Status:** No authentication required (internal/trusted network)

**Production Recommendations:**
1. Add API Key authentication
2. Implement rate limiting
3. Enable CORS restrictions (currently `*`)
4. Use HTTPS only
5. Add request signing for voice samples

---

## 🌐 Environment Variables (FE cần biết)

Frontend nên config các URL sau:

```javascript
// Production
const API_BASE_URL = 'https://<your-app>.azurewebsites.net';
const WS_BASE_URL = 'wss://<your-app>.azurewebsites.net';

// Development (local)
const API_BASE_URL = 'http://localhost:8000';
const WS_BASE_URL = 'ws://localhost:8000';
```

---

## 📝 Integration Examples

### Voice Authentication Flow

```javascript
// 1. Enroll user (3 samples)
for (let i = 0; i < 3; i++) {
  const formData = new FormData();
  formData.append('audio_file', audioBlob);
  
  const response = await fetch(`${API_BASE_URL}/voice/users/USR001/enroll`, {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  console.log(`Sample ${i+1}/3:`, result.message);
}

// 2. Verify user
const formData = new FormData();
formData.append('audio_file', audioBlob);

const verifyResponse = await fetch(`${API_BASE_URL}/voice/users/USR001/verify`, {
  method: 'POST',
  body: formData
});

const verifyResult = await verifyResponse.json();
if (verifyResult.verified) {
  console.log('Authentication successful!');
} else {
  console.log('Authentication failed:', verifyResult.message);
}
```

### Question Management Flow

```javascript
// Check and register question
const checkResponse = await fetch(`${API_BASE_URL}/questions/check-and-register`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: 'session_123',
    question_text: 'AI là gì?',
    speaker: 'Nguyen Van A',
    timestamp: new Date().toISOString()
  })
});

const result = await checkResponse.json();
if (result.is_duplicate) {
  alert('Câu hỏi bị trùng!');
  console.log('Similar:', result.similar_questions);
} else {
  alert('Câu hỏi đã được lưu!');
}
```

---

## 🚀 Quick Start Guide

### Step 1: Health Check
```bash
curl https://<your-app>.azurewebsites.net/health
```

### Step 2: View API Docs
Open browser: `https://<your-app>.azurewebsites.net/docs`

### Step 3: Test Voice Enroll
```bash
curl -X POST "https://<your-app>.azurewebsites.net/voice/users/USR001/enroll" \
  -F "audio_file=@sample.wav"
```

### Step 4: Test Question Check
```bash
curl -X POST "https://<your-app>.azurewebsites.net/questions/check-duplicate" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","question_text":"AI là gì?"}'
```

---

## 📞 Support

- **Swagger UI:** `/docs`
- **Health Check:** `/health`
- **Base Info:** `GET /` (root endpoint)

**Note:** Tất cả endpoints đều support CORS `*` (hiện tại). Production nên restrict lại.
