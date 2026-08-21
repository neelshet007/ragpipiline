// Global Application State
let currentMode = 'voice';
let isRecording = false;
let recognition = null;
let websocket = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  initWebSpeech();
  initCanvasWaveform();
  initWebSocket();
  fetchHealthStatus();
});

// Switch between Voice and Text mode
function switchMode(mode) {
  currentMode = mode;
  document.getElementById('btn-mode-voice').classList.toggle('active', mode === 'voice');
  document.getElementById('btn-mode-text').classList.toggle('active', mode === 'text');
  document.getElementById('section-voice').classList.toggle('hidden', mode !== 'voice');
  document.getElementById('section-text').classList.toggle('hidden', mode !== 'text');
}

// Fetch System Health Status
async function fetchHealthStatus() {
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      document.getElementById('badge-status').textContent = `● System Online (${data.bm25_docs} Docs)`;
    }
  } catch (err) {
    document.getElementById('badge-status').textContent = '● System Offline';
    document.getElementById('badge-status').className = 'badge badge-danger';
  }
}

// Web Speech API Initialization
function initWebSpeech() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'hi-IN';

    recognition.onstart = () => {
      isRecording = true;
      document.getElementById('mic-btn').classList.add('recording');
      document.getElementById('voice-status-text').textContent = '🎙️ Listening... Speak now in Hindi or English!';
    };

    recognition.onresult = (event) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      document.getElementById('display-query').textContent = `"${transcript}"`;

      if (event.results[0].isFinal) {
        stopRecording();
        sendQueryPayload(transcript, 'voice');
      }
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      stopRecording();
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        document.getElementById('voice-status-text').textContent = '❌ Microphone permission denied. Please allow mic access in your browser location bar.';
      } else {
        document.getElementById('voice-status-text').textContent = `Speech error: ${event.error}. Try clicking mic again or use Text Mode.`;
      }
    };

    recognition.onend = () => {
      stopRecording();
    };
  } else {
    document.getElementById('voice-status-text').textContent = '⚠️ Web Speech API is not supported in this browser. Please use Chrome / Edge or switch to Text Mode.';
  }
}

// Toggle Mic Recording with Explicit Permission Request
async function toggleRecording() {
  if (isRecording) {
    stopRecording();
    return;
  }

  // Request explicit mic permission first to trigger browser prompt
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      // Stop temporary track right after permission prompt succeeds
      stream.getTracks().forEach(track => track.stop());
    } catch (err) {
      console.error('Microphone access denied:', err);
      document.getElementById('voice-status-text').textContent = '❌ Microphone permission denied. Click the lock/tune icon in the browser address bar to allow mic access.';
      alert('Microphone permission is required for Voice Mode. Please allow mic access in your browser address bar.');
      return;
    }
  }

  if (recognition) {
    try {
      recognition.start();
    } catch (e) {
      console.error('Failed to start speech recognition:', e);
      stopRecording();
    }
  } else {
    alert('Web Speech API is not supported in your browser. Please switch to Text Mode.');
  }
}

function stopRecording() {
  isRecording = false;
  document.getElementById('mic-btn').classList.remove('recording');
  document.getElementById('voice-status-text').textContent = 'Click the microphone button to start voice input';
  if (recognition) {
    try {
      recognition.stop();
    } catch (e) {}
  }
}

// Canvas Waveform Animation
function initCanvasWaveform() {
  const canvas = document.getElementById('waveform-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let step = 0;

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = isRecording ? '#00e676' : '#00f2fe';

    const height = canvas.height;
    const width = canvas.width;
    const amplitude = isRecording ? 30 : 5;

    for (let x = 0; x < width; x++) {
      const y = height / 2 + Math.sin((x + step) * 0.05) * amplitude * Math.sin(x * 0.01);
      if (x === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    step += 2;
    requestAnimationFrame(draw);
  }
  draw();
}

// Submit Text Query
function submitTextQuery() {
  const input = document.getElementById('text-query-input');
  const query = input.value.trim();
  if (!query) return;

  document.getElementById('display-query').textContent = `"${query}"`;
  sendQueryPayload(query, 'text');
  input.value = '';
}

// Initialize WebSocket Connection
function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/rag`;

  try {
    websocket = new WebSocket(wsUrl);
    websocket.onopen = () => console.log('[+] WebSocket connected to /ws/rag');
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.status === 'completed') {
        renderRAGResponse(data);
      }
    };
    websocket.onclose = () => console.log('[-] WebSocket closed');
  } catch (err) {
    console.log('[-] WebSocket connection failed');
  }
}

// Dispatch Query Payload via REST or WebSocket
async function sendQueryPayload(queryText, mode) {
  const protocolSelect = document.getElementById('protocol');
  const protocol = protocolSelect ? protocolSelect.value : 'rest';
  document.getElementById('display-answer').textContent = 'Synthesizing response...';

  if (protocol === 'ws' && websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.send(JSON.stringify({
      type: 'text',
      payload: queryText,
      language: 'hi',
      top_k: 3
    }));
    return;
  }

  // REST API Execution
  try {
    const endpoint = mode === 'voice' ? '/api/v1/voice' : '/api/v1/query';
    const body = mode === 'voice' 
      ? { text_transcript: queryText, language: 'hi', top_k: 3 }
      : { query: queryText, top_k: 3, fusion_mode: 'rrf' };

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (response.ok) {
      const data = await response.json();
      renderRAGResponse(data);
    } else {
      document.getElementById('display-answer').textContent = 'Error: Failed to process request.';
    }
  } catch (err) {
    document.getElementById('display-answer').textContent = `Error: ${err.message}`;
  }
}

// Render RAG Core Response & Metrics
function renderRAGResponse(data) {
  // Render Answer & Query
  document.getElementById('display-answer').textContent = data.answer;
  if (data.query) {
    document.getElementById('display-query').textContent = `"${data.query}"`;
  }

  // Update Latency Metrics Dashboard
  const lat = data.latency || {};
  document.getElementById('val-stt').textContent = `${(lat.stt_ms || 0.0).toFixed(1)} ms`;
  document.getElementById('val-embed').textContent = `${(lat.embed_ms || 0.0).toFixed(1)} ms`;
  document.getElementById('val-retrieval').textContent = `${(lat.retrieval_ms || 0.0).toFixed(1)} ms`;
  document.getElementById('val-tts').textContent = `${(lat.tts_ms || 0.0).toFixed(1)} ms`;

  const totalMs = (lat.total_pipeline_ms || 0.0).toFixed(1);
  document.getElementById('val-total').textContent = `${totalMs} ms`;

  // Update Sub-200ms Compliance Badge
  const compBadge = document.getElementById('badge-compliance');
  if (data.sub_200ms_target_met) {
    compBadge.textContent = `Sub-200ms Target Met (${totalMs} ms)`;
    compBadge.className = 'badge badge-success';
  } else {
    compBadge.textContent = `Sub-200ms Target: Exceeded (${totalMs} ms)`;
    compBadge.className = 'badge badge-danger';
  }

  // Play Synthesized Audio Response
  if (data.audio_base64) {
    const audioContainer = document.getElementById('audio-container');
    const audioPlayer = document.getElementById('tts-audio-player');
    const format = data.audio_format || 'mp3';
    audioPlayer.src = `data:audio/${format};base64,${data.audio_base64}`;
    audioContainer.classList.remove('hidden');
    audioPlayer.play().catch(e => console.log('Audio autoplay prevented by browser'));
  }

  // Render Source Document Passages
  const sourcesContainer = document.getElementById('display-sources');
  sourcesContainer.innerHTML = '';

  const sources = data.sources || [];
  document.getElementById('sources-count').textContent = `${sources.length} Passages`;

  if (sources.length === 0) {
    sourcesContainer.innerHTML = '<div class="empty-state">No context passages retrieved for this query.</div>';
    return;
  }

  sources.forEach((src) => {
    const itemDiv = document.createElement('div');
    itemDiv.className = 'source-item';
    const scoreVal = src.score ? src.score.toFixed(4) : 'N/A';

    itemDiv.innerHTML = `
      <div class="source-meta">
        <span>Rank #${src.rank} — Doc ID: ${src.document_id}</span>
        <span>Relevance Score: ${scoreVal}</span>
      </div>
      <div class="source-text">${src.chunk_text}</div>
    `;
    sourcesContainer.appendChild(itemDiv);
  });
}
