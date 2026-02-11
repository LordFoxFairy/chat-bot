/**
 * Chat Bot Desktop Client
 * Handles WebSocket communication, audio recording/playback, and UI
 */

// Configuration
const CONFIG = {
    wsUrl: 'ws://localhost:8765',
    reconnectDelay: 3000,
    maxReconnectAttempts: 10,
    audioSampleRate: 16000,
    audioChannels: 1,
};

// State
let ws = null;
let sessionId = null;
let isRecording = false;
let mediaRecorder = null;
let audioContext = null;
let audioProcessor = null;
let mediaStream = null;
let reconnectAttempts = 0;
let reconnectTimer = null;
let currentBotMessage = null;
let audioQueue = [];
let isPlayingAudio = false;

// DOM Elements
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const messagesArea = document.getElementById('messages');
const asrPreview = document.getElementById('asr-preview');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    setupEventListeners();
});

// WebSocket Connection
function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    updateStatus('connecting', 'Connecting...');

    try {
        ws = new WebSocket(CONFIG.wsUrl);
        ws.binaryType = 'arraybuffer';

        ws.onopen = handleWebSocketOpen;
        ws.onclose = handleWebSocketClose;
        ws.onerror = handleWebSocketError;
        ws.onmessage = handleWebSocketMessage;
    } catch (error) {
        console.error('WebSocket connection error:', error);
        scheduleReconnect();
    }
}

function handleWebSocketOpen() {
    console.log('WebSocket connected');
    updateStatus('connected', 'Connected');
    reconnectAttempts = 0;

    // Send session registration
    sendEvent('SYSTEM_CLIENT_SESSION_START', {});
}

function handleWebSocketClose(event) {
    console.log('WebSocket closed:', event.code, event.reason);
    updateStatus('disconnected', 'Disconnected');
    sessionId = null;
    scheduleReconnect();
}

function handleWebSocketError(error) {
    console.error('WebSocket error:', error);
    updateStatus('disconnected', 'Connection Error');
}

function handleWebSocketMessage(event) {
    if (event.data instanceof ArrayBuffer) {
        // Binary audio data
        handleAudioData(event.data);
        return;
    }

    try {
        const message = JSON.parse(event.data);
        handleStreamEvent(message);
    } catch (error) {
        console.error('Failed to parse message:', error, event.data);
    }
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    if (reconnectAttempts >= CONFIG.maxReconnectAttempts) {
        addSystemMessage('Connection failed. Please restart the application.');
        return;
    }

    reconnectAttempts++;
    updateStatus('connecting', `Reconnecting (${reconnectAttempts}/${CONFIG.maxReconnectAttempts})...`);

    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connectWebSocket();
    }, CONFIG.reconnectDelay);
}

// Event Handling
function handleStreamEvent(event) {
    const { event_type, event_data, session_id, state } = event;

    switch (event_type) {
        case 'SYSTEM_SERVER_SESSION_START':
            sessionId = session_id;
            console.log('Session started:', sessionId);
            addSystemMessage('Session started');
            break;

        case 'SERVER_TEXT_RESPONSE':
            handleTextResponse(event_data);
            break;

        case 'SERVER_AUDIO_RESPONSE':
            handleAudioResponse(event_data);
            break;

        case 'ASR_UPDATE':
            handleASRUpdate(event_data);
            break;

        case 'SERVER_SYSTEM_MESSAGE':
            addSystemMessage(event_data?.text || 'System message');
            break;

        case 'ERROR':
            addErrorMessage(event_data?.text || 'An error occurred');
            break;

        case 'CONFIG_SNAPSHOT':
            handleConfigSnapshot(event_data);
            break;

        case 'MODULE_STATUS_REPORT':
            handleStatusReport(event_data);
            break;

        default:
            console.log('Unknown event type:', event_type, event);
    }
}

function handleTextResponse(data) {
    const text = data?.text || '';
    const isFinal = data?.is_final !== false;

    if (!currentBotMessage) {
        currentBotMessage = addMessage('', 'bot');
    }

    // Append streaming text
    if (isFinal) {
        currentBotMessage.textContent += text;
        currentBotMessage = null;
    } else {
        currentBotMessage.textContent += text;
    }

    scrollToBottom();
}

function handleAudioResponse(data) {
    if (data?.data) {
        // Base64 encoded audio
        const audioBytes = base64ToArrayBuffer(data.data);
        audioQueue.push(audioBytes);
        playNextAudio();
    }
}

function handleASRUpdate(data) {
    const text = data?.text || '';
    const isFinal = data?.is_final;

    if (text) {
        asrPreview.textContent = text;
        asrPreview.classList.remove('hidden');

        if (isFinal) {
            // Add as user message when ASR is final
            addMessage(text, 'user');
            asrPreview.classList.add('hidden');
            asrPreview.textContent = '';
        }
    }
}

function handleAudioData(arrayBuffer) {
    audioQueue.push(arrayBuffer);
    playNextAudio();
}

// Audio Playback
async function playNextAudio() {
    if (isPlayingAudio || audioQueue.length === 0) return;

    isPlayingAudio = true;
    const audioData = audioQueue.shift();

    try {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: CONFIG.audioSampleRate
            });
        }

        // 检测音频格式并解码
        const audioBuffer = await decodeAudioData(audioData);
        if (!audioBuffer) {
            console.error('Failed to decode audio data');
            isPlayingAudio = false;
            playNextAudio();
            return;
        }

        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);

        source.onended = () => {
            isPlayingAudio = false;
            playNextAudio();
        };

        source.start();
    } catch (error) {
        console.error('Audio playback error:', error);
        isPlayingAudio = false;
        playNextAudio();
    }
}

/**
 * 检测并解码音频数据
 * 支持 MP3, WAV, PCM 等格式
 */
async function decodeAudioData(arrayBuffer) {
    // 检测是否是 MP3 格式 (以 ID3 标签或 0xFF 0xFB 开头)
    const header = new Uint8Array(arrayBuffer.slice(0, 4));
    const isMP3 = (header[0] === 0x49 && header[1] === 0x44 && header[2] === 0x33) ||  // ID3
                  (header[0] === 0xFF && (header[1] & 0xE0) === 0xE0);  // MP3 frame sync

    // 检测是否是 WAV 格式 (以 RIFF 开头)
    const isWAV = (header[0] === 0x52 && header[1] === 0x49 && header[2] === 0x46 && header[3] === 0x46);

    if (isMP3 || isWAV) {
        // 使用 Web Audio API 解码压缩音频
        try {
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
            return audioBuffer;
        } catch (error) {
            console.error('Failed to decode compressed audio:', error);
            // 降级到 PCM 处理
            return pcmToAudioBuffer(arrayBuffer);
        }
    } else {
        // 假设是 PCM 16-bit 数据
        return pcmToAudioBuffer(arrayBuffer);
    }
}

function pcmToAudioBuffer(arrayBuffer) {
    const int16Array = new Int16Array(arrayBuffer);
    const float32Array = new Float32Array(int16Array.length);

    for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
    }

    const audioBuffer = audioContext.createBuffer(
        CONFIG.audioChannels,
        float32Array.length,
        CONFIG.audioSampleRate
    );
    audioBuffer.getChannelData(0).set(float32Array);

    return audioBuffer;
}

// Audio Recording
async function startRecording() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: CONFIG.audioSampleRate,
                channelCount: CONFIG.audioChannels,
                echoCancellation: true,
                noiseSuppression: true,
            }
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: CONFIG.audioSampleRate
        });

        const source = audioContext.createMediaStreamSource(mediaStream);
        audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);

        audioProcessor.onaudioprocess = (event) => {
            if (!isRecording) return;

            const inputData = event.inputBuffer.getChannelData(0);
            const pcmData = floatTo16BitPCM(inputData);

            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(pcmData);
            }
        };

        source.connect(audioProcessor);
        audioProcessor.connect(audioContext.destination);

        isRecording = true;
        micBtn.classList.add('recording');
        console.log('Recording started');
    } catch (error) {
        console.error('Failed to start recording:', error);
        addErrorMessage('Microphone access denied');
    }
}

function stopRecording() {
    isRecording = false;
    micBtn.classList.remove('recording');

    if (audioProcessor) {
        audioProcessor.disconnect();
        audioProcessor = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    // Send speech end event
    sendEvent('CLIENT_SPEECH_END', {});

    console.log('Recording stopped');
}

function floatTo16BitPCM(float32Array) {
    const buffer = new ArrayBuffer(float32Array.length * 2);
    const view = new DataView(buffer);

    for (let i = 0; i < float32Array.length; i++) {
        const s = Math.max(-1, Math.min(1, float32Array[i]));
        view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return buffer;
}

// Message Handling
function sendTextMessage(text) {
    if (!text.trim()) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        addErrorMessage('Not connected to server');
        return;
    }

    addMessage(text, 'user');
    sendEvent('CLIENT_TEXT_INPUT', {
        text: text,
        language: 'zh-CN',
        is_final: true
    });

    messageInput.value = '';
}

function sendEvent(eventType, eventData) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const event = {
        event_type: eventType,
        event_data: eventData,
        tag_id: generateId(),
        session_id: sessionId,
        timestamp: Date.now() / 1000
    };

    ws.send(JSON.stringify(event));
}

// UI Updates
function addMessage(text, type) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message message-${type}`;
    msgDiv.textContent = text;

    messagesArea.appendChild(msgDiv);
    scrollToBottom();

    return msgDiv;
}

function addSystemMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message message-system';
    msgDiv.textContent = text;

    messagesArea.appendChild(msgDiv);
    scrollToBottom();
}

function addErrorMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message message-error';
    msgDiv.textContent = text;

    messagesArea.appendChild(msgDiv);
    scrollToBottom();
}

function updateStatus(state, text) {
    statusIndicator.className = `status-indicator ${state}`;
    statusText.textContent = text;
}

function scrollToBottom() {
    messagesArea.scrollTop = messagesArea.scrollHeight;
}

// Event Listeners
function setupEventListeners() {
    sendBtn.addEventListener('click', () => {
        sendTextMessage(messageInput.value);
    });

    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendTextMessage(messageInput.value);
        }
    });

    // Push-to-talk
    micBtn.addEventListener('mousedown', startRecording);
    micBtn.addEventListener('mouseup', stopRecording);
    micBtn.addEventListener('mouseleave', () => {
        if (isRecording) stopRecording();
    });

    // Touch support for mobile
    micBtn.addEventListener('touchstart', (e) => {
        e.preventDefault();
        startRecording();
    });
    micBtn.addEventListener('touchend', (e) => {
        e.preventDefault();
        stopRecording();
    });

    // Configuration Modal
    setupConfigListeners();
}

// Configuration & Status
const configModal = document.getElementById('config-modal');
const settingsBtn = document.getElementById('settings-btn');
const closeConfigBtn = document.getElementById('close-config-btn');
const refreshStatusBtn = document.getElementById('refresh-status-btn');
const reloadConfigBtn = document.getElementById('reload-config-btn');
const saveConfigBtn = document.getElementById('save-config-btn');
const statusGrid = document.getElementById('module-status-grid');
const configContainer = document.getElementById('config-container');

let currentConfig = null;

function setupConfigListeners() {
    // Open modal
    settingsBtn.addEventListener('click', () => {
        configModal.classList.remove('hidden');
        requestConfig();
        requestModuleStatus();
    });

    // Close modal
    closeConfigBtn.addEventListener('click', () => {
        configModal.classList.add('hidden');
    });

    // Close when clicking outside
    configModal.addEventListener('click', (e) => {
        if (e.target === configModal) {
            configModal.classList.add('hidden');
        }
    });

    // Refresh status
    refreshStatusBtn.addEventListener('click', requestModuleStatus);

    // Reload config
    reloadConfigBtn.addEventListener('click', requestConfig);

    // Save config
    saveConfigBtn.addEventListener('click', saveConfig);
}

function requestConfig() {
    sendEvent('CONFIG_GET', {});
    configContainer.innerHTML = '<div class="loading-spinner">Loading configuration...</div>';
}

function requestModuleStatus() {
    sendEvent('MODULE_STATUS_GET', {});

    // Add pulsing effect to existing cards to show loading
    const cards = statusGrid.querySelectorAll('.status-card');
    if (cards.length === 0) {
        // Show skeleton if no cards
        statusGrid.innerHTML = `
            <div class="status-card skeleton"></div>
            <div class="status-card skeleton"></div>
            <div class="status-card skeleton"></div>
            <div class="status-card skeleton"></div>
        `;
    } else {
        cards.forEach(card => card.style.opacity = '0.7');
    }
}

function handleConfigSnapshot(config) {
    currentConfig = config;
    renderConfig(config);
}

function renderConfig(config) {
    configContainer.innerHTML = '';

    // Iterate through sections (e.g., asr, vad, llm, tts)
    for (const [sectionKey, sectionData] of Object.entries(config)) {
        if (typeof sectionData !== 'object' || sectionData === null) continue;

        const group = document.createElement('div');
        group.className = 'config-group';

        const header = document.createElement('div');
        header.className = 'config-group-header';
        header.textContent = sectionKey.toUpperCase();
        group.appendChild(header);

        let hasItems = false;

        // Iterate through items in section
        for (const [key, value] of Object.entries(sectionData)) {
            // Skip large objects or arrays for simple UI
            if (typeof value === 'object' && value !== null) continue;

            hasItems = true;
            const item = document.createElement('div');
            item.className = 'config-item';

            const label = document.createElement('label');
            label.className = 'config-label';
            label.textContent = formatConfigKey(key);
            item.appendChild(label);

            let input;

            if (typeof value === 'boolean') {
                input = document.createElement('select');
                input.className = 'config-select';
                input.innerHTML = `
                    <option value="true" ${value ? 'selected' : ''}>Enabled</option>
                    <option value="false" ${!value ? 'selected' : ''}>Disabled</option>
                `;
                input.dataset.type = 'boolean';
            } else if (typeof value === 'number') {
                input = document.createElement('input');
                input.type = 'number';
                input.className = 'config-input';
                input.value = value;
                input.dataset.type = 'number';
            } else {
                input = document.createElement('input');
                input.type = 'text';
                input.className = 'config-input';
                input.value = value;
                input.dataset.type = 'string';
            }

            input.dataset.section = sectionKey;
            input.dataset.key = key;
            item.appendChild(input);
            group.appendChild(item);
        }

        if (hasItems) {
            configContainer.appendChild(group);
        }
    }

    if (configContainer.children.length === 0) {
        configContainer.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--text-secondary);">No editable configuration found</div>';
    }
}

function formatConfigKey(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function saveConfig() {
    if (!currentConfig) return;

    const newConfig = JSON.parse(JSON.stringify(currentConfig));
    const inputs = configContainer.querySelectorAll('input, select');

    inputs.forEach(input => {
        const section = input.dataset.section;
        const key = input.dataset.key;
        const type = input.dataset.type;

        let value = input.value;

        if (type === 'boolean') {
            value = value === 'true';
        } else if (type === 'number') {
            value = parseFloat(value);
        }

        if (newConfig[section]) {
            newConfig[section][key] = value;
        }
    });

    sendEvent('CONFIG_SET', newConfig);

    // Provide feedback
    saveConfigBtn.textContent = 'Saved!';
    saveConfigBtn.style.backgroundColor = 'var(--success-color)';
    setTimeout(() => {
        saveConfigBtn.textContent = 'Save Changes';
        saveConfigBtn.style.backgroundColor = '';
    }, 2000);
}

function handleStatusReport(status) {
    statusGrid.innerHTML = '';

    for (const [module, state] of Object.entries(status)) {
        const card = document.createElement('div');
        const isRunning = state === 'running' || state === true || (typeof state === 'object' && state.status === 'running');
        const isError = state === 'error' || (typeof state === 'object' && state.status === 'error');

        let statusClass = 'stopped';
        let statusIcon = '●';
        let statusText = 'Stopped';

        if (isRunning) {
            statusClass = 'running';
            statusText = 'Running';
        } else if (isError) {
            statusClass = 'error';
            statusText = 'Error';
            statusIcon = '⚠';
        }

        card.className = `status-card ${statusClass}`;

        const name = module.toUpperCase();

        card.innerHTML = `
            <span class="module-name">${name}</span>
            <div class="module-state">
                <span>${statusIcon}</span>
                <span>${statusText}</span>
            </div>
        `;

        statusGrid.appendChild(card);
    }
}

// Utilities
function generateId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

function base64ToArrayBuffer(base64) {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

// Cleanup on unload
window.addEventListener('beforeunload', () => {
    if (isRecording) stopRecording();
    if (ws) ws.close();
});
