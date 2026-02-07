// Tauri WebSocket handling
const ws = new WebSocket('ws://localhost:8000');
const statusDiv = document.getElementById('status-bar');
const messagesDiv = document.getElementById('messages');
const input = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');

ws.onopen = () => {
  statusDiv.textContent = 'Status: Connected';
  statusDiv.className = 'status-connected';
};

ws.onclose = () => {
  statusDiv.textContent = 'Status: Disconnected';
  statusDiv.className = 'status-disconnected';
};

ws.onmessage = (event) => {
  const message = event.data;
  const msgDiv = document.createElement('div');
  msgDiv.className = 'message message-bot';
  // Attempt to parse JSON if possible, otherwise treat as plain text
  try {
      const data = JSON.parse(message);
      msgDiv.textContent = data.content || data.text || JSON.stringify(data);
  } catch (e) {
      msgDiv.textContent = message;
  }
  messagesDiv.appendChild(msgDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
};

sendBtn.addEventListener('click', sendMessage);
input.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') sendMessage();
});

function sendMessage() {
  const text = input.value.trim();
  if (text) {
      ws.send(JSON.stringify({ type: 'text', content: text }));
      const msgDiv = document.createElement('div');
      msgDiv.className = 'message message-user';
      msgDiv.textContent = text;
      messagesDiv.appendChild(msgDiv);
      input.value = '';
      messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }
}
