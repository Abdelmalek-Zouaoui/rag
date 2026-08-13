// ============================================================
// State & element refs
// ============================================================
const sidebar = document.getElementById('sidebar');
const toggleSidebarBtn = document.getElementById('toggleSidebar');
const toggleSidebarTopBtn = document.getElementById('toggleSidebarTop');
const themeToggleBtn = document.getElementById('themeToggle');

const chatWrapper = document.getElementById('chatWrapper');
const welcome = document.getElementById('welcome');
const messagesEl = document.getElementById('messages');

const fileInput = document.getElementById('fileInput');
const attachBtn = document.getElementById('attachBtn');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const uploadStatus = document.getElementById('uploadStatus');
const fileListEl = document.getElementById('fileList');

let isSending = false;

// ============================================================
// Theme
// ============================================================
function initTheme() {
    const saved = localStorage.getItem('theme');
    const theme = saved || 'light';
    applyTheme(theme);
}

function applyTheme(theme) {
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        themeToggleBtn.textContent = '☀️';
    } else {
        document.documentElement.removeAttribute('data-theme');
        themeToggleBtn.textContent = '🌙';
    }
    localStorage.setItem('theme', theme);
}

themeToggleBtn.addEventListener('click', () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    applyTheme(isDark ? 'light' : 'dark');
});

// ============================================================
// Sidebar
// ============================================================
function toggleSidebar() {
    sidebar.classList.toggle('collapsed');
}
toggleSidebarBtn.addEventListener('click', toggleSidebar);
toggleSidebarTopBtn.addEventListener('click', toggleSidebar);

// ============================================================
// File list (sidebar)
// ============================================================
async function refreshFileList() {
    try {
        const res = await fetch('/api/files');
        if (!res.ok) throw new Error('Failed to load files');
        const data = await res.json();
        renderFileList(data.files || []);
    } catch (err) {
        console.error(err);
    }
}

function renderFileList(files) {
    fileListEl.innerHTML = '';

    if (!files.length) {
        const li = document.createElement('li');
        li.className = 'file-empty';
        li.textContent = 'No files uploaded yet.';
        fileListEl.appendChild(li);
        return;
    }

    files.forEach((name) => {
        const li = document.createElement('li');
        li.textContent = `📄 ${name}`;
        fileListEl.appendChild(li);
    });
}

// ============================================================
// Upload
// ============================================================
attachBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async () => {
    if (!fileInput.files.length) return;
    await uploadFiles(fileInput.files);
    fileInput.value = '';
});

async function uploadFiles(fileList) {
    const formData = new FormData();
    Array.from(fileList).forEach((file) => formData.append('files', file));

    setUploadStatus(`Uploading ${fileList.length} file(s)...`, '');

    try {
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || 'Upload failed');
        }

        let msg = `Indexed: ${data.saved.join(', ')}`;
        if (data.skipped && data.skipped.length) {
            msg += ` (skipped unsupported: ${data.skipped.join(', ')})`;
        }
        setUploadStatus(msg, 'success');
        await refreshFileList();
    } catch (err) {
        setUploadStatus(err.message || 'Upload failed', 'error');
    }
}

function setUploadStatus(text, kind) {
    uploadStatus.textContent = text;
    uploadStatus.className = 'upload-status' + (kind ? ` ${kind}` : '');
    if (kind === 'success') {
        setTimeout(() => {
            if (uploadStatus.textContent === text) {
                uploadStatus.textContent = '';
                uploadStatus.className = 'upload-status';
            }
        }, 5000);
    }
}

// ============================================================
// Chat
// ============================================================
function useSuggestion(text) {
    questionInput.value = text;
    sendMessage();
}
window.useSuggestion = useSuggestion;

function showChatView() {
    if (welcome && welcome.style.display !== 'none') {
        welcome.style.display = 'none';
    }
}

function addMessage(role, text) {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;

    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
}

function addTypingIndicator() {
    const wrapper = document.createElement('div');
    wrapper.className = 'message ai';
    wrapper.id = 'typingIndicator';

    const bubble = document.createElement('div');
    bubble.className = 'bubble typing-indicator';
    bubble.innerHTML = '<span></span><span></span><span></span>';

    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    scrollToBottom();
}

function removeTypingIndicator() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
}

function scrollToBottom() {
    chatWrapper.scrollTop = chatWrapper.scrollHeight;
}

async function sendMessage() {
    const text = questionInput.value.trim();
    if (!text || isSending) return;

    showChatView();
    addMessage('user', text);
    questionInput.value = '';
    sendBtn.disabled = true;
    isSending = true;

    addTypingIndicator();

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text }),
        });

        const data = await res.json();
        removeTypingIndicator();

        if (!res.ok) {
            addMessage('ai', `⚠️ ${data.detail || 'Something went wrong.'}`);
        } else {
            addMessage('ai', data.answer);
        }
    } catch (err) {
        removeTypingIndicator();
        addMessage('ai', '⚠️ Could not reach the server. Is it running?');
    } finally {
        sendBtn.disabled = false;
        isSending = false;
        questionInput.focus();
    }
}

sendBtn.addEventListener('click', sendMessage);
questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// ============================================================
// Init
// ============================================================
initTheme();
refreshFileList();