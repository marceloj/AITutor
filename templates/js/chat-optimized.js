// Chat functionality optimized
class ChatManager {
    constructor() {
        this.chatArea = document.querySelector('.flex-1.overflow-y-auto');
        this.timerDiv = document.getElementById('response-timer');
        this.agentMode = false;
        this.canvasMode = false;
        
        // Cache DOM elements
        this.reliabilityScore = document.getElementById('reliabilityScore');
        this.reliabilityBar = document.getElementById('reliabilityBar');
        this.coherenceScore = document.getElementById('coherenceScore');
        this.coherenceBar = document.getElementById('coherenceBar');
        this.fluencyScore = document.getElementById('fluencyScore');
        this.fluencyBar = document.getElementById('fluencyBar');
        this.groundednessScore = document.getElementById('groundednessScore');
        this.groundednessBar = document.getElementById('groundednessBar');
        this.relevanceScore = document.getElementById('relevanceScore');
        this.relevanceBar = document.getElementById('relevanceBar');
        this.retrievalScore = document.getElementById('retrievalScore');
        this.retrievalBar = document.getElementById('retrievalBar');
        
        this.init();
    }
    
    init() {
        this.loadChatHistory();
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Clear chat button
        const clearBtn = document.getElementById('clearChatBtn');
        if (clearBtn) {
            clearBtn.onclick = () => this.clearChat();
        }
    }
    
    updateStatusBar(msg) {
        console.log(msg); // Simple status update
    }
    
    // Optimized message sending with better error handling
    async sendMessage(message) {
        if (!message.trim()) return;
        
        this.appendMessage(`User: ${message}`, 'user-bubble');
        
        const start = Date.now();
        let timerInterval;
        
        if (this.timerDiv) {
            this.timerDiv.innerHTML = '<span class="spinner"></span>Esperando respuesta...';
            this.timerDiv.style.display = 'block';
            
            timerInterval = setInterval(() => {
                const elapsed = ((Date.now() - start) / 1000).toFixed(1);
                this.timerDiv.innerHTML = `<span class="spinner"></span>Esperando respuesta... (${elapsed}s)`;
            }, 100);
        }
        
        try {
            const formData = new FormData();
            formData.append('message', message);
            formData.append('agent_mode', this.agentMode ? 'on' : 'off');
            
            const response = await fetch('/chat', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            await this.processStreamResponse(response, start);
            
        } catch (error) {
            console.error('Error:', error);
            if (this.timerDiv) {
                this.timerDiv.innerHTML = '❌ Error en la respuesta';
                setTimeout(() => { this.timerDiv.style.display = 'none'; }, 3500);
            }
            this.appendMessage("Error: No se pudo obtener respuesta", "error-bubble");
        } finally {
            if (timerInterval) clearInterval(timerInterval);
        }
    }
    
    // Optimized stream processing
    async processStreamResponse(response, startTime) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let responseContent = '';
        let placeholderMsg = null;
        
        // Create placeholder message
        if (!this.canvasMode && this.chatArea) {
            placeholderMsg = this.createPlaceholderMessage();
        }
        
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                buffer += chunk;
                
                // Process scores and evaluations
                this.processScoresAndEvaluations(buffer);
                
                responseContent += buffer;
                
                // Update placeholder content
                if (placeholderMsg && !this.canvasMode) {
                    this.updatePlaceholderContent(placeholderMsg, responseContent);
                }
                
                buffer = '';
            }
            
            // Finalize response
            if (this.timerDiv) {
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                this.timerDiv.innerHTML = `✅ Respondió en ${elapsed}s`;
                setTimeout(() => { this.timerDiv.style.display = 'none'; }, 3500);
            }
            
        } catch (error) {
            console.error('Stream processing error:', error);
            throw error;
        }
    }
    
    createPlaceholderMessage() {
        const wrapper = document.createElement('div');
        wrapper.className = 'flex items-start gap-4 mb-4';
        
        const avatar = document.createElement('div');
        avatar.className = 'w-10 h-10 rounded-full bg-cover bg-center shrink-0';
        avatar.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
        
        const bubble = document.createElement('div');
        bubble.className = 'flex flex-col';
        
        const name = document.createElement('span');
        name.className = 'font-bold text-[var(--text-primary)] mb-1';
        name.textContent = 'AI Tutor';
        
        const msgDiv = document.createElement('div');
        msgDiv.className = 'chat-bubble tutor-bubble';
        msgDiv.style.minHeight = '60px';
        msgDiv.innerHTML = '<div class="markdown-body">Procesando...</div>';
        
        bubble.appendChild(name);
        bubble.appendChild(msgDiv);
        wrapper.appendChild(avatar);
        wrapper.appendChild(bubble);
        
        if (this.chatArea) {
            this.chatArea.appendChild(wrapper);
            this.chatArea.scrollTop = this.chatArea.scrollHeight;
        }
        
        return { wrapper, msgDiv };
    }
    
    updatePlaceholderContent(placeholderMsg, content) {
        if (!placeholderMsg || !placeholderMsg.msgDiv) return;
        
        // Clean content from think tags and scores
        const cleanContent = content
            .replace(/<think>[\s\S]*?<\/think>/g, '')
            .replace(/<think>[\s\S]*/g, '')
            .replace(/\[SCORE:[^\]]*\]/g, '')
            .replace(/\[EVAL:[^\]]*\]/g, '');
        
        const markdownDiv = placeholderMsg.msgDiv.querySelector('.markdown-body');
        if (markdownDiv && cleanContent.trim()) {
            markdownDiv.innerHTML = marked.parse(cleanContent);
        }
    }
    
    processScoresAndEvaluations(buffer) {
        // Process reliability score
        const scoreMatch = buffer.match(/\[SCORE:(\d{1,3})\]/);
        if (scoreMatch) {
            const score = parseInt(scoreMatch[1]);
            this.updateScoreDisplay(this.reliabilityScore, this.reliabilityBar, score);
            window.lastReliabilityScore = score;
        }
        
        // Process multi-evaluation
        const evalMatch = buffer.match(/\[EVAL:(\d{1,3}),(\d{1,3}),(\d{1,3}),(\d{1,3}),(\d{1,3})\]/);
        if (evalMatch) {
            const [_, coherence, fluency, groundedness, relevance, retrieval] = evalMatch;
            this.updateScoreDisplay(this.coherenceScore, this.coherenceBar, coherence);
            this.updateScoreDisplay(this.fluencyScore, this.fluencyBar, fluency);
            this.updateScoreDisplay(this.groundednessScore, this.groundednessBar, groundedness);
            this.updateScoreDisplay(this.relevanceScore, this.relevanceBar, relevance);
            this.updateScoreDisplay(this.retrievalScore, this.retrievalBar, retrieval);
        }
    }
    
    updateScoreDisplay(scoreElement, barElement, value) {
        if (scoreElement && barElement) {
            scoreElement.textContent = value + '%';
            barElement.style.width = value + '%';
        }
    }
    
    // Optimized message appending
    appendMessage(content, className) {
        // Save to localStorage
        const chatHistory = JSON.parse(localStorage.getItem('chatHistory') || '[]');
        chatHistory.push({ content, className, timestamp: Date.now() });
        localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
        
        if (!this.chatArea) return;
        
        const wrapper = document.createElement('div');
        wrapper.className = 'flex items-start gap-4 mb-4';
        
        const avatar = document.createElement('div');
        avatar.className = 'w-10 h-10 rounded-full bg-cover bg-center shrink-0';
        
        const bubble = document.createElement('div');
        bubble.className = 'flex flex-col';
        
        const name = document.createElement('span');
        name.className = 'font-bold text-[var(--text-primary)] mb-1';
        
        // Configure based on message type
        if (className.includes('user')) {
            avatar.style.background = 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)';
            name.textContent = 'User';
        } else {
            avatar.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            name.textContent = 'AI Tutor';
        }
        
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-bubble ${className}`;
        msgDiv.innerHTML = marked.parse(content);
        
        // Add action buttons for tutor messages
        if (className.includes('tutor')) {
            this.addMessageActions(bubble, content);
        }
        
        bubble.appendChild(name);
        bubble.appendChild(msgDiv);
        wrapper.appendChild(avatar);
        wrapper.appendChild(bubble);
        
        this.chatArea.appendChild(wrapper);
        this.chatArea.scrollTop = this.chatArea.scrollHeight;
    }
    
    addMessageActions(bubble, content) {
        const actions = document.createElement('div');
        actions.className = 'flex gap-2 mt-2';
        
        // Copy button
        const copyBtn = document.createElement('button');
        copyBtn.innerHTML = '📋';
        copyBtn.title = 'Copiar respuesta';
        copyBtn.className = 'p-2 rounded hover:bg-gray-200 transition-colors';
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(content);
            this.updateStatusBar('Respuesta copiada');
        };
        
        // Speak button
        const speakBtn = document.createElement('button');
        speakBtn.innerHTML = '🔊';
        speakBtn.title = 'Escuchar respuesta';
        speakBtn.className = 'p-2 rounded hover:bg-gray-200 transition-colors';
        speakBtn.onclick = () => {
            const utterance = new SpeechSynthesisUtterance(content);
            utterance.lang = 'es-ES';
            speechSynthesis.speak(utterance);
            this.updateStatusBar('Reproduciendo respuesta...');
        };
        
        actions.appendChild(copyBtn);
        actions.appendChild(speakBtn);
        bubble.appendChild(actions);
    }
    
    // Optimized chat history loading with batching
    loadChatHistory() {
        if (!this.chatArea) return;
        this.chatArea.innerHTML = '';
        
        const chatHistory = JSON.parse(localStorage.getItem('chatHistory') || '[]');
        
        // Limit history to prevent performance issues
        const limitedHistory = chatHistory.slice(-50);
        if (limitedHistory.length !== chatHistory.length) {
            localStorage.setItem('chatHistory', JSON.stringify(limitedHistory));
        }
        
        // Load messages in batches
        this.loadMessageBatch(limitedHistory, 0, 10);
    }
    
    loadMessageBatch(messages, startIndex, batchSize) {
        if (startIndex >= messages.length) return;
        
        const endIndex = Math.min(startIndex + batchSize, messages.length);
        const batch = messages.slice(startIndex, endIndex);
        
        // Use document fragment for better performance
        const fragment = document.createDocumentFragment();
        
        batch.forEach(msg => {
            // Create message element without appending to DOM yet
            const messageElement = this.createMessageElement(msg.content, msg.className);
            fragment.appendChild(messageElement);
        });
        
        this.chatArea.appendChild(fragment);
        
        // Load next batch with delay to prevent blocking
        if (endIndex < messages.length) {
            requestAnimationFrame(() => {
                this.loadMessageBatch(messages, endIndex, batchSize);
            });
        } else {
            // Scroll to bottom after all messages are loaded
            this.chatArea.scrollTop = this.chatArea.scrollHeight;
        }
    }
    
    createMessageElement(content, className) {
        const wrapper = document.createElement('div');
        wrapper.className = 'flex items-start gap-4 mb-4';
        
        // Implementation similar to appendMessage but returns element instead of appending
        // ... (implementation details)
        
        return wrapper;
    }
    
    clearChat() {
        localStorage.removeItem('chatHistory');
        if (this.chatArea) {
            this.chatArea.innerHTML = '';
        }
        this.updateStatusBar('Historial de chat borrado');
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatManager = new ChatManager();
});

// Global functions for backward compatibility
function sendMessage(message) {
    if (window.chatManager) {
        window.chatManager.sendMessage(message);
    }
}

function updateStatusBar(msg) {
    if (window.chatManager) {
        window.chatManager.updateStatusBar(msg);
    }
}
