/**
 * CHATBOT RLT - Interface JavaScript
 * Gère l'interaction utilisateur avec le chatbot
 */

class ChatbotRLT {
    constructor() {
        this.isOpen = false;
        this.messages = [];
        this.isTyping = false;
        
        this.init();
    }
    
    init() {
        // Créer les éléments du chatbot
        this.createChatbotElements();
        
        // Attacher les événements
        this.attachEvents();
        
        // Message de bienvenue
        this.addWelcomeMessage();
    }
    
    createChatbotElements() {
        // Bouton flottant
        const button = document.createElement('button');
        button.className = 'chatbot-button';
        button.innerHTML = '💬';
        button.id = 'chatbot-toggle';
        document.body.appendChild(button);
        
        // Conteneur du chatbot
        const container = document.createElement('div');
        container.className = 'chatbot-container';
        container.id = 'chatbot-container';
        container.innerHTML = `
            <div class="chatbot-header">
                <div>
                    <h3>🤖 Assistant RLT</h3>
                    <div class="status">En ligne</div>
                </div>
                <button class="chatbot-close" id="chatbot-close">×</button>
            </div>
            <div class="chatbot-messages" id="chatbot-messages">
                <!-- Messages générés dynamiquement -->
            </div>
            <div class="chatbot-input">
                <input 
                    type="text" 
                    id="chatbot-input" 
                    placeholder="Posez votre question..."
                    autocomplete="off"
                />
                <button id="chatbot-send">➤</button>
            </div>
        `;
        document.body.appendChild(container);
        
        this.button = button;
        this.container = container;
        this.messagesContainer = document.getElementById('chatbot-messages');
        this.input = document.getElementById('chatbot-input');
        this.sendButton = document.getElementById('chatbot-send');
    }
    
    attachEvents() {
        // Toggle chatbot
        this.button.addEventListener('click', () => this.toggle());
        document.getElementById('chatbot-close').addEventListener('click', () => this.close());
        
        // Envoyer message
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
    }
    
    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }
    
    open() {
        this.isOpen = true;
        this.container.classList.add('active');
        this.button.classList.add('active');
        this.input.focus();
        
        // Scroll vers le bas
        setTimeout(() => this.scrollToBottom(), 100);
    }
    
    close() {
        this.isOpen = false;
        this.container.classList.remove('active');
        this.button.classList.remove('active');
    }
    
    addWelcomeMessage() {
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'welcome-message';
        welcomeDiv.innerHTML = `
            <h4>👋 Bienvenue !</h4>
            <p>Je suis votre assistant pour la plateforme RLT. Je peux vous aider à comprendre les modèles, utiliser le site et interpréter les résultats.</p>
        `;
        this.messagesContainer.appendChild(welcomeDiv);
        
        // Ajouter des suggestions initiales
        this.addSuggestions([
            "Qu'est-ce que RLT ?",
            "Comment utiliser le site ?",
            "Quels datasets disponibles ?",
            "Aide"
        ]);
    }
    
    async sendMessage() {
        const message = this.input.value.trim();
        
        if (!message) return;
        
        // Désactiver l'input pendant l'envoi
        this.input.disabled = true;
        this.sendButton.disabled = true;
        
        // Ajouter le message de l'utilisateur
        this.addMessage(message, 'user');
        
        // Vider l'input
        this.input.value = '';
        
        // Afficher l'indicateur de frappe
        this.showTypingIndicator();
        
        try {
            // Envoyer la requête au backend
            const response = await fetch('/api/chatbot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            
            // Simuler un petit délai pour plus de réalisme
            await this.delay(500);
            
            // Retirer l'indicateur de frappe
            this.hideTypingIndicator();
            
            // Ajouter la réponse du bot
            this.addMessage(data.response, 'bot');
            
            // Ajouter les suggestions si disponibles
            if (data.suggestions && data.suggestions.length > 0) {
                this.addSuggestions(data.suggestions);
            }
            
        } catch (error) {
            console.error('Erreur chatbot:', error);
            this.hideTypingIndicator();
            this.addMessage("😔 Désolé, une erreur s'est produite. Veuillez réessayer.", 'bot');
        } finally {
            // Réactiver l'input
            this.input.disabled = false;
            this.sendButton.disabled = false;
            this.input.focus();
        }
    }
    
    addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = sender === 'bot' ? '🤖' : '👤';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        // Convertir les retours à la ligne et le markdown simple
        const formattedText = this.formatMessage(text);
        content.innerHTML = formattedText;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        this.messages.push({ text, sender, timestamp: new Date() });
    }
    
    formatMessage(text) {
        // Remplacer les ** par <strong>
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Remplacer les * par des bullet points
        text = text.replace(/^• /gm, '→ ');
        
        // Convertir les retours à la ligne
        text = text.replace(/\n/g, '<br>');
        
        return text;
    }
    
    addSuggestions(suggestions) {
        const suggestionsDiv = document.createElement('div');
        suggestionsDiv.className = 'suggestions';
        
        suggestions.forEach(suggestion => {
            const btn = document.createElement('button');
            btn.className = 'suggestion-btn';
            btn.textContent = suggestion;
            btn.addEventListener('click', () => {
                this.input.value = suggestion;
                this.sendMessage();
            });
            suggestionsDiv.appendChild(btn);
        });
        
        this.messagesContainer.appendChild(suggestionsDiv);
        this.scrollToBottom();
    }
    
    showTypingIndicator() {
        this.isTyping = true;
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        this.messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.isTyping = false;
        const typingDiv = document.getElementById('typing-indicator');
        if (typingDiv) {
            typingDiv.remove();
        }
    }
    
    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialiser le chatbot quand la page est chargée
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.chatbot = new ChatbotRLT();
        console.log('🤖 Chatbot RLT initialisé avec succès');
    } catch (error) {
        console.error('❌ Erreur lors de l\'initialisation du chatbot:', error);
    }
});

// Fallback: initialiser après un délai si DOMContentLoaded a déjà été déclenché
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(() => {
        if (!window.chatbot) {
            try {
                window.chatbot = new ChatbotRLT();
                console.log('🤖 Chatbot RLT initialisé (fallback)');
            } catch (error) {
                console.error('❌ Erreur lors de l\'initialisation du chatbot (fallback):', error);
            }
        }
    }, 100);
}
