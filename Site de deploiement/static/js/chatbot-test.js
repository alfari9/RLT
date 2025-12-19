// Test simple du chatbot
console.log('===== CHATBOT TEST START =====');

// Créer immédiatement le bouton sans attendre DOMContentLoaded
(function() {
    console.log('1. Script chargé');
    
    function createChatbotButton() {
        console.log('2. createChatbotButton appelée');
        
        // Supprimer l'ancien bouton s'il existe
        const oldButton = document.getElementById('chatbot-toggle');
        if (oldButton) {
            console.log('3. Suppression de l\'ancien bouton');
            oldButton.remove();
        }
        
        // Créer le bouton
        const button = document.createElement('button');
        button.id = 'chatbot-toggle';
        button.className = 'chatbot-button';
        button.innerHTML = '💬';
        button.style.cssText = `
            position: fixed !important;
            bottom: 30px !important;
            right: 30px !important;
            width: 60px !important;
            height: 60px !important;
            border-radius: 50% !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            cursor: pointer !important;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 28px !important;
            z-index: 10000 !important;
        `;
        
        button.onclick = function() {
            alert('Chatbot cliqué ! Le chatbot fonctionne. 🎉');
        };
        
        document.body.appendChild(button);
        console.log('4. Bouton ajouté au DOM');
        console.log('5. Position du bouton:', window.getComputedStyle(button).position);
        console.log('6. Z-index du bouton:', window.getComputedStyle(button).zIndex);
        console.log('7. Display du bouton:', window.getComputedStyle(button).display);
    }
    
    // Essayer immédiatement si le DOM est déjà prêt
    if (document.body) {
        console.log('8. DOM body existe déjà, création immédiate');
        createChatbotButton();
    } else {
        console.log('9. Attente du DOM');
        document.addEventListener('DOMContentLoaded', createChatbotButton);
    }
    
    // Fallback après 100ms
    setTimeout(function() {
        const button = document.getElementById('chatbot-toggle');
        if (!button) {
            console.log('10. Fallback: création du bouton après 100ms');
            createChatbotButton();
        } else {
            console.log('11. Bouton déjà présent');
        }
    }, 100);
    
    console.log('===== CHATBOT TEST END =====');
})();
