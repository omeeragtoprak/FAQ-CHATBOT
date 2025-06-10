document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const navButtons = document.querySelectorAll('.nav-button');
    const sections = document.querySelectorAll('main section');
    const faqList = document.getElementById('faq-list');

    // Navigation
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            navButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            sections.forEach(section => section.classList.remove('active'));
            document.getElementById(button.dataset.target).classList.add('active');
        });
    });

    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        if (isUser) messageDiv.classList.add('user');

        const avatar = document.createElement('div');
        avatar.classList.add('avatar');
        avatar.innerHTML = isUser ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

        const content = document.createElement('div');
        content.classList.add('content');
        content.textContent = message;

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Animate message appearance
        anime({
            targets: messageDiv,
            translateY: [20, 0],
            opacity: [0, 1],
            duration: 500,
            easing: 'easeOutCubic'
        });
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (message) {
            addMessage(message, true);
            userInput.value = '';
            adjustTextareaHeight();

            chatMessages.appendChild(createTypingIndicator());
            
            setTimeout(() => {
                fetch('/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: message }),
                })
                .then(response => response.json())
                .then(data => {
                    removeTypingIndicator();
                    addMessage(data.answer);
                })
                .catch(error => {
                    console.error('Error:', error);
                    removeTypingIndicator();
                    addMessage('Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.');
                });
            }, 1000 + Math.random() * 1000);
        }
    }

    function createTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.classList.add('message', 'bot', 'typing-indicator');
        indicator.innerHTML = '<div class="avatar"><i class="fas fa-robot"></i></div><div class="content"><span></span><span></span><span></span></div>';
        return indicator;
    }

    function removeTypingIndicator() {
        const indicator = document.querySelector('.typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    function adjustTextareaHeight() {
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    userInput.addEventListener('input', adjustTextareaHeight);

    // FAQ Section
    const faqData = [
        { question: "Kredi kartı başvurusu nasıl yapabilirim?", answer: "Kredi kartı başvurusu için web sitemizden veya mobil uygulamamızdan online başvuru yapabilir, ya da en yakın Garanti BBVA şubesini ziyaret edebilirsiniz." },
        { question: "Hesap bakiyemi nasıl öğrenebilirim?", answer: "Hesap bakiyenizi internet bankacılığı, mobil uygulama, ATM'ler veya 444 0 333 Alo Garanti BBVA'yı arayarak öğrenebilirsiniz." },
        { question: "Şubelerinizin çalışma saatleri nedir?", answer: "Garanti BBVA şubeleri genel olarak hafta içi 09:00-17:00 saatleri arasında hizmet vermektedir. Bazı şubelerimizin çalışma saatleri farklılık gösterebilir." },
        { question: "Yurt dışında kartımı nasıl kullanabilirim?", answer: "Yurt dışı kullanımı için kartınızı aktif etmeniz gerekmektedir. Bu işlemi internet bankacılığı, mobil uygulama veya 444 0 333 Alo Garanti BBVA üzerinden yapabilirsiniz." },
        { question: "Kredi başvurusu için gerekli belgeler nelerdir?", answer: "Kredi başvurusu için genellikle kimlik belgesi ve gelir belgesi gerekmektedir. Detaylı bilgi için web sitemizi ziyaret edebilir veya 444 0 333 Alo Garanti BBVA'yı arayabilirsiniz." }
    ];

    faqData.forEach((item, index) => {
        const faqItem = document.createElement('div');
        faqItem.classList.add('faq-item');
        faqItem.innerHTML = `
            <div class="faq-question" data-index="${index}">
                ${item.question}
                <i class="fas fa-chevron-down"></i>
            </div>
            <div class="faq-answer">${item.answer}</div>
        `;
        faqList.appendChild(faqItem);
    });

    faqList.addEventListener('click', (e) => {
        if (e.target.classList.contains('faq-question')) {
            const answer = e.target.nextElementSibling;
            const icon = e.target.querySelector('i');
            answer.classList.toggle('active');
            icon.classList.toggle('fa-chevron-up');
            icon.classList.toggle('fa-chevron-down');

            anime({
                targets: answer,
                height: answer.classList.contains('active') ? [0, answer.scrollHeight] : [answer.scrollHeight, 0],
                duration: 300,
                easing: 'easeInOutQuad'
            });
        }
    });
});