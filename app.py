# requirements.txt
torch
transformers
flask
flask-cors
nltk
textblob
datasets
accelerate
safetensors
streamlit

# ==================================================
# COMPLETE MENTAL HEALTH CHATBOT IMPLEMENTATION
# Save this as: mental_health_chatbot.py
# ==================================================

import os
import re
import json
import uuid
import sqlite3
import datetime
from pathlib import Path
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    pipeline
)
from datasets import Dataset

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    print("NLTK download may have failed - continuing anyway")

class SafetyFilter:
    """Content filtering and crisis detection system"""
    
    def __init__(self):
        try:
            self.analyzer = SentimentIntensityAnalyzer()
        except:
            print("Warning: VADER sentiment analyzer not available")
            self.analyzer = None
        
        # Crisis keywords that require immediate attention
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'hurt myself',
            'self harm', 'die', 'overdose', 'jump', 'cut myself',
            'want to die', 'no point living', 'better off dead'
        ]
        
        # Inappropriate content patterns
        self.inappropriate_patterns = [
            r'\b(hate|violence|abuse)\b',
            r'\b(drug dealing|illegal)\b'
        ]
    
    def detect_crisis(self, text):
        """Detect if user input indicates crisis situation"""
        text_lower = text.lower()
        for keyword in self.crisis_keywords:
            if keyword in text_lower:
                return True
        return False
    
    def check_inappropriate_content(self, text):
        """Check for inappropriate content"""
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, text.lower()):
                return True
        return False
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of user input"""
        if not self.analyzer:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        scores = self.analyzer.polarity_scores(text)
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }
    
    def filter_input(self, user_input):
        """Main filtering function"""
        if self.detect_crisis(user_input):
            return {
                'safe': False,
                'type': 'crisis',
                'message': "I'm concerned about what you've shared. Please reach out to a crisis helpline: 988 (Suicide & Crisis Lifeline) or emergency services immediately. You matter, and help is available."
            }
        
        if self.check_inappropriate_content(user_input):
            return {
                'safe': False,
                'type': 'inappropriate',
                'message': "I'm here to provide supportive conversation. Let's focus on your wellbeing and positive topics."
            }
        
        return {'safe': True}

class MentalHealthChatbot:
    """Main chatbot class for training and inference"""
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.model_path = "./fine_tuned_model"
        self.trained = False
        
        # Try to load existing model, fallback to base model
        try:
            if os.path.exists(self.model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
                self.trained = True
                print("Loaded fine-tuned model")
            else:
                raise FileNotFoundError("No fine-tuned model found")
        except:
            print("Loading base DialoGPT model")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
    
    def create_training_data(self):
        """Create sample training data for mental health conversations"""
        training_data = [
            {
                "input": "I'm feeling really anxious about work",
                "response": "I understand that work anxiety can feel overwhelming. It's completely normal to feel this way. Would you like to talk about what specifically is causing you stress?"
            },
            {
                "input": "I can't sleep and feel worried",
                "response": "Sleep troubles often go hand in hand with worry. You're not alone in this. Have you tried any relaxation techniques before bedtime?"
            },
            {
                "input": "I feel so lonely lately",
                "response": "Loneliness can be really difficult to handle. Thank you for sharing that with me. Sometimes talking about it can help. What's been contributing to these feelings?"
            },
            {
                "input": "I'm stressed about my relationship",
                "response": "Relationship stress can weigh heavily on your mind. It takes courage to acknowledge these feelings. What aspects of your relationship are causing you the most concern?"
            },
            {
                "input": "I feel overwhelmed with everything",
                "response": "Feeling overwhelmed is a sign that you're dealing with a lot right now. It's okay to feel this way. Sometimes breaking things down into smaller steps can help. What's feeling most overwhelming today?"
            },
            {
                "input": "I'm having trouble concentrating",
                "response": "Difficulty concentrating can be frustrating and is often a sign of stress or anxiety. You're not alone in experiencing this. Have you noticed any patterns in when it's harder to focus?"
            },
            {
                "input": "I feel sad all the time",
                "response": "Persistent sadness can be really draining. I'm glad you're reaching out about this. These feelings are valid and it's important to talk about them. How long have you been feeling this way?"
            },
            {
                "input": "I don't know what to do anymore",
                "response": "Feeling lost and uncertain can be incredibly difficult. Thank you for trusting me with these feelings. Sometimes when we feel stuck, talking through our thoughts can help provide clarity. What's been weighing on your mind the most?"
            }
        ]
        
        # Save training data
        os.makedirs("data", exist_ok=True)
        with open("data/training_data.json", "w") as f:
            json.dump(training_data, f, indent=2)
        
        return training_data
    
    def prepare_dataset(self, data):
        """Prepare training dataset from conversation data"""
        conversations = []
        for item in data:
            # Format: <input> <eos> <response> <eos>
            conversation = f"{item['input']}{self.tokenizer.eos_token}{item['response']}{self.tokenizer.eos_token}"
            conversations.append(conversation)
        
        # Tokenize conversations
        tokenized = self.tokenizer(
            conversations,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        })
    
    def train_model(self, training_data, output_dir="./fine_tuned_model"):
        """Fine-tune the model with mental health conversation data"""
        print("Preparing dataset for training...")
        dataset = self.prepare_dataset(training_data)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Reduced for memory
            save_steps=100,
            save_total_limit=2,
            logging_steps=50,
            learning_rate=5e-5,
            warmup_steps=50,
            no_cuda=torch.cuda.is_available() == False,  # Use CPU if no GPU
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        print("Starting model training...")
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
        
        # Update model path and set trained flag
        self.model_path = output_dir
        self.trained = True
    
    def generate_response(self, user_input, max_length=150, temperature=0.7):
        """Generate empathetic response to user input"""
        # Encode input
        input_text = f"{user_input}{self.tokenizer.eos_token}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate response with the model
        with torch.no_grad():
            try:
                output = self.model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + 100,  # Relative to input length
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
                
                # Decode and clean response
                full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                response = full_response.replace(user_input, "").strip()
                
                # Fallback responses if generation fails
                if not response or len(response) < 10:
                    fallback_responses = [
                        "I'm here to listen. Can you tell me more about how you're feeling?",
                        "Thank you for sharing that with me. How has this been affecting you?",
                        "I understand this is difficult. Would you like to talk more about what's been on your mind?",
                        "Your feelings are valid. What would be most helpful for you right now?",
                        "I appreciate you opening up to me. How long have you been experiencing this?"
                    ]
                    import random
                    response = random.choice(fallback_responses)
                
                return response
                
            except Exception as e:
                print(f"Generation error: {e}")
                return "I'm here to support you. Can you tell me more about what's on your mind?"

class DatabaseManager:
    """Handle all database operations for session logging"""
    
    def __init__(self, db_path="logs/sessions.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT,
                user_input TEXT,
                bot_response TEXT,
                sentiment_score REAL,
                timestamp DATETIME,
                crisis_flag BOOLEAN,
                session_start DATETIME
            )
        ''')
        
        # Analytics table for aggregated data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                date TEXT PRIMARY KEY,
                total_conversations INTEGER,
                avg_sentiment REAL,
                crisis_count INTEGER,
                unique_sessions INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_session(self, session_id, user_input, bot_response, sentiment_score, crisis_flag=False):
        """Log a conversation turn to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = datetime.datetime.now()
        
        cursor.execute('''
            INSERT INTO sessions (id, user_input, bot_response, sentiment_score, timestamp, crisis_flag, session_start)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, user_input, bot_response, sentiment_score, current_time, crisis_flag, current_time))
        
        conn.commit()
        conn.close()
    
    def get_session_history(self, session_id):
        """Retrieve conversation history for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_input, bot_response, sentiment_score, timestamp, crisis_flag
            FROM sessions WHERE id = ? ORDER BY timestamp
        ''', (session_id,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'user_input': row[0],
                'bot_response': row[1],
                'sentiment_score': row[2],
                'timestamp': row[3],
                'crisis_flag': row[4]
            })
        
        conn.close()
        return history
    
    def get_analytics(self):
        """Get conversation analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_conversations,
                AVG(sentiment_score) as avg_sentiment,
                SUM(CASE WHEN crisis_flag = 1 THEN 1 ELSE 0 END) as crisis_count,
                COUNT(DISTINCT id) as unique_sessions
            FROM sessions
        ''')
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            'total_conversations': stats[0] or 0,
            'average_sentiment': round(stats[1] or 0, 3),
            'crisis_interventions': stats[2] or 0,
            'unique_sessions': stats[3] or 0
        }

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Initialize components
print("Initializing Mental Health Chatbot...")
chatbot = MentalHealthChatbot()
safety_filter = SafetyFilter()
db_manager = DatabaseManager()

# Create training data and train model if not already trained
if not chatbot.trained:
    print("Creating training data and training model...")
    training_data = chatbot.create_training_data()
    chatbot.train_model(training_data)

print("Chatbot initialized successfully!")

# Flask Routes
@app.route('/')
def home():
    """Serve the main chat interface"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mental Health Support Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }

        .chat-header h1 {
            font-size: 1.5rem;
            margin-bottom: 5px;
        }

        .chat-header p {
            font-size: 0.9rem;
            opacity: 0.9;
        }

        .disclaimer {
            background: #fff3cd;
            color: #856404;
            padding: 10px;
            text-align: center;
            font-size: 0.8rem;
            border-bottom: 1px solid #ffeaa7;
        }

        .crisis-banner {
            background: #ff6b6b;
            color: white;
            padding: 10px;
            text-align: center;
            animation: pulse 2s infinite;
            display: none;
        }

        .crisis-banner a {
            color: white;
            text-decoration: underline;
            font-weight: bold;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.8; }
            100% { opacity: 1; }
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }

        .message {
            margin-bottom: 20px;
            max-width: 80%;
            animation: fadeIn 0.3s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .user-message {
            margin-left: auto;
            text-align: right;
        }

        .user-message .message-content {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border-radius: 20px 20px 5px 20px;
        }

        .bot-message .message-content {
            background: white;
            color: #333;
            border-radius: 20px 20px 20px 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #4facfe;
        }

        .message-content {
            padding: 15px 20px;
            display: inline-block;
            word-wrap: break-word;
            line-height: 1.5;
            max-width: 100%;
        }

        .message-time {
            font-size: 0.7rem;
            color: #999;
            margin-top: 5px;
        }

        .typing-indicator {
            display: none;
            padding: 15px 20px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #4facfe;
            margin-bottom: 20px;
        }

        .typing-dots {
            display: flex;
            gap: 4px;
        }

        .typing-dots span {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4facfe;
            animation: typing 1.4s infinite ease-in-out;
        }

        .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
        .typing-dots span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes typing {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }

        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #eee;
        }

        .input-wrapper {
            display: flex;
            gap: 10px;
            align-items: flex-end;
        }

        #messageInput {
            flex: 1;
            border: 2px solid #e1e5e9;
            border-radius: 25px;
            padding: 15px 20px;
            font-size: 1rem;
            resize: none;
            outline: none;
            min-height: 50px;
            max-height: 120px;
            transition: border-color 0.3s ease;
        }

        #messageInput:focus {
            border-color: #4facfe;
        }

        #sendButton {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 15px 25px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 80px;
        }

        #sendButton:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(79, 172, 254, 0.3);
        }

        #sendButton:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .stats-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
            font-size: 1.2rem;
        }

        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .chat-container {
                width: 100%;
                height: 100vh;
                border-radius: 0;
            }
            
            .message {
                max-width: 90%;
            }
            
            .stats-button {
                bottom: 10px;
                right: 10px;
                width: 50px;
                height: 50px;
            }
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }

        .modal-content {
            background-color: white;
            margin: 5% auto;
            padding: 20px;
            border-radius: 10px;
            width: 90%;
            max-width: 500px;
        }

        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }

        .close:hover { color: black; }
    </style>
</head>
<body>
    <div class="chat-container">
        <header class="chat-header">
            <h1>💙 Mental Wellness Companion</h1>
            <p>A safe space for support and conversation</p>
        </header>
        
        <div class="disclaimer">
            <strong>Important:</strong> This is an AI assistant for emotional support. In crisis situations, please contact 988 (Suicide & Crisis Lifeline) or emergency services.
        </div>
        
        <div class="crisis-banner" id="crisis-banner">
            <strong>Crisis Resources:</strong> 
            <a href="tel:988">📞 988 Suicide & Crisis Lifeline</a> | 
            <a href="tel:911">🚨 911 Emergency</a>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot-message">
                <div class="message-content">
                    Hello! I'm here to listen and support you. This is a safe space where you can share what's on your mind. How are you feeling today?
                </div>
                <div class="message-time">Just now</div>
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
        
        <div class="chat-input-container">
            <div class="input-wrapper">
                <textarea 
                    id="messageInput" 
                    placeholder="Share what's on your mind... (Press Enter to send, Shift+Enter for new line)" 
                    rows="1"
                ></textarea>
                <button id="sendButton">Send</button>
            </div>
        </div>
    </div>
    
    <button class="stats-button" onclick="showStats()" title="View Statistics">📊</button>
    
    <!-- Statistics Modal -->
    <div id="statsModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeStats()">&times;</span>
            <h2>📊 Conversation Statistics</h2>
            <div id="statsContent">Loading...</div>
        </div>
    </div>

    <script>
        class MentalHealthChatbot {
            constructor() {
                this.sessionId = this.generateSessionId();
                this.messageInput = document.getElementById('messageInput');
                this.sendButton = document.getElementById('sendButton');
                this.chatMessages = document.getElementById('chatMessages');
                this.crisisBanner = document.getElementById('crisis-banner');
                this.typingIndicator = document.getElementById('typingIndicator');
                
                this.initializeEventListeners();
                this.autoResizeTextarea();
            }
            
            generateSessionId() {
                return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
            }
            
            initializeEventListeners() {
                this.sendButton.addEventListener('click', () => this.sendMessage());
                this.messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.sendMessage();
                    }
                });
                
                // Auto-focus on input
                this.messageInput.focus();
            }
            
            autoResizeTextarea() {
                this.messageInput.addEventListener('input', function() {
                    this.style.height = 'auto';
                    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
                });
            }
            
            async sendMessage() {
                const message = this.messageInput.value.trim();
                if (!message) return;
                
                // Disable input while processing
                this.setInputState(false);
                
                // Add user message to chat
                this.addMessage(message, 'user');
                this.messageInput.value = '';
                this.messageInput.style.height = 'auto';
                
                // Show typing indicator
                this.showTypingIndicator();
                
                try {
                    // Send to backend
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            session_id: this.sessionId
                        })
                    });
                    
                    const data = await response.json();
                    
                    // Hide typing indicator
                    this.hideTypingIndicator();
                    
                    if (data.error) {
                        this.addMessage('I apologize, but I encountered an error. Please try again. If this continues, please reach out to a human counselor.', 'bot');
                    } else {
                        // Show crisis banner if needed
                        if (data.crisis_detected) {
                            this.showCrisisBanner();
                        }
                        
                        // Add bot response
                        this.addMessage(data.response, 'bot');
                        
                        // Log sentiment for debugging (optional)
                        if (data.sentiment && data.sentiment.compound < -0.5) {
                            console.log('Detected negative sentiment:', data.sentiment.compound);
                        }
                    }
                    
                } catch (error) {
                    console.error('Error:', error);
                    this.hideTypingIndicator();
                    this.addMessage('I seem to be having connection issues. Please try again in a moment. If you need immediate help, please contact 988 or emergency services.', 'bot');
                } finally {
                    this.setInputState(true);
                }
            }
            
            addMessage(content, type) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}-message`;
                
                const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                
                messageDiv.innerHTML = `
                    <div class="message-content">${this.formatMessage(content)}</div>
                    <div class="message-time">${time}</div>
                `;
                
                this.chatMessages.appendChild(messageDiv);
                this.scrollToBottom();
            }
            
            formatMessage(content) {
                // Simple formatting for better readability
                return content
                    .replace(/\n/g, '<br>')
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>');
            }
            
            showTypingIndicator() {
                this.typingIndicator.style.display = 'block';
                this.scrollToBottom();
            }
            
            hideTypingIndicator() {
                this.typingIndicator.style.display = 'none';
            }
            
            setInputState(enabled) {
                this.messageInput.disabled = !enabled;
                this.sendButton.disabled = !enabled;
                if (enabled) {
                    this.messageInput.focus();
                }
            }
            
            showCrisisBanner() {
                this.crisisBanner.style.display = 'block';
                setTimeout(() => {
                    this.crisisBanner.style.display = 'none';
                }, 15000); // Hide after 15 seconds
            }
            
            scrollToBottom() {
                setTimeout(() => {
                    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
                }, 100);
            }
        }

        // Statistics functions
        async function showStats() {
            const modal = document.getElementById('statsModal');
            const content = document.getElementById('statsContent');
            
            modal.style.display = 'block';
            content.innerHTML = 'Loading statistics...';
            
            try {
                const response = await fetch('/analytics');
                const data = await response.json();
                
                content.innerHTML = `
                    <div style="text-align: center; margin: 20px 0;">
                        <h3>📈 Platform Statistics</h3>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
                            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
                                <strong style="color: #4facfe;">Total Conversations</strong><br>
                                <span style="font-size: 1.5em;">${data.total_conversations}</span>
                            </div>
                            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
                                <strong style="color: #28a745;">Unique Sessions</strong><br>
                                <span style="font-size: 1.5em;">${data.unique_sessions}</span>
                            </div>
                            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
                                <strong style="color: #ffc107;">Avg Sentiment</strong><br>
                                <span style="font-size: 1.5em;">${data.average_sentiment}</span>
                            </div>
                            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
                                <strong style="color: #dc3545;">Crisis Interventions</strong><br>
                                <span style="font-size: 1.5em;">${data.crisis_interventions}</span>
                            </div>
                        </div>
                        <p style="margin-top: 20px; color: #666; font-size: 0.9em;">
                            This chatbot has helped provide support across ${data.unique_sessions} unique sessions.
                        </p>
                    </div>
                `;
            } catch (error) {
                content.innerHTML = '<p>Error loading statistics.</p>';
            }
        }

        function closeStats() {
            document.getElementById('statsModal').style.display = 'none';
        }

        // Initialize the chatbot when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            new MentalHealthChatbot();
        });

        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('statsModal');
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
    </script>
</body>
</html>
    '''

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and return AI responses"""
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not user_input:
            return jsonify({'error': 'Empty message'}), 400
        
        # Safety filtering
        safety_check = safety_filter.filter_input(user_input)
        if not safety_check['safe']:
            db_manager.log_session(session_id, user_input, safety_check['message'], 
                                 -1.0, crisis_flag=True)
            return jsonify({
                'response': safety_check['message'],
                'session_id': session_id,
                'crisis_detected': True
            })
        
        # Generate AI response
        bot_response = chatbot.generate_response(user_input)
        
        # Analyze sentiment
        sentiment = safety_filter.analyze_sentiment(user_input)
        
        # Log session
        db_manager.log_session(session_id, user_input, bot_response, 
                             sentiment['compound'], crisis_flag=False)
        
        return jsonify({
            'response': bot_response,
            'session_id': session_id,
            'sentiment': sentiment,
            'crisis_detected': False
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error', 'message': 'I apologize, but I encountered an error. Please try again.'}), 500

@app.route('/analytics', methods=['GET'])
def analytics():
    """Get conversation analytics and statistics"""
    try:
        stats = db_manager.get_analytics()
        return jsonify(stats)
    except Exception as e:
        print(f"Error in analytics endpoint: {e}")
        return jsonify({'error': 'Could not retrieve analytics'}), 500

@app.route('/sessions/<session_id>', methods=['GET'])
def get_session_history(session_id):
    """Retrieve conversation history for a specific session"""
    try:
        history = db_manager.get_session_history(session_id)
        return jsonify({'history': history})
    except Exception as e:
        print(f"Error retrieving session history: {e}")
        return jsonify({'error': 'Could not retrieve session history'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': chatbot.trained,
        'timestamp': datetime.datetime.now().isoformat()
    })
