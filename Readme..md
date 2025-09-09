# 💙 Mental Health Support Chatbot

> **An AI-powered empathetic companion designed to provide emotional support and mental wellness assistance**

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.3+-green.svg)
![Transformers](https://img.shields.io/badge/transformers-v4.30+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 🌟 Features

### Core Functionality
- 🤖 **Fine-tuned AI Model**: Uses DialoGPT with mental health conversation training
- 🛡️ **Advanced Safety Systems**: Crisis detection with automatic resource links
- 💬 **Empathetic Responses**: Trained to provide caring, supportive dialogue
- 📊 **Sentiment Analysis**: Real-time emotion tracking and monitoring
- 📝 **Session Logging**: Comprehensive conversation history and analytics

### User Experience
- 🎨 **Beautiful Interface**: Modern, responsive web design
- 📱 **Mobile Friendly**: Optimized for all device sizes
- ⚡ **Real-time Chat**: Instant responses with typing indicators
- 🚨 **Crisis Support**: Automatic emergency resource display
- 📈 **Analytics Dashboard**: View conversation statistics and insights

### Technical Excellence
- 🐳 **Docker Ready**: Complete containerization support
- ☁️ **Cloud Deployable**: Render, Heroku, and Replit configurations
- 🔒 **Privacy Focused**: Local data storage with optional encryption
- 🧪 **Testing Suite**: Comprehensive test coverage
- ⚙️ **Easy Setup**: One-file implementation with auto-configuration

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- 4GB+ RAM (for model loading)
- Internet connection (for initial model download)

### Installation

1. **Clone or Download**
   ```bash
   # Save the mental_health_chatbot.py file to your directory
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python mental_health_chatbot.py
   ```

4. **Access Your Chatbot**
   ```
   Open http://localhost:5000 in your web browser
   ```

**That's it!** 🎉 Your mental health support chatbot is ready to help!

---

## 📋 Usage Examples

### Basic Conversation
```
User: "I'm feeling really anxious about work"
Bot: "I understand that work anxiety can feel overwhelming. It's completely normal to feel this way. Would you like to talk about what specifically is causing you stress?"
```

### Crisis Detection
```
User: "I'm having thoughts of hurting myself"
Bot: "I'm concerned about what you've shared. Please reach out to a crisis helpline: 988 (Suicide & Crisis Lifeline) or emergency services immediately. You matter, and help is available."
[Crisis banner automatically appears with emergency contacts]
```

---

## 🛠️ Advanced Usage

### Command Line Options

```bash
# Run comprehensive tests
python mental_health_chatbot.py test

# Train model only (without starting server)
python mental_health_chatbot.py train-only

# Generate deployment configuration files
python mental_health_chatbot.py create-deployment
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main chat interface |
| `/chat` | POST | Send message to chatbot |
| `/analytics` | GET | View conversation statistics |
| `/sessions/<id>` | GET | Retrieve session history |
| `/health` | GET | System health check |
| `/system/status` | GET | Performance monitoring |

### Example API Usage

```python
import requests

# Send a message to the chatbot
response = requests.post('http://localhost:5000/chat', json={
    'message': 'I need someone to talk to',
    'session_id': 'unique_session_id'
})

data = response.json()
print(data['response'])  # Bot's empathetic response
```

---

## 🎨 Customization

### Adding Training Data

Modify the `create_training_data()` method to include more conversation examples:

```python
def create_training_data(self):
    training_data = [
        {
            "input": "Your custom input here",
            "response": "Your custom empathetic response here"
        },
        # Add more examples...
    ]
    return training_data
```

### Customizing Safety Filters

Update crisis keywords in the `SafetyFilter` class:

```python
self.crisis_keywords = [
    'your_custom_crisis_word',
    'another_keyword',
    # Add more crisis indicators...
]
```

### Interface Styling

The web interface styling can be modified in the `home()` route's HTML template. Key CSS classes:

- `.chat-container` - Main chat window
- `.message` - Individual messages
- `.crisis-banner` - Emergency alert banner
- `.chat-input-container` - Input area

---

## 🚀 Deployment

### Option 1: Render (Recommended)

1. Upload your code to GitHub
2. Connect your repository to [Render](https://render.com)
3. Use the generated `render.yaml` configuration
4. Deploy as a Web Service

### Option 2: Heroku

```bash
# Install Heroku CLI, then:
git init
git add .
git commit -m "Initial deployment"
heroku create your-chatbot-name
git push heroku main
```

### Option 3: Docker

```bash
# Build the container
docker build -t mental-health-chatbot .

# Run the container
docker run -p 5000:5000 mental-health-chatbot
```

### Option 4: Replit

1. Create a new Replit project
2. Upload `mental_health_chatbot.py` and `requirements.txt`
3. Run the project - Replit handles the rest!

---

## 📊 Analytics & Monitoring

### Built-in Analytics

Access the analytics dashboard by clicking the 📊 button in the chat interface, or visit `/analytics`:

- **Total Conversations**: Number of messages processed
- **Unique Sessions**: Individual user interactions
- **Average Sentiment**: Overall emotional tone tracking
- **Crisis Interventions**: Safety system activations

### Performance Monitoring

Enable system monitoring by installing `psutil`:

```bash
pip install psutil
```

Then access `/system/status` for real-time system metrics.

---

## 🧪 Testing

### Automated Testing

```bash
# Run all tests
python mental_health_chatbot.py test
```

### Manual Testing Scenarios

1. **Normal Conversation**: "I'm feeling stressed about school"
2. **Crisis Detection**: "I want to hurt myself"
3. **Inappropriate Content**: Test content filtering
4. **Empty Messages**: Verify error handling

---

## ⚠️ Important Disclaimers

### Medical Disclaimer
**This chatbot is for educational and emotional support purposes only and should NOT replace professional mental health services.**

### Crisis Resources
In emergency situations, users should immediately contact:
- **988** - Suicide & Crisis Lifeline (US)
- **911** - Emergency Services
- **Local mental health professionals**

### Privacy Notice
- Conversations are logged locally for improvement purposes
- No data is shared with third parties
- Consider implementing data retention policies for production use

---

## 🔧 Technical Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │ ←→ │   Flask API      │ ←→ │   AI Model      │
│   (HTML/CSS/JS) │    │   (Routes/Logic) │    │   (DialoGPT)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ↓
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Safety Filter  │ ←→ │  Session Logger  │ ←→ │   Database      │
│  (Crisis Det.)  │    │  (Analytics)     │    │   (SQLite)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

- **Frontend**: Single-page application with modern CSS and JavaScript
- **Backend**: Flask REST API with comprehensive error handling
- **AI Engine**: Fine-tuned DialoGPT model for empathetic responses
- **Safety Layer**: Multi-stage content filtering and crisis detection
- **Data Layer**: SQLite database for session logging and analytics

---

## 🤝 Contributing

### Adding Features

1. **New Safety Filters**: Extend the `SafetyFilter` class
2. **Additional Training Data**: Expand the conversation dataset
3. **UI Improvements**: Modify the HTML/CSS in the `home()` route
4. **API Endpoints**: Add new routes to the Flask application

### Code Style

- Follow PEP 8 Python style guidelines
- Add docstrings for all functions and classes
- Include error handling for all external operations
- Write tests for new functionality

---

## 📈 Performance Optimization

### For Large Scale Deployment

1. **Model Optimization**:
   ```python
   # Use model quantization for faster inference
   model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```

2. **Caching Responses**:
   - Implement Redis for frequent response caching
   - Add database query optimization

3. **Load Balancing**:
   - Use multiple worker processes with Gunicorn
   - Implement horizontal scaling with container orchestration

---

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors**:
```bash
# Clear model cache and retry
rm -rf ~/.cache/huggingface/
python mental_health_chatbot.py train-only
```

**Memory Issues**:
- Reduce batch size in training configuration
- Use CPU-only mode: `export CUDA_VISIBLE_DEVICES=""`

**Port Already in Use**:
```bash
# Kill existing process
lsof -ti:5000 | xargs kill -9
```

**NLTK Data Missing**:
```python
import nltk
nltk.download('vader_lexicon')
```

---

## 📚 Educational Resources

### Mental Health Support
- [National Suicide Prevention Lifeline](https://suicidepreventionlifeline.org/)
- [Crisis Text Line](https://www.crisistextline.org/)
- [Mental Health America](https://www.mhanational.org/)

### Technical Learning
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Flask Tutorial](https://flask.palletsprojects.com/tutorial/)
- [Mental Health AI Ethics](https://www.who.int/publications/i/item/9789240029200)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Hugging Face** - For the transformers library and pre-trained models
- **Microsoft** - For the DialoGPT model architecture
- **Mental Health Community** - For guidance on ethical AI development
- **Open Source Contributors** - For the foundational libraries used

---

## 📞 Support

For technical support or questions:

1. **Check the troubleshooting section above**
2. **Review the test results**: `python mental_health_chatbot.py test`
3. **Examine the logs** in the `logs/` directory
4. **Verify your requirements** match `requirements.txt`

---

## 🌟 Star This Project

If this Mental Health Support Chatbot has been helpful for your learning or project, please consider giving it a star! ⭐

Your support helps others discover this resource and encourages continued development of ethical AI tools for mental wellness.

---

*Built with ❤️ for mental health awareness and AI education*

**Remember: Technology should augment human compassion, never replace it. This tool is designed to provide immediate support while encouraging users to seek professional help when needed.**