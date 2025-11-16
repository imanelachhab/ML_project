import streamlit as st
import joblib
import random

# ==================== CONFIGURATION ====================
MODEL_PATH = "mon_modele_xgboost_smote.pkl"
# ======================================================

class MLChatbot:
    def __init__(self):
        try:
            self.model_data = joblib.load(MODEL_PATH)
            self.ml_loaded = True
            st.success("ML model loaded successfully")
        except Exception as e:
            st.error(f"Error loading ML model: {e}")
            self.ml_loaded = False
    
    def detect_emotion_ml(self, text):
        if not self.ml_loaded:
            return "joy"
        
        try:
            xgb_model = self.model_data['model']
            vectorizer = self.model_data['vectorizer']
            label_encoder = self.model_data['label_encoder']
            
            text_vec = vectorizer.transform([text])
            emotion_code = xgb_model.predict(text_vec)[0]
            emotion = label_encoder.inverse_transform([emotion_code])[0]
            
            return emotion
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return 'neutral'
    
    def generate_response(self, emotion, user_text):
        """Generate empathetic responses based on detected emotion"""
        responses = {
            'joy': [
                "That's wonderful! I'm glad you're feeling happy. What made your day so great?",
                "It's great to see you in such a positive mood! Keep that energy going!",
                "Your joy is contagious! Tell me more about what's bringing you happiness.",
                "I love your positive energy! What's making you so happy today?"
            ],
            'sadness': [
                "I'm sorry you're feeling down. Would you like to talk about what's bothering you?",
                "It's okay to feel sad sometimes. I'm here to listen if you need someone to talk to.",
                "I understand this is a difficult moment. How can I support you?",
                "Your feelings are valid. Take your time, I'm here for you."
            ],
            'anger': [
                "I can sense your frustration. Take a deep breath. What happened?",
                "It sounds like something really upset you. Would you like to share what's on your mind?",
                "Your feelings are valid. Let's talk about what's making you angry.",
                "I understand you're upset. Sometimes it helps to talk about it."
            ],
            'fear': [
                "I understand you're feeling anxious. You're not alone. What's worrying you?",
                "It's natural to feel afraid sometimes. Let's work through this together.",
                "I'm here for you. Tell me what's making you feel this way.",
                "Take a moment to breathe. I'm here to help you through this."
            ],
            'love': [
                "How beautiful! Love is such a powerful emotion. Tell me more!",
                "That's heartwarming! I'd love to hear more about what you're experiencing.",
                "Love makes the world brighter. Share more about this feeling!",
                "What a wonderful feeling! I'm happy for you."
            ],
            'surprise': [
                "Oh wow! That must have been unexpected. What happened?",
                "Surprises can be exciting! Tell me all about it.",
                "I'm curious to know more about what surprised you!",
                "That sounds interesting! What was the surprise?"
            ],
            'neutral': [
                "I'm listening. Tell me more about what's on your mind.",
                "How are you feeling right now? I'm here to chat.",
                "I understand. Would you like to share more?",
                "I'm here for you. What would you like to talk about?"
            ]
        }
        
        emotion_lower = emotion.lower()
        if emotion_lower in responses:
            return random.choice(responses[emotion_lower])
        else:
            return f"I understand you're feeling {emotion}. Tell me more about what's going on."

def main():
    st.set_page_config(
        page_title="Emotion Chatbot",
        layout="centered"
    )
    
    st.title("Emotion Detection Chatbot")
    st.markdown("Powered by XGBoost ML Model")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How are you feeling today?"}
        ]
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = MLChatbot()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if user_input := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing emotion..."):
                emotion = st.session_state.chatbot.detect_emotion_ml(user_input)
                response = st.session_state.chatbot.generate_response(emotion, user_input)
            
            st.info(f"Detected Emotion: {emotion.upper()}")
            st.write(response)
            
            full_response = f"Detected Emotion: {emotion.upper()}\n\n{response}"
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response
            })

if __name__ == "__main__":
    main()