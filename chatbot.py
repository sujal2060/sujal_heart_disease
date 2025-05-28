import requests
import json

class HeartDiseaseChatbot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.together.xyz/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def get_response(self, user_message):
        try:
            payload = {
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a helpful medical assistant specializing in heart disease. 
                        Provide accurate, helpful, and empathetic responses about heart health, 
                        symptoms, and general medical advice. Always remind users to consult with 
                        healthcare professionals for specific medical concerns."""
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            return response.json()['choices'][0]['message']['content']
            
        except Exception as e:
            return f"I apologize, but I'm having trouble connecting right now. Please try again later. Error: {str(e)}" 