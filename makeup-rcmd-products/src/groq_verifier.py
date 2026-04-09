import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqVerifier:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            self.client = None
        else:
            self.client = Groq(api_key=api_key)

    def verify_product(self, user_profile: dict, product_name: str, ingredients: str) -> str:
        if not self.client:
            return "⚠️ Groq API is not configured. Please add your GROQ_API_KEY to the .env file to use this feature."

        profile_str = ", ".join(f"{k}: {v}" for k, v in user_profile.items())
        
        ingredients_text = f" which contains the following ingredients: {ingredients}" if ingredients else ""
        prompt = f"""You are an expert dermatologist assisting a user with skincare recommendations.
The user has the following skin profile: {profile_str}.

They want to use a product named '{product_name}'{ingredients_text}.

Is this product suitable for their specific skin profile?
CRITICAL INSTRUCTION: State if it's suitable or dangerous, and give the reasoning based on the ingredients it typically contains.
YOUR ENTIRE RESPONSE MUST NOT EXCEED 2 LINES TOTAL. Be extremely brief (e.g. Yes/No. Reason).
"""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a professional dermatologist providing targeted skincare advice."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error contacting Groq API: {str(e)}"
