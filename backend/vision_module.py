import google.generativeai as genai
import os
import logging
import time
from typing import List, Optional, Union
from PIL import Image
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def retry_with_backoff(func, max_retries=5, initial_delay=2):
    """Exponential backoff decorator for Gemini API calls."""
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Catch ResourceExhausted (429) or other API errors
                if "429" in str(e) or "quota" in str(e).lower():
                    logger.warning(f"Rate limit hit. Retrying in {delay}s... (Attempt {i+1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e
        return func(*args, **kwargs)
    return wrapper

class VisionModule:
    """
    A production-ready module for multimodal analysis using Google's Gemini API.
    Handles image analysis and response synthesis with robust error handling and retry logic.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment. Multimodal features will be limited.")
            self.model = None
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"VisionModule initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {str(e)}")
            self.model = None

    def analyze_image(self, image_path: str, query_text: str) -> str:
        if not self.model:
            return "Vision analysis unavailable."
        
        @retry_with_backoff
        def _generate():
            image = Image.open(image_path)
            prompt = (
                f"Analyze this research paper figure in the context of: '{query_text}'\n"
                "Explain the data and relevance purely for scientific research."
            )
            return self.model.generate_content([prompt, image])

        try:
            response = _generate()
            return response.text if response and response.text else "No analysis generated."
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return f"Error: {str(e)}"

    def synthesize_response(self, query: str, text_contexts: List[str], vision_analyses: List[str]) -> str:
        if not self.model:
            return "Synthesis service unavailable."

        combined_text = "\n\n".join(text_contexts)[:10000] # Truncate to save tokens/rate limits
        combined_vision = "\n\n".join(vision_analyses)

        prompt = (
            f"User Query: {query}\n\n"
            f"Context:\n{combined_text}\n\n"
            f"Figures:\n{combined_vision}\n\n"
            "Synthesize a professional research answer."
        )

        @retry_with_backoff
        def _generate():
            return self.model.generate_content(prompt)

        try:
            response = _generate()
            return response.text if response and response.text else "Could not synthesize answer."
        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            return f"Error: {str(e)}"

if __name__ == "__main__":
    # Quick connectivity test
    vm = VisionModule()
    if vm.model:
        print("VisionModule ready for production.")
    else:
        print("VisionModule configuration failed.")
