import google.generativeai as genai
import os
import logging
from typing import List, Optional, Union
from PIL import Image
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class VisionModule:
    """
    A production-ready module for multimodal analysis using Google's Gemini API.
    Handles image analysis and response synthesis with robust error handling.
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
        """
        Analyze an image in the context of a user query.
        
        Args:
            image_path: Path to the image file.
            query_text: User's question or context for the image.
            
        Returns:
            Description and analysis of the image relative to the query.
        """
        if not self.model:
            return "Vision analysis is currently unavailable (API not configured)."
        
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"

        try:
            image = Image.open(image_path)
            prompt = (
                f"You are an expert research assistant. Analyze this figure extracted from a scientific paper "
                f"to answer the following query: '{query_text}'\n\n"
                "Focus on interpreting data, trends, and significance relevant to the user's question."
            )
            
            response = self.model.generate_content([prompt, image])
            
            if response and response.text:
                return response.text
            return "No analysis could be generated for this image."
            
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return f"Error during image analysis: {str(e)}"

    def synthesize_response(self, 
                           query: str, 
                           text_contexts: List[str], 
                           vision_analyses: List[str]) -> str:
        """
        Produce a final synthesized answer combining text retrieval and visual analysis.
        
        Args:
            query: The original user question.
            text_contexts: List of relevant text chunks from the RAG pipeline.
            vision_analyses: List of descriptions for retrieved figures.
            
        Returns:
            A comprehensive, professional response.
        """
        if not self.model:
            # Fallback if model is not configured, though synthesis is core
            return "Synthesis service is unavailable. Please check your API configuration."

        combined_text_context = "\n\n---\n\n".join(text_contexts)
        combined_vision_context = "\n\n".join(vision_analyses) if vision_analyses else "No relevant visual data found."

        prompt = (
            f"User Query: {query}\n\n"
            "SYSTEM INSTRUCTIONS:\n"
            "You are an advanced Multimodal Research Intelligence Engine. Your goal is to provide a "
            "comprehensive, accurate, and professional answer based ONLY on the provided contexts.\n\n"
            "RELEVANT TEXT CHUNKS:\n"
            f"{combined_text_context}\n\n"
            "VISUAL ANALYSIS FROM DIAGRAMS/FIGURES:\n"
            f"{combined_vision_context}\n\n"
            "FINAL TASK:\n"
            "Integrate both text and visual descriptions into a cohesive answer. "
            "Cite figures when referencing visual data. If the information is missing, state so clearly."
        )

        try:
            response = self.model.generate_content(prompt)
            if response and response.text:
                return response.text
            return "I apologize, but I couldn't synthesize a complete answer based on the available data."
            
        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            return f"Error synthesizing final response: {str(e)}"

if __name__ == "__main__":
    # Quick connectivity test
    vm = VisionModule()
    if vm.model:
        print("VisionModule ready for production.")
    else:
        print("VisionModule configuration failed.")
