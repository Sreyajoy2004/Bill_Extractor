import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for all settings"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # ← ADDED
    
    # Zoho Books API
    ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
    ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
    ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")
    ZOHO_ORGANIZATION_ID = os.getenv("ZOHO_ORGANIZATION_ID")
    ZOHO_API_DOMAIN = os.getenv("ZOHO_API_DOMAIN", "https://books.zoho.com/api/v3")
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    IMAGES_DIR = DATA_DIR / "images"
    GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    
    # Ensure directories exist
    for dir_path in [DATA_DIR, IMAGES_DIR, GROUND_TRUTH_DIR, OUTPUTS_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # Model configurations with approximate costs
    MODELS: Dict[str, Dict[str, Any]] = {
        # Groq Models (Text-only)
        "groq-llama-versatile": {
            "name": "llama-3.3-70b-versatile",
            "provider": "groq",
            "cost_per_1k_tokens": 0.000,
            "description": "Groq Llama 3.3 70B versatile (text-only)"
        },
        "groq-llama-instant": {
            "name": "llama-3.1-8b-instant",
            "provider": "groq",
            "cost_per_1k_tokens": 0.000,
            "description": "Groq Llama 3.1 8B instant (text-only)"
        },
        "groq-llama-scout": {
            "name": "meta-llama/llama-4-scout-17b-16e-instruct",
            "provider": "groq",
            "cost_per_1k_tokens": 0.000,
            "description": "Groq Llama 4 Scout - Latest text model"
        },
        # OpenAI Models
        "openai-gpt-4o-mini": {
            "name": "gpt-4o-mini",
            "provider": "openai",
            "cost_per_1k_tokens": 0.0005,
            "description": "OpenAI GPT-4o-mini - Cheaper vision model"
        },
        "gpt-4o": {
            "name": "gpt-4o-2024-08-06",
            "provider": "openai",
            "cost_per_1k_tokens": 0.005,
            "description": "OpenAI GPT-4o - Best vision model"
        },
        # Anthropic Models
        "claude-haiku": {
            "name": "claude-3-5-haiku-20241022",
            "provider": "anthropic",
            "cost_per_1k_tokens": 0.00025,
            "description": "Anthropic Claude 3.5 Haiku - Fast and cheap"
        },
        "claude-sonnet": {
            "name": "claude-3-5-sonnet-20241022",
            "provider": "anthropic",
            "cost_per_1k_tokens": 0.005,
            "description": "Anthropic Claude 3.5 Sonnet - Balanced"
        },
        "claude-opus": {
            "name": "claude-3-opus-20240229",
            "provider": "anthropic",
            "cost_per_1k_tokens": 0.015,
            "description": "Anthropic Claude 3 Opus - Most capable"
        },
        # Google Models
        "gemini-flash": {
            "name": "gemini-2.0-flash",
            "provider": "google",
            "cost_per_1k_tokens": 0.000075,
            "description": "Google Gemini Flash - Fast, cheap"
        },
        "gemini-pro": {
            "name": "gemini-2.5-pro",
            "provider": "google",
            "cost_per_1k_tokens": 0.00125,
            "description": "Google Gemini Pro - More accurate"
        }
    }
    
    # Evaluation settings
    EXACT_MATCH_THRESHOLD = 1.0
    FUZZY_MATCH_THRESHOLD = 0.8
    
    @classmethod
    def get_model_config(cls, model_key: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        if model_key not in cls.MODELS:
            raise ValueError(f"Model '{model_key}' not found. Available: {list(cls.MODELS.keys())}")
        return cls.MODELS[model_key]
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required API keys are present"""
        required_keys = ["GOOGLE_API_KEY"]  # At minimum need Gemini
        missing = [key for key in required_keys if not getattr(cls, key)]
        
        if missing:
            print(f"Warning: Missing required API keys: {missing}")
            print("Please add your API keys to the .env file")
            return False
        
        print("All required API keys found!")
        return True

# Auto-validate on import
if __name__ != "__main__":
    Config.validate()