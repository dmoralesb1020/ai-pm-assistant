"""
Utility functions and configuration management
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Generation Parameters
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Paths
    KNOWLEDGE_BASE_PATH = "data/pm_knowledge"
    VECTOR_STORE_PATH = "data/vectorstore"
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        return True


def get_config():
    """Get validated configuration"""
    Config.validate()
    return Config