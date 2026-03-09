"""
Utility functions and configuration management
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env (for local development)
load_dotenv()

class Config:
    """Application configuration"""
    
    @staticmethod
    def _get_secret(key: str, default=None):
        """
        Get secret from Streamlit secrets (cloud) or environment variable (local)
        
        Args:
            key: Secret key name
            default: Default value if not found
            
        Returns:
            Secret value
        """
        # Try Streamlit secrets first (for cloud deployment)
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except (ImportError, FileNotFoundError, KeyError):
            pass
        
        # Fall back to environment variables (for local development)
        return os.getenv(key, default)
    
    # OpenAI Settings
    @property
    def OPENAI_API_KEY(self):
        return self._get_secret("OPENAI_API_KEY")
    
    @property
    def OPENAI_MODEL(self):
        return self._get_secret("OPENAI_MODEL", "gpt-4o-mini")
    
    @property
    def EMBEDDING_MODEL(self):
        return self._get_secret("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Generation Parameters
    @property
    def MAX_TOKENS(self):
        return int(self._get_secret("MAX_TOKENS", "2000"))
    
    @property
    def TEMPERATURE(self):
        return float(self._get_secret("TEMPERATURE", "0.7"))
    
    # Paths
    KNOWLEDGE_BASE_PATH = "data/pm_knowledge"
    VECTOR_STORE_PATH = "data/vectorstore"
    
    def validate(self):
        """Validate that required configuration is present"""
        if not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "For local development: Set it in your .env file. "
                "For Streamlit Cloud: Add it in the Secrets section of your app settings."
            )
        return True


def get_config():
    """Get validated configuration"""
    config = Config()
    config.validate()
    return config