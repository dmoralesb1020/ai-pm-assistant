"""
LLM Client - OpenAI API wrapper with error handling and token tracking
"""
import time
import logging
from typing import List, Dict, Optional, Generator
from openai import OpenAI, OpenAIError
from src.utils import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """
    Wrapper for OpenAI API with error handling, retry logic, and token tracking
    """
    
    def __init__(self):
        """Initialize OpenAI client with configuration"""
        self.config = get_config()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
        # Pricing per 1M tokens
        self.pricing = {
            "gpt-4o-mini": {
                "input": 0.150,   # $0.150 per 1M input tokens
                "output": 0.600   # $0.600 per 1M output tokens
            },
            "gpt-4o": {
                "input": 2.50,    # $2.50 per 1M input tokens
                "output": 10.00   # $10.00 per 1M output tokens
            }
        }
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Dict:
        """
        Generate completion from OpenAI API with retry logic
        
        Args:
            prompt: User prompt/question
            system_message: System instructions (optional)
            temperature: Randomness (0-2, default from config)
            max_tokens: Maximum tokens to generate (default from config)
            model: Model to use (default from config)
            max_retries: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            Dict with keys: content, tokens_used, cost, model
        """
        # Use config defaults if not specified
        temperature = temperature or self.config.TEMPERATURE
        max_tokens = max_tokens or self.config.MAX_TOKENS
        model = model or self.config.OPENAI_MODEL
        
        # Build messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Retry loop
        for attempt in range(max_retries):
            try:
                logger.info(f"Calling OpenAI API (attempt {attempt + 1}/{max_retries})")
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Extract response data
                content = response.choices[0].message.content
                
                # Token usage
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                # Calculate cost
                cost = self._calculate_cost(model, input_tokens, output_tokens)
                
                # Update tracking
                self.total_tokens_used += total_tokens
                self.total_cost += cost
                
                logger.info(
                    f"✅ Success | Tokens: {total_tokens} | Cost: ${cost:.4f} | "
                    f"Total cost: ${self.total_cost:.4f}"
                )
                
                return {
                    "content": content,
                    "tokens_used": total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost,
                    "model": model
                }
                
            except OpenAIError as e:
                logger.error(f"❌ OpenAI API error (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Max retries reached. Raising exception.")
                    raise
            
            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}")
                raise
    
    def generate_stream(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Generate streaming completion from OpenAI API
        
        Args:
            prompt: User prompt/question
            system_message: System instructions (optional)
            temperature: Randomness (0-2, default from config)
            max_tokens: Maximum tokens to generate (default from config)
            model: Model to use (default from config)
            max_retries: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            
        Yields:
            Content chunks as they arrive
        """
        temperature = temperature or self.config.TEMPERATURE
        max_tokens = max_tokens or self.config.MAX_TOKENS
        model = model or self.config.OPENAI_MODEL
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            logger.info("Starting streaming completion...")
            
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            raise
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> Dict:
        """
        Multi-turn chat completion
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [
                         {"role": "system", "content": "You are a PM assistant"},
                         {"role": "user", "content": "What is a sprint?"},
                         {"role": "assistant", "content": "A sprint is..."},
                         {"role": "user", "content": "How long should it be?"}
                     ]
            temperature: Randomness (0-2)
            max_tokens: Maximum tokens to generate
            model: Model to use
            
        Returns:
            Dict with keys: content, tokens_used, cost, model
        """
        temperature = temperature or self.config.TEMPERATURE
        max_tokens = max_tokens or self.config.MAX_TOKENS
        model = model or self.config.OPENAI_MODEL
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            cost = self._calculate_cost(model, input_tokens, output_tokens)
            
            self.total_tokens_used += total_tokens
            self.total_cost += cost
            
            return {
                "content": content,
                "tokens_used": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "model": model
            }
            
        except Exception as e:
            logger.error(f"❌ Chat error: {e}")
            raise
    
    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost based on token usage
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        if model not in self.pricing:
            logger.warning(f"Unknown model '{model}', using gpt-4o-mini pricing")
            model = "gpt-4o-mini"
        
        input_cost = (input_tokens / 1_000_000) * self.pricing[model]["input"]
        output_cost = (output_tokens / 1_000_000) * self.pricing[model]["output"]
        
        return input_cost + output_cost
    
    def get_usage_stats(self) -> Dict:
        """
        Get cumulative usage statistics
        
        Returns:
            Dict with total_tokens and total_cost
        """
        return {
            "total_tokens": self.total_tokens_used,
            "total_cost": self.total_cost
        }
    
    def reset_usage_stats(self):
        """Reset usage tracking counters"""
        self.total_tokens_used = 0
        self.total_cost = 0.0
        logger.info("Usage stats reset")