
import hashlib
from typing import Optional, Dict, Any
from enum import Enum
import time

from agno.agent import Agent
from agno.models.deepseek import DeepSeek

from config.settings import settings
from common.logger import logger


class UserIntent(str, Enum):
    """User intent types for chat interactions."""
    NORMAL_CHAT = "normal_chat"
    IMAGE_GENERATION = "image_generation"


class IntentAnalysisService:
    """Service for analyzing user intent using Deepseek."""
    
    def __init__(self):
        self.logger = logger
        # Simple in-memory cache for intent analysis results
        self._cache: Dict[str, tuple[UserIntent, float, float]] = {}  # hash -> (intent, confidence, timestamp)
        self.cache_ttl = 300  # 5 minutes cache TTL
        
        # System prompt for intent detection
        self.system_prompt = """You are an expert intent analysis AI. Your job is to analyze user messages and determine if they want to generate an image or have a normal conversation.

CRITICAL INSTRUCTIONS:
- You must respond with ONLY ONE WORD: either "image_generation" or "normal_chat"
- Do not explain your reasoning, just return the classification
- Do not add any punctuation, spaces, or extra characters

CLASSIFICATION RULES:

Return "image_generation" if the user:
- Explicitly asks to generate, create, make, or produce an image/picture/photo/drawing/artwork/visual
- Uses phrases like "draw me", "create an image of", "generate a picture", "make a visual", "show me a picture"
- Asks for illustrations, diagrams, charts, infographics, or visual representations
- Wants to see visual concepts, designs, or artistic creations
- Requests modifications to images or visual content
- Asks for visual comparisons or before/after images

Return "normal_chat" for:
- General questions and conversations
- Text-based requests for information, explanations, or advice
- Code-related questions or programming help
- Mathematical calculations or analysis
- Writing assistance (essays, emails, stories) unless specifically asking for visual storyboards
- File analysis or data processing (unless explicitly asking for visual charts/graphs)
- General problem-solving discussions
- When unclear or ambiguous - default to normal_chat

EXAMPLES:

"Can you draw me a sunset over mountains?" → image_generation
"Create an image of a futuristic city" → image_generation
"Generate a logo for my company" → image_generation
"Show me what a quantum computer looks like" → image_generation
"Make a diagram explaining photosynthesis" → image_generation

"What is photosynthesis?" → normal_chat
"Help me write a resume" → normal_chat
"Explain quantum computing" → normal_chat
"What's the weather like?" → normal_chat
"How do I fix this Python error?" → normal_chat
"Tell me about the history of computers" → normal_chat"""

    def _get_cache_key(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a cache key for the message and context."""
        cache_input = message
        if context and context.get("previous_messages"):
            cache_input += "|" + "|".join(context["previous_messages"][-3:])
        return hashlib.md5(cache_input.lower().encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if a cache entry is still valid."""
        return time.time() - timestamp < self.cache_ttl
    
    def _get_from_cache(self, cache_key: str) -> Optional[tuple[UserIntent, float]]:
        """Get intent and confidence from cache if valid."""
        if cache_key in self._cache:
            intent, confidence, timestamp = self._cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.logger.debug(f"Cache hit for intent analysis: {intent.value}")
                return intent, confidence
            else:
                # Remove expired entry
                del self._cache[cache_key]
        return None
    
    def _store_in_cache(self, cache_key: str, intent: UserIntent, confidence: float):
        """Store intent analysis result in cache."""
        self._cache[cache_key] = (intent, confidence, time.time())
        # Simple cache cleanup - remove oldest entries if cache gets too large
        if len(self._cache) > 1000:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][2])
            del self._cache[oldest_key]
    


    async def analyze_intent(self, message: str) -> UserIntent:
        """
        Analyze user message to determine intent.
        
        Args:
            message: The user's input message
            context: Optional context from previous messages
            
        Returns:
            UserIntent: Either NORMAL_CHAT or IMAGE_GENERATION
        """
        # Check cache first
        cache_key = self._get_cache_key(message)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result[0]  # Return just the intent
        
        try:
            # Create a temporary agent for intent analysis
            agent = Agent(
                model=DeepSeek(
                    id="deepseek-chat",  # Use reasoning model for better intent detection
                    api_key=settings.deepseek_api_key,
                    temperature=0.1,  # Low temperature for consistent results
                    max_tokens=10     # We only need one word response
                ),
                instructions=self.system_prompt,
                markdown=False,
                stream=False,  # No need for streaming for intent analysis
            )
            
            # Get intent classification
            response = await agent.arun(message)
            
            # Extract the intent from response
            intent_text = response.content.strip().lower()
            
            self.logger.debug(f"Intent analysis for message '{message[:50]}...': {intent_text}")
            
            # Map response to intent enum
            if "image_generation" in intent_text:
                intent = UserIntent.IMAGE_GENERATION
            else:
                intent = UserIntent.NORMAL_CHAT
            
            # Store in cache with default confidence
            self._store_in_cache(cache_key, intent, 0.8)
            return intent
                
        except Exception as e:
            self.logger.error(f"Error in intent analysis: {str(e)}")
            # Fallback to normal chat on error to avoid breaking functionality
            return UserIntent.NORMAL_CHAT

    async def analyze_intent_with_confidence(self, message: str, context: Optional[Dict[str, Any]] = None) -> tuple[UserIntent, float]:
        """
        Analyze user message with confidence score.
        
        Args:
            message: The user's input message
            context: Optional context from previous messages
            
        Returns:
            Tuple of (UserIntent, confidence_score)
        """
        # Check cache first
        cache_key = self._get_cache_key(message, context)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result[0], cached_result[1]  # Return intent and confidence
        
        try:
            # Enhanced system prompt that also returns confidence
            confidence_system_prompt = self.system_prompt + """

Additionally, after your classification, add a confidence score from 0.0 to 1.0 on the next line.
Format your response exactly like this:
image_generation
0.95

or

normal_chat
0.80"""

            agent = Agent(
                model=DeepSeek(
                    id="deepseek-chat",
                    api_key=settings.deepseek_api_key,
                    temperature=0.1,
                    max_tokens=20
                ),
                instructions=confidence_system_prompt,
                markdown=False,
                stream=False,
            )
            
            analysis_message = message
            if context and context.get("previous_messages"):
                recent_context = " ".join(context["previous_messages"][-3:])
                analysis_message = f"Context: {recent_context}\n\nCurrent message: {message}"
            
            response = await agent.arun(analysis_message)
            lines = response.content.strip().split('\n')
            
            intent_text = lines[0].strip().lower()
            confidence = float(lines[1]) if len(lines) > 1 and lines[1].replace('.', '').isdigit() else 0.8
            
            intent = UserIntent.IMAGE_GENERATION if "image_generation" in intent_text else UserIntent.NORMAL_CHAT
            
            self.logger.debug(f"Intent analysis for message '{message[:50]}...': {intent.value} (confidence: {confidence})")
            
            # Store in cache
            self._store_in_cache(cache_key, intent, confidence)
            
            return intent, confidence
            
        except Exception as e:
            self.logger.error(f"Error in confidence-based intent analysis: {str(e)}")
            return UserIntent.NORMAL_CHAT, 0.5


# Global instance for reuse
_intent_service_instance: Optional[IntentAnalysisService] = None

def get_intent_service() -> IntentAnalysisService:
    """Get or create the global intent analysis service instance."""
    global _intent_service_instance
    if _intent_service_instance is None:
        _intent_service_instance = IntentAnalysisService()
    return _intent_service_instance
