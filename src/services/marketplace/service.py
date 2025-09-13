from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database import get_db
from dependencies.security import RBAC, JWTBearer
from models import AgentModel, LLMModel, UserModel
from models.agent_model import AgentCategoryModel
from models.llm_category_model import LLMCategoryModel, LLMModelCategoryModel
from models.marketplace_model import MarketplaceAssistantModel, MarketplaceReviewModel
from services.__base.acquire import Acquire

from .schema import (
  MarketplaceAssistantDetail,
  MarketplaceAssistantItem,
  MarketplaceAssistantsResponse,
  MarketplaceCategoriesResponse,
  MarketplaceCategory,
  MarketplaceFeaturedResponse,
  MarketplacePricing,
  MarketplaceReview,
  MarketplaceReviewCreate,
  MarketplaceReviewsResponse,
  MarketplaceSpecifications,
)


class MarketplaceService:
  """Marketplace service for managing assistant marketplace data."""

  http_exposed = [
    "get=assistants",
    "get=assistant_detail",
    "get=categories",
    "get=featured",
    "get=reviews",
    "post=create_review",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.logger = acquire.logger

  async def get_assistants(
    self,
    org_id: UUID,
    category: Optional[str] = None,
    search: Optional[str] = None,
    offset: int = 0,
    limit: Optional[int] = None,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("marketplace", "read")),
  ) -> MarketplaceAssistantsResponse:
    """Get marketplace assistants with filtering and pagination."""
    try:
      # Base query for published assistants
      query = (
        select(MarketplaceAssistantModel)
        .where(MarketplaceAssistantModel.is_published)
        .order_by(desc(MarketplaceAssistantModel.rating_avg), desc(MarketplaceAssistantModel.conversation_count))
      )

      # Apply category filter
      if category and category != "all":
        if category == "models":
          query = query.where(MarketplaceAssistantModel.assistant_type == "llm_model")
        elif category == "agents":
          query = query.where(MarketplaceAssistantModel.assistant_type == "agent")
        elif category == "featured":
          query = query.where(MarketplaceAssistantModel.is_featured)
        elif category == "trending":
          # Trending: high conversation count in recent period
          query = query.where(MarketplaceAssistantModel.conversation_count >= 10)
        elif category == "top":
          # Top rated
          query = query.where(MarketplaceAssistantModel.rating_avg >= 4.5)
        else:
          # Specific category
          query = query.where(MarketplaceAssistantModel.category == category)

      # Apply search filter
      if search:
        # We'll search in the associated agent/model names
        search_term = f"%{search.lower()}%"

        # Subquery for matching LLM models
        llm_subquery = select(LLMModel.id).where(or_(func.lower(LLMModel.name).like(search_term), func.lower(LLMModel.provider).like(search_term)))

        # Subquery for matching agents
        agent_subquery = select(AgentModel.id).where(
          or_(func.lower(AgentModel.name).like(search_term), func.lower(AgentModel.description).like(search_term))
        )

        query = query.where(
          or_(
            and_(MarketplaceAssistantModel.assistant_type == "llm_model", MarketplaceAssistantModel.assistant_id.in_(llm_subquery)),
            and_(MarketplaceAssistantModel.assistant_type == "agent", MarketplaceAssistantModel.assistant_id.in_(agent_subquery)),
          )
        )

      # Get total count
      count_query = select(func.count()).select_from(query.subquery())
      total_result = await session.execute(count_query)
      total = total_result.scalar() or 0

      # Apply pagination
      paginated_query = query.offset(offset)
      if limit is not None:
        paginated_query = paginated_query.limit(limit)

      result = await session.execute(paginated_query)
      marketplace_assistants = result.scalars().all()

      # Transform to response items
      assistants = []
      for marketplace_assistant in marketplace_assistants:
        assistant_item = await self._create_assistant_item(marketplace_assistant, session)
        assistants.append(assistant_item)

      has_more = limit is not None and offset + limit < total

      return MarketplaceAssistantsResponse(assistants=assistants, total=total, has_more=has_more)

    except Exception as e:
      self.logger.error(f"Error getting marketplace assistants: {e}")
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error retrieving marketplace assistants")

  async def get_assistant_detail(
    self, assistant_id: str, session: AsyncSession = Depends(get_db), user: dict = Depends(RBAC("marketplace", "read"))
  ) -> MarketplaceAssistantDetail:
    """Get detailed assistant information."""
    try:
      # Find the marketplace assistant
      query = (
        select(MarketplaceAssistantModel)
        .options(selectinload(MarketplaceAssistantModel.reviews), selectinload(MarketplaceAssistantModel.usage))
        .where(MarketplaceAssistantModel.id == UUID(assistant_id), MarketplaceAssistantModel.is_published)
      )

      result = await session.execute(query)
      marketplace_assistant = result.scalar_one_or_none()

      if not marketplace_assistant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assistant not found")

      # Create detailed response
      detail = await self._create_assistant_detail(marketplace_assistant, session, user.get("id"))
      return detail

    except HTTPException:
      raise
    except Exception as e:
      self.logger.error(f"Error getting assistant detail: {e}")
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error retrieving assistant details")

  async def get_categories(self, session: AsyncSession = Depends(get_db), user: dict = Depends(JWTBearer())) -> MarketplaceCategoriesResponse:
    """Get marketplace categories with counts."""
    try:
      # Get LLM category counts
      llm_category_query = (
        select(LLMCategoryModel.name, func.count(LLMModelCategoryModel.model_id).label("count"))
        .join(LLMModelCategoryModel, LLMCategoryModel.id == LLMModelCategoryModel.category_id)
        .join(
          MarketplaceAssistantModel,
          and_(
            MarketplaceAssistantModel.assistant_id == LLMModelCategoryModel.model_id,
            MarketplaceAssistantModel.assistant_type == "llm_model",
            MarketplaceAssistantModel.is_published,
          ),
        )
        .where(LLMModelCategoryModel.is_primary, LLMCategoryModel.is_active)
        .group_by(LLMCategoryModel.name)
        .order_by(func.count(LLMModelCategoryModel.model_id).desc())
      )

      # Get Agent category counts
      agent_category_query = (
        select(AgentCategoryModel.name, func.count(AgentModel.id).label("count"))
        .join(AgentModel, AgentCategoryModel.id == AgentModel.category_id)
        .join(
          MarketplaceAssistantModel,
          and_(
            MarketplaceAssistantModel.assistant_id == AgentModel.id,
            MarketplaceAssistantModel.assistant_type == "agent",
            MarketplaceAssistantModel.is_published,
          ),
        )
        .where(AgentCategoryModel.is_active)
        .group_by(AgentCategoryModel.name)
        .order_by(func.count(AgentModel.id).desc())
      )

      llm_result = await session.execute(llm_category_query)
      agent_result = await session.execute(agent_category_query)

      llm_categories = llm_result.all()
      agent_categories = agent_result.all()

      categories = []
      total_count = 0

      # Add LLM categories
      for category_name, count in llm_categories:
        categories.append(MarketplaceCategory(id=category_name, name=category_name.replace("-", " ").title(), count=count))
        total_count += count

      # Add Agent categories
      for category_name, count in agent_categories:
        categories.append(MarketplaceCategory(id=category_name, name=category_name.replace("-", " ").title(), count=count))
        total_count += count

      # Add "All" category at the beginning
      categories.insert(0, MarketplaceCategory(id="all", name="All", count=total_count))

      return MarketplaceCategoriesResponse(categories=categories)

    except Exception as e:
      self.logger.error(f"Error getting categories: {e}")
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error retrieving categories")

  async def get_featured(
    self, category: Optional[str] = None, session: AsyncSession = Depends(get_db), user: dict = Depends(JWTBearer())
  ) -> MarketplaceFeaturedResponse:
    """Get featured assistants by category using weighted calculation."""
    try:
      # Initialize all categories
      categories: Dict[str, List[Dict[str, Any]]] = {
        "advanced-language-models": [],
        "ai-agents": [],
        "custom-ai-solutions": [],
        "specialized-ai-agents": [],
      }

      # Calculate featured for advanced-language-models (LLM models)
      if not category or category == "advanced-language-models":
        # Weighted featured calculation with multiple normalization methods
        # Method 1: Fixed divisor (IMPLEMENTED)
        featured_query = (
          select(
            MarketplaceAssistantModel,
            # Featured score calculation with comments for other methods
            (MarketplaceAssistantModel.rating_avg * 0.7 + (MarketplaceAssistantModel.conversation_count / 100.0) * 0.3).label("featured_score"),
          )
          .where(
            MarketplaceAssistantModel.is_published,
            MarketplaceAssistantModel.assistant_type == "llm_model",
            MarketplaceAssistantModel.rating_count >= 4,  # Minimum reviews for statistical significance
          )
          .order_by(desc("featured_score"))
          .limit(3)
        )

        # Method 2: Percentage-based normalization (COMMENTED)
        # max_conversations_subquery = select(func.max(MarketplaceAssistantModel.conversation_count))
        # featured_score = (rating_avg * 0.7) + ((conversation_count / max_conversations) * 5 * 0.3)

        # Method 3: Log scale normalization (COMMENTED)
        # featured_score = (rating_avg * 0.7) + (func.log(conversation_count + 1) * 0.3)

        featured_result = await session.execute(featured_query)
        featured_data = featured_result.all()

        for position, (marketplace_assistant, score) in enumerate(featured_data, 1):
          assistant_item = await self._create_assistant_item(marketplace_assistant, session)

          # Add position and score to response
          assistant_dict = assistant_item.model_dump()
          assistant_dict["position"] = position
          assistant_dict["featured_score"] = round(float(score), 2)

          categories["advanced-language-models"].append(assistant_dict)

      # Future categories (empty for now as requested)
      # ai-agents, custom-ai-solutions, specialized-ai-agents will remain empty arrays

      # Filter response based on requested category
      if category:
        if category in categories:
          return MarketplaceFeaturedResponse(categories={category: categories[category]})
        else:
          return MarketplaceFeaturedResponse(categories={category: []})

      # Return all categories
      return MarketplaceFeaturedResponse(categories=categories)

    except Exception as e:
      self.logger.error(f"Error getting featured assistants: {e}")
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error retrieving featured assistants")

  async def get_reviews(
    self, assistant_id: str, offset: int = 0, limit: int = 10, session: AsyncSession = Depends(get_db), user: dict = Depends(JWTBearer())
  ) -> MarketplaceReviewsResponse:
    """Get reviews for a specific assistant."""
    try:
      # Get reviews with user info
      query = (
        select(MarketplaceReviewModel, UserModel.first_name, UserModel.last_name, UserModel.email)
        .join(UserModel, MarketplaceReviewModel.user_id == UserModel.id)
        .where(MarketplaceReviewModel.marketplace_assistant_id == UUID(assistant_id))
        .order_by(desc(MarketplaceReviewModel.created_at))
        .offset(offset)
        .limit(limit)
      )

      result = await session.execute(query)
      review_data = result.all()

      # Get total count and rating stats
      stats_query = select(
        func.count(MarketplaceReviewModel.id).label("total"),
        func.avg(MarketplaceReviewModel.rating).label("avg_rating"),
        func.count().filter(MarketplaceReviewModel.rating == 5).label("rating_5"),
        func.count().filter(MarketplaceReviewModel.rating == 4).label("rating_4"),
        func.count().filter(MarketplaceReviewModel.rating == 3).label("rating_3"),
        func.count().filter(MarketplaceReviewModel.rating == 2).label("rating_2"),
        func.count().filter(MarketplaceReviewModel.rating == 1).label("rating_1"),
      ).where(MarketplaceReviewModel.marketplace_assistant_id == UUID(assistant_id))

      stats_result = await session.execute(stats_query)
      stats = stats_result.first()

      # Transform reviews
      reviews = []
      for review_model, first_name, last_name, email in review_data:
        # Handle empty strings and None values properly
        first = (first_name or "").strip()
        last = (last_name or "").strip()

        # Try to create a meaningful username
        if first and last:
          user_name = f"{first} {last}"
        elif first:
          user_name = first
        elif last:
          user_name = last
        elif email:
          # Use email prefix as fallback (before @ symbol)
          user_name = email.split("@")[0].replace(".", " ").replace("_", " ").title()
        else:
          user_name = "Anonymous"
        reviews.append(
          MarketplaceReview(
            id=str(review_model.id),
            user_id=str(review_model.user_id),
            user_name=user_name,
            rating=review_model.rating,
            title=review_model.title,
            content=review_model.content,
            created_at=review_model.created_at.isoformat(),
          )
        )

      # Calculate rating breakdown
      if stats:
        total_reviews = stats.total or 0
        if total_reviews > 0:
          rating_breakdown = {
            "5": round((stats.rating_5 / total_reviews) * 100, 1),
            "4": round((stats.rating_4 / total_reviews) * 100, 1),
            "3": round((stats.rating_3 / total_reviews) * 100, 1),
            "2": round((stats.rating_2 / total_reviews) * 100, 1),
            "1": round((stats.rating_1 / total_reviews) * 100, 1),
          }
          average_rating = float(stats.avg_rating or 0)
        else:
          rating_breakdown = {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0}
          average_rating = 0.0
      else:
        total_reviews = 0
        rating_breakdown = {"5": 0, "4": 0, "3": 0, "2": 0, "1": 0}
        average_rating = 0.0

      return MarketplaceReviewsResponse(
        reviews=reviews, total_reviews=total_reviews, rating_breakdown=rating_breakdown, average_rating=round(average_rating, 1)
      )

    except Exception as e:
      self.logger.error(f"Error getting reviews: {e}")
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error retrieving reviews")

  async def post_create_review(
    self,
    assistant_id: str,
    review_data: MarketplaceReviewCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("marketplace", "write")),
  ) -> MarketplaceReview:
    """Create a new review for a specific assistant (allows multiple reviews per user per assistant)."""
    try:
      # Validate assistant_id is a proper UUID
      try:
        assistant_uuid = UUID(assistant_id)
      except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid assistant_id format: {assistant_id}. Must be a valid UUID.")

      # Verify the marketplace assistant exists and is published
      assistant_query = select(MarketplaceAssistantModel).where(
        MarketplaceAssistantModel.id == assistant_uuid, MarketplaceAssistantModel.is_published
      )
      assistant_result = await session.execute(assistant_query)
      marketplace_assistant = assistant_result.scalar_one_or_none()

      if not marketplace_assistant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assistant not found or not published")

      user_id = UUID(user["id"])

      # Create a new review (allow multiple reviews per user per assistant)
      new_review = MarketplaceReviewModel(
        marketplace_assistant_id=assistant_uuid,
        user_id=user_id,
        rating=review_data.rating,
        title=review_data.title,
        content=review_data.content,
      )

      session.add(new_review)
      await session.commit()
      await session.refresh(new_review)

      # Get user info for response
      user_query = select(UserModel).where(UserModel.id == user_id)
      user_result = await session.execute(user_query)
      user_model = user_result.scalar_one()

      return MarketplaceReview(
        id=str(new_review.id),
        user_id=str(new_review.user_id),
        user_name=f"{user_model.first_name} {user_model.last_name}".strip() or "Anonymous",
        rating=new_review.rating,
        title=new_review.title,
        content=new_review.content,
        created_at=new_review.created_at.isoformat(),
      )

    except HTTPException:
      raise
    except Exception as e:
      self.logger.error(f"Error creating/updating review: {e}")
      await session.rollback()
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error creating/updating review")

  async def _create_assistant_item(self, marketplace_assistant: MarketplaceAssistantModel, session: AsyncSession) -> MarketplaceAssistantItem:
    """Create marketplace assistant item from database model."""
    # Get the actual assistant data (LLM or Agent)
    if marketplace_assistant.assistant_type == "llm_model":
      query = select(LLMModel).where(LLMModel.id == marketplace_assistant.assistant_id)
      result = await session.execute(query)
      assistant_model = result.scalar_one_or_none()

      name = assistant_model.name if assistant_model else "Unknown Model"
      description = f"{assistant_model.provider} language model" if assistant_model else "LLM Model"
      provider = assistant_model.provider if assistant_model else "unknown"
      tags = assistant_model.tags if assistant_model and assistant_model.tags else []

      # Extract specifications from model_metadata
      specifications = None
      if assistant_model and assistant_model.model_metadata and isinstance(assistant_model.model_metadata, dict):
        specs_data = assistant_model.model_metadata.get("specifications", {})
        if specs_data:
          # Handle max_output_tokens - it might be an int or dict with 'default'/'maximum'
          max_output_tokens = specs_data.get("max_output_tokens")
          if isinstance(max_output_tokens, dict):
            max_output_tokens = max_output_tokens.get("default") or max_output_tokens.get("maximum")

          # Handle context_window - might also be a dict
          context_window = specs_data.get("context_window")
          if isinstance(context_window, dict):
            context_window = context_window.get("default") or context_window.get("maximum")

          specifications = MarketplaceSpecifications(
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            pricing_per_token=specs_data.get("pricing_per_token"),
            capabilities=specs_data.get("capabilities"),
            performance_tier=specs_data.get("performance_tier"),
            knowledge_cutoff=specs_data.get("knowledge_cutoff"),
            training_cutoff=specs_data.get("training_cutoff"),
            model_version=specs_data.get("model_version"),
            reasoning_score=specs_data.get("reasoning_score"),
            speed_score=specs_data.get("speed_score"),
            modalities=specs_data.get("modalities"),
            features=specs_data.get("features"),
            thinking_mode=specs_data.get("thinking_mode"),
            reasoning_token_support=specs_data.get("reasoning_token_support"),
            rate_limits=specs_data.get("rate_limits"),
            endpoints=specs_data.get("endpoints"),
          )

      # Get primary category for LLM models
      category_query = (
        select(LLMCategoryModel.name)
        .join(LLMModelCategoryModel, LLMCategoryModel.id == LLMModelCategoryModel.category_id)
        .where(LLMModelCategoryModel.model_id == marketplace_assistant.assistant_id, LLMModelCategoryModel.is_primary)
        .limit(1)
      )
      category_result = await session.execute(category_query)
      category = category_result.scalar() or "advanced-language-models"

    else:
      query = select(AgentModel).where(AgentModel.id == marketplace_assistant.assistant_id)
      result = await session.execute(query)
      assistant_model = result.scalar_one_or_none()

      name = assistant_model.name if assistant_model else "Unknown Agent"
      description = assistant_model.description or "Custom AI agent" if assistant_model else "Agent"
      provider = "user"
      tags = assistant_model.tags if assistant_model and assistant_model.tags else []
      specifications = None  # Agents don't have specifications

      # Get category for agents
      if assistant_model and assistant_model.category_id:
        category_query = select(AgentCategoryModel.name).where(AgentCategoryModel.id == assistant_model.category_id)
        category_result = await session.execute(category_query)
        category = category_result.scalar_one_or_none() or "general"
      else:
        category = "general"

    # Get is_active status from the underlying model/agent
    is_active = True  # Default for agents
    if marketplace_assistant.assistant_type == "llm_model" and assistant_model:
      is_active = assistant_model.is_active
    elif marketplace_assistant.assistant_type == "agent" and assistant_model:
      # For agents, we can use is_active if the field exists, otherwise default to True
      is_active = getattr(assistant_model, "is_active", True)

    return MarketplaceAssistantItem(
      id=str(marketplace_assistant.id),
      assistant_id=str(marketplace_assistant.assistant_id),
      name=name,
      description=description,
      type="core" if marketplace_assistant.assistant_type == "llm_model" else "agent",
      provider=provider,
      developer=provider,
      verified=False,  # TODO: Add verification system
      rating=float(marketplace_assistant.rating_avg),
      review_count=marketplace_assistant.rating_count,
      conversation_count=str(marketplace_assistant.conversation_count),
      category=category,
      is_featured=False,  # Featured is now calculated dynamically
      is_new=False,  # TODO: Calculate based on created_at
      is_popular=marketplace_assistant.conversation_count >= 10,
      is_active=is_active,  # Include actual model/agent active status
      tags=tags,
      pricing=MarketplacePricing(type=marketplace_assistant.pricing_type, price=None),
      icon=None,  # TODO: Add icon support
      screenshots=None,  # TODO: Add screenshots support
      marketplace_features=None,  # TODO: Add features support
      integrations=None,  # TODO: Add integrations support
      specifications=specifications,
    )

  async def _create_assistant_detail(
    self, marketplace_assistant: MarketplaceAssistantModel, session: AsyncSession, user_id: Optional[str] = None
  ) -> MarketplaceAssistantDetail:
    """Create detailed marketplace assistant from database model."""
    # Create base assistant item
    base_item = await self._create_assistant_item(marketplace_assistant, session)

    # Get reviews (limited)
    review_query = (
      select(MarketplaceReviewModel, UserModel.first_name, UserModel.last_name, UserModel.email)
      .join(UserModel, MarketplaceReviewModel.user_id == UserModel.id)
      .where(MarketplaceReviewModel.marketplace_assistant_id == marketplace_assistant.id)
      .order_by(desc(MarketplaceReviewModel.created_at))
      .limit(5)
    )

    review_result = await session.execute(review_query)
    review_data = review_result.all()

    reviews = []
    for review_model, first_name, last_name, email in review_data:
      # Handle empty strings and None values properly
      first = (first_name or "").strip()
      last = (last_name or "").strip()

      # Try to create a meaningful username
      if first and last:
        user_name = f"{first} {last}"
      elif first:
        user_name = first
      elif last:
        user_name = last
      elif email:
        # Use email prefix as fallback (before @ symbol)
        user_name = email.split("@")[0].replace(".", " ").replace("_", " ").title()
      else:
        user_name = "Anonymous"
      reviews.append(
        MarketplaceReview(
          id=str(review_model.id),
          user_id=str(review_model.user_id),
          user_name=user_name,
          rating=review_model.rating,
          title=review_model.title,
          content=review_model.content,
          created_at=review_model.created_at.isoformat(),
        )
      )

    # Get user context if user_id provided
    user_context = {}
    if user_id:
      # Check if user has reviewed this assistant
      user_review_query = select(MarketplaceReviewModel).where(
        MarketplaceReviewModel.marketplace_assistant_id == marketplace_assistant.id, MarketplaceReviewModel.user_id == UUID(user_id)
      )
      user_review_result = await session.execute(user_review_query)
      user_review = user_review_result.scalar_one_or_none()

      user_context = {
        "user_review": {"rating": user_review.rating, "title": user_review.title, "created_at": user_review.created_at.isoformat()}
        if user_review
        else None,
        "can_review": True,  # Allow multiple reviews per user per assistant
      }

    return MarketplaceAssistantDetail(
      **base_item.model_dump(),
      overview=None,  # TODO: Add overview support
      reviews=reviews,
      mcp_servers=[],  # TODO: Add MCP server support
      tool_categories=[],  # TODO: Add tool categories support
      user_context=user_context,
    )
