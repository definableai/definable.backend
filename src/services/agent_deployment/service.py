from datetime import datetime
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC, APIKeyAuth
from models import AgentDeploymentLogModel, AgentDeploymentTraceModel, AgentModel
from services.__base.acquire import Acquire

from .schema import (
  LogFilter,
  LogResponse,
  PaginatedLogResponse,
  PaginatedTraceResponse,
  TraceFilter,
  TraceResponse,
  WebhookLogPayload,
  WebhookResponse,
  WebhookTracePayload,
)


class AgentDeploymentService:
  """Agent deployment webhook service."""

  http_exposed = [
    "post=webhook_logs",
    "post=webhook_traces",
    "get=logs",
    "get=traces",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.logger = acquire.logger
    self.ws_manager = acquire.ws_manager

  def _convert_to_naive_utc(self, dt: datetime) -> datetime:
    """Convert timezone-aware datetime to naive UTC datetime."""
    if dt.tzinfo is not None:
      # Convert to UTC and make naive
      return dt.astimezone(datetime.now().astimezone().tzinfo).replace(tzinfo=None)
    return dt

  async def post_webhook_logs(
    self,
    payload: WebhookLogPayload,
    session: AsyncSession = Depends(get_db),
    auth: dict = Depends(APIKeyAuth()),
  ) -> WebhookResponse:
    """Process incoming deployment log webhook."""
    try:
      # Verify agent exists and get organization_id
      agent_query = select(AgentModel).where(AgentModel.id == payload.agent_id)
      agent_result = await session.execute(agent_query)
      agent = agent_result.scalar_one_or_none()

      if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent with id {payload.agent_id} not found")

      # Verify API key has access to this organization
      if auth.get("org_id") != str(agent.organization_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="API key does not have access to this agent's organization")

      # Convert timezone-aware datetime to naive UTC
      naive_timestamp = self._convert_to_naive_utc(payload.timestamp)

      # Get user_id and api_key_id from auth context (from API key)
      user_id = UUID(auth.get("user_id"))
      api_key_id = UUID(auth.get("api_key_id"))

      # Create log entry
      log_entry = AgentDeploymentLogModel(
        agent_id=payload.agent_id,
        organization_id=agent.organization_id,
        user_id=user_id,  # Use user_id from API key auth
        api_key_id=api_key_id,  # Use api_key_id from API key auth
        deployment_id=payload.deployment_id,
        log_type=payload.log_type,
        log_level=payload.log_level,
        message=payload.message,
        log_metadata=payload.metadata,
        timestamp=naive_timestamp,  # Use converted naive datetime
        source=payload.source,
      )

      session.add(log_entry)
      await session.commit()
      await session.refresh(log_entry)

      # Broadcast real-time event via WebSocket
      await self._broadcast_log_event(agent.organization_id, log_entry)

      self.logger.info(
        "Agent deployment log processed",
        agent_id=str(payload.agent_id),
        deployment_id=payload.deployment_id,
        log_type=payload.log_type,
        user_id=str(user_id),
        api_key_id=str(api_key_id),
      )

      return WebhookResponse()

    except HTTPException:
      raise
    except Exception as e:
      print("error", e)
      self.logger.error("Error processing log webhook", error=str(e))
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process log webhook")

  async def post_webhook_traces(
    self,
    payload: WebhookTracePayload,
    session: AsyncSession = Depends(get_db),
    auth: dict = Depends(APIKeyAuth()),
  ) -> WebhookResponse:
    """Process incoming deployment trace webhook."""
    try:
      # Verify agent exists and get organization_id
      agent_query = select(AgentModel).where(AgentModel.id == payload.agent_id)
      agent_result = await session.execute(agent_query)
      agent = agent_result.scalar_one_or_none()

      if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent with id {payload.agent_id} not found")

      # Verify API key has access to this organization
      if auth.get("org_id") != str(agent.organization_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="API key does not have access to this agent's organization")

      # Convert timezone-aware datetimes to naive UTC
      naive_start_time = self._convert_to_naive_utc(payload.start_time)
      naive_end_time = self._convert_to_naive_utc(payload.end_time) if payload.end_time else None

      # Calculate duration if both times provided
      duration_ms = None
      if naive_end_time and naive_start_time:
        duration_ms = int((naive_end_time - naive_start_time).total_seconds() * 1000)

      # Get user_id and api_key_id from auth context (from API key)
      user_id = UUID(auth.get("user_id"))
      api_key_id = UUID(auth.get("api_key_id"))

      # Create trace entry
      trace_entry = AgentDeploymentTraceModel(
        agent_id=payload.agent_id,
        organization_id=agent.organization_id,
        user_id=user_id,  # Use user_id from API key auth
        api_key_id=api_key_id,  # Use api_key_id from API key auth
        deployment_id=payload.deployment_id,
        trace_id=payload.trace_id,
        span_id=payload.span_id,
        parent_span_id=payload.parent_span_id,
        operation_name=payload.operation_name,
        start_time=naive_start_time,  # Use converted naive datetime
        end_time=naive_end_time,  # Use converted naive datetime
        duration_ms=duration_ms or payload.duration_ms,
        status=payload.status,
        tags=payload.tags,
        trace_logs=payload.logs,
      )

      session.add(trace_entry)
      await session.commit()
      await session.refresh(trace_entry)

      # Broadcast real-time event via WebSocket
      await self._broadcast_trace_event(agent.organization_id, trace_entry)

      self.logger.info(
        "Agent deployment trace processed",
        agent_id=str(payload.agent_id),
        deployment_id=payload.deployment_id,
        trace_id=payload.trace_id,
        user_id=str(user_id),
        api_key_id=str(api_key_id),
      )

      return WebhookResponse()

    except HTTPException:
      raise
    except Exception as e:
      self.logger.error("Error processing trace webhook", error=str(e))
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process trace webhook")

  async def get_logs(
    self,
    filters: LogFilter = Depends(),
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("agents", "read")),
  ) -> PaginatedLogResponse:
    """Get deployment logs with filtering and pagination."""
    try:
      # Build base query
      query = select(AgentDeploymentLogModel).join(AgentModel)
      count_query = select(func.count(AgentDeploymentLogModel.id)).join(AgentModel)

      # Apply organization filter for security
      org_id = user.get("org_id")
      if org_id:
        query = query.where(AgentModel.organization_id == org_id)
        count_query = count_query.where(AgentModel.organization_id == org_id)

      # Apply filters
      conditions = []

      if filters.agent_id:
        conditions.append(AgentDeploymentLogModel.agent_id == filters.agent_id)
      if filters.user_id:
        conditions.append(AgentDeploymentLogModel.user_id == filters.user_id)
      if filters.api_key_id:
        conditions.append(AgentDeploymentLogModel.api_key_id == filters.api_key_id)
      if filters.deployment_id:
        conditions.append(AgentDeploymentLogModel.deployment_id == filters.deployment_id)
      if filters.log_type:
        conditions.append(AgentDeploymentLogModel.log_type == filters.log_type)
      if filters.log_level:
        conditions.append(AgentDeploymentLogModel.log_level == filters.log_level)
      if filters.source:
        conditions.append(AgentDeploymentLogModel.source == filters.source)
      if filters.start_time:
        # Convert timezone-aware filter to naive for comparison
        naive_start = self._convert_to_naive_utc(filters.start_time)
        conditions.append(AgentDeploymentLogModel.timestamp >= naive_start)
      if filters.end_time:
        # Convert timezone-aware filter to naive for comparison
        naive_end = self._convert_to_naive_utc(filters.end_time)
        conditions.append(AgentDeploymentLogModel.timestamp <= naive_end)

      if conditions:
        query = query.where(and_(*conditions))
        count_query = count_query.where(and_(*conditions))

      # Get total count
      total_result = await session.execute(count_query)
      total = total_result.scalar() or 0

      # Apply pagination and ordering
      query = query.order_by(desc(AgentDeploymentLogModel.timestamp))
      query = query.offset(filters.skip).limit(filters.limit)

      # Execute query
      result = await session.execute(query)
      logs = result.scalars().all()

      # Convert to response format
      log_responses = [LogResponse.from_orm(log) for log in logs]

      return PaginatedLogResponse(
        logs=log_responses,
        total=total,
        skip=filters.skip,
        limit=filters.limit,
      )

    except Exception as e:
      self.logger.error("Error retrieving deployment logs", error=str(e))
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve deployment logs")

  async def get_traces(
    self,
    filters: TraceFilter = Depends(),
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("agents", "read")),
  ) -> PaginatedTraceResponse:
    """Get deployment traces with filtering and pagination."""
    try:
      # Build base query
      query = select(AgentDeploymentTraceModel).join(AgentModel)
      count_query = select(func.count(AgentDeploymentTraceModel.id)).join(AgentModel)

      # Apply organization filter for security
      org_id = user.get("org_id")
      if org_id:
        query = query.where(AgentModel.organization_id == org_id)
        count_query = count_query.where(AgentModel.organization_id == org_id)

      # Apply filters
      conditions = []

      if filters.agent_id:
        conditions.append(AgentDeploymentTraceModel.agent_id == filters.agent_id)
      if filters.user_id:
        conditions.append(AgentDeploymentTraceModel.user_id == filters.user_id)
      if filters.api_key_id:
        conditions.append(AgentDeploymentTraceModel.api_key_id == filters.api_key_id)
      if filters.deployment_id:
        conditions.append(AgentDeploymentTraceModel.deployment_id == filters.deployment_id)
      if filters.trace_id:
        conditions.append(AgentDeploymentTraceModel.trace_id == filters.trace_id)
      if filters.status:
        conditions.append(AgentDeploymentTraceModel.status == filters.status)
      if filters.operation_name:
        conditions.append(AgentDeploymentTraceModel.operation_name.ilike(f"%{filters.operation_name}%"))
      if filters.start_time:
        # Convert timezone-aware filter to naive for comparison
        naive_start = self._convert_to_naive_utc(filters.start_time)
        conditions.append(AgentDeploymentTraceModel.start_time >= naive_start)
      if filters.end_time:
        # Convert timezone-aware filter to naive for comparison
        naive_end = self._convert_to_naive_utc(filters.end_time)
        conditions.append(AgentDeploymentTraceModel.start_time <= naive_end)

      if conditions:
        query = query.where(and_(*conditions))
        count_query = count_query.where(and_(*conditions))

      # Get total count
      total_result = await session.execute(count_query)
      total = total_result.scalar() or 0

      # Apply pagination and ordering
      query = query.order_by(desc(AgentDeploymentTraceModel.start_time))
      query = query.offset(filters.skip).limit(filters.limit)

      # Execute query
      result = await session.execute(query)
      traces = result.scalars().all()

      # Convert to response format
      trace_responses = [TraceResponse.from_orm(trace) for trace in traces]

      return PaginatedTraceResponse(
        traces=trace_responses,
        total=total,
        skip=filters.skip,
        limit=filters.limit,
      )

    except Exception as e:
      self.logger.error("Error retrieving deployment traces", error=str(e))
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve deployment traces")

  async def _broadcast_log_event(self, org_id: UUID, log_entry: AgentDeploymentLogModel) -> None:
    """Broadcast log event via WebSocket."""
    try:
      event_data = {
        "type": "log",
        "id": str(log_entry.id),
        "agent_id": str(log_entry.agent_id),
        "user_id": str(log_entry.user_id),
        "api_key_id": str(log_entry.api_key_id),
        "deployment_id": log_entry.deployment_id,
        "log_type": log_entry.log_type,
        "log_level": log_entry.log_level,
        "message": log_entry.message,
        "timestamp": log_entry.timestamp.isoformat(),
        "source": log_entry.source,
      }

      await self.ws_manager.broadcast(
        org_id=org_id,
        data=event_data,
        resource="agents",
        required_action="write",
      )
    except Exception as e:
      self.logger.error("Failed to broadcast log event", error=str(e))

  async def _broadcast_trace_event(self, org_id: UUID, trace_entry: AgentDeploymentTraceModel) -> None:
    """Broadcast trace event via WebSocket."""
    try:
      event_data = {
        "type": "trace",
        "id": str(trace_entry.id),
        "agent_id": str(trace_entry.agent_id),
        "user_id": str(trace_entry.user_id),
        "api_key_id": str(trace_entry.api_key_id),
        "deployment_id": trace_entry.deployment_id,
        "trace_id": trace_entry.trace_id,
        "span_id": trace_entry.span_id,
        "operation_name": trace_entry.operation_name,
        "status": trace_entry.status,
        "start_time": trace_entry.start_time.isoformat(),
        "end_time": trace_entry.end_time.isoformat() if trace_entry.end_time else None,
        "duration_ms": trace_entry.duration_ms,
      }

      await self.ws_manager.broadcast(
        org_id=org_id,
        data=event_data,
        resource="agents",
        required_action="write",
      )
    except Exception as e:
      self.logger.error("Failed to broadcast trace event", error=str(e))
