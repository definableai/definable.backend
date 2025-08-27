"""Job service schemas."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field

from models.job_model import JobStatus


class JobCreate(BaseModel):
  """Job creation schema."""

  name: str = Field(..., min_length=1, max_length=100)
  description: Optional[str] = Field(None, description="Job description")
  context: Optional[Dict[str, Any]] = Field(None, description="Job context")


class JobResponse(BaseModel):
  """Job response schema."""

  id: UUID
  name: str
  description: Optional[str]
  created_by: UUID
  status: JobStatus
  message: Optional[str]
  created_at: datetime
  updated_at: datetime
  context: Optional[Dict[str, Any]] = None
  parent_job_id: Optional[UUID] = None

  class Config:
    from_attributes = True


class JobUpdate(BaseModel):
  """Job update schema."""

  status: Optional[JobStatus] = None
  message: Optional[str] = None


class JobRestartRequest(BaseModel):
  """Request to restart a job."""

  override_context: Optional[Dict[str, Any]] = Field(None, description="Context overrides")
  description: Optional[str] = Field(None, description="New job description")


class JobChainRequest(BaseModel):
  """Request to create a job chain."""

  jobs: List[Dict[str, Any]] = Field(..., description="List of job configurations")


class JobDetailsResponse(BaseModel):
  """Detailed job information response."""

  job: Dict[str, Any]
  trail: List[Dict[str, Any]]
  trail_summary: Dict[str, Any]


class JobTrailResponse(BaseModel):
  """Job trail response."""

  job_id: str
  trail: List[Dict[str, Any]]
  summary: Dict[str, Any]


class TaskTypeResponse(BaseModel):
  """Available task types response."""

  task_types: List[str]
  registry_info: Dict[str, Any]


class RestartRecommendationsResponse(BaseModel):
  """Job restart recommendations response."""

  can_restart: bool
  job_info: Dict[str, Any]
  trail_info: Dict[str, Any]
  context_analysis: Dict[str, Any]
  restart_strategy: str


class JobChainResponse(BaseModel):
  """Job chain creation response."""

  chain_id: str
  jobs: List[Dict[str, Any]]
  total_jobs: int
  status: str
