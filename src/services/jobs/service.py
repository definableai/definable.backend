"""Job service for managing background jobs."""

from http import HTTPStatus
from typing import List, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC, InternalAuth
from models.auth_model import UserModel
from models.job_model import JobModel, JobStatus
from models.org_model import OrganizationMemberModel, OrganizationModel
from services.__base.acquire import Acquire

from .schema import (
  JobCreate,
  JobResponse,
  JobUpdate,
)


class JobService:
  """Job service for managing background jobs."""

  http_exposed = [
    "post=create",
    "get=get",
    "get=list",
    "get=list_org_jobs",
    "put=update",
    "delete=cancel",
    "post=update_status",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.ws_manager = acquire.ws_manager
    self.logger = acquire.logger

  async def post_create(
    self,
    org_id: UUID,
    job_data: JobCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("jobs", "write")),
  ) -> JobResponse:
    """Create a new job."""
    db_job = JobModel(
      name=job_data.name,
      description=job_data.description,
      created_by=user["id"],
      status=JobStatus.PENDING,
      context=job_data.context,
    )
    session.add(db_job)
    await session.commit()
    await session.refresh(db_job)
    return JobResponse.model_validate(db_job)

  async def get_get(
    self,
    org_id: UUID,
    job_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("jobs", "read")),
  ) -> JobResponse:
    """Get a job by ID."""
    query = select(JobModel).where(JobModel.id == job_id)
    result = await session.execute(query)
    job = result.scalar_one_or_none()

    if job is None:
      raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Job not found")

    return JobResponse.model_validate(job)

  async def get_list(
    self,
    org_id: UUID,
    status: Optional[JobStatus] = None,
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("jobs", "read")),
  ) -> List[JobResponse]:
    """List jobs for the current user. Returns empty list [] if no jobs found."""
    query = select(JobModel).where(JobModel.created_by == user["id"])

    if status is not None:
      query = query.where(JobModel.status == status)

    query = query.order_by(JobModel.created_at.desc()).offset(offset).limit(limit)

    result = await session.execute(query)
    jobs = result.scalars().all()

    # Return empty list if no jobs found
    if not jobs:
      return []

    return [JobResponse.model_validate(job) for job in jobs]

  async def get_list_org_jobs(
    self,
    org_id: UUID,
    status: Optional[JobStatus] = None,
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("jobs", "read")),
  ) -> List[JobResponse]:
    """List all jobs for the organization. Returns empty list [] if no jobs found."""
    # Query jobs for all users in the organization
    query = (
      select(JobModel)
      .join(UserModel, JobModel.created_by == UserModel.id)
      .join(OrganizationMemberModel, UserModel.id == OrganizationMemberModel.user_id)
      .where(OrganizationMemberModel.organization_id == org_id)
      .where(OrganizationMemberModel.status == "active")
    )

    if status is not None:
      query = query.where(JobModel.status == status)

    query = query.order_by(JobModel.created_at.desc()).offset(offset).limit(limit)

    result = await session.execute(query)
    jobs = result.scalars().all()

    # Return empty list if no jobs found
    if not jobs:
      return []

    return [JobResponse.model_validate(job) for job in jobs]

  async def put_update(
    self,
    org_id: UUID,
    job_id: UUID,
    job_data: JobUpdate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("jobs", "write")),
  ) -> JobResponse:
    """Update a job."""
    query = select(JobModel).where(JobModel.id == job_id)
    result = await session.execute(query)
    job = result.scalar_one_or_none()

    if job is None:
      raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Job not found")

    # Update fields
    update_data = job_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
      setattr(job, field, value)

    await session.commit()
    await session.refresh(job)
    return JobResponse.model_validate(job)

  async def delete_cancel(
    self,
    org_id: UUID,
    job_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("jobs", "delete")),
  ) -> dict:
    """Cancel a job."""
    query = select(JobModel).where(JobModel.id == job_id)
    result = await session.execute(query)
    job = result.scalar_one_or_none()

    if job is None:
      raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Job not found")

    # Only allow cancelling pending or processing jobs
    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
      raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Job cannot be cancelled")

    job.status = JobStatus.CANCELLED
    job.message = "Job cancelled by user"
    await session.commit()

    return {"message": "Job cancelled successfully"}

  async def post_update_status(
    self,
    request: Request,
    session: AsyncSession = Depends(get_db),
    auth_context: dict = Depends(InternalAuth()),
  ) -> dict:
    """Update the status of a job and broadcast the update to the client.

    This is a generic endpoint for internal services and background tasks to update job status.
    Requires internal token authentication.

    Expected payload:
    {
      "org_id": "uuid",
      "job_id": "uuid",
      "status": int (0=PENDING, 1=PROCESSING, 2=COMPLETED, 3=FAILED, 4=CANCELLED),
      "message": "optional status message",
      "data": {} // optional additional data for WebSocket broadcast
    }
    """
    try:
      # Parse and validate request payload
      payload = await request.json()

      # Validate required fields
      org_id = payload.get("org_id")
      job_id = payload.get("job_id")
      status = payload.get("status")

      if not org_id:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="org_id is required")

      if not job_id:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="job_id is required")

      if status is None:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="status is required")

      # Convert string IDs to UUIDs
      try:
        org_id = UUID(str(org_id))
        job_id = UUID(str(job_id))
      except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Invalid UUID format: {e}")

      # Validate status value
      try:
        status = int(status)
        if status not in [item.value for item in JobStatus]:
          raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Invalid status. Valid values: {[item.value for item in JobStatus]}")
        status_enum = JobStatus(status)
      except (ValueError, TypeError) as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Invalid status value: {e}")

      # Verify organization exists (internal services are trusted, no access control needed)
      org_query = select(OrganizationModel).where(OrganizationModel.id == org_id)
      org_result = await session.execute(org_query)
      organization = org_result.scalar_one_or_none()

      if not organization:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Organization not found")

      # Get and validate job exists
      job_query = select(JobModel).where(JobModel.id == job_id)
      job_result = await session.execute(job_query)
      job = job_result.scalar_one_or_none()

      if not job:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Job not found")

      # Validate status transitions (prevent invalid transitions)
      current_status = JobStatus(job.status)

      # Define valid transitions
      valid_transitions = {
        JobStatus.PENDING: [JobStatus.PROCESSING, JobStatus.FAILED, JobStatus.CANCELLED],
        JobStatus.PROCESSING: [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
        JobStatus.COMPLETED: [],  # Completed jobs cannot change status
        JobStatus.FAILED: [JobStatus.PENDING, JobStatus.PROCESSING],  # Allow retry
        JobStatus.CANCELLED: [JobStatus.PENDING, JobStatus.PROCESSING],  # Allow restart
      }

      if status_enum != current_status and status_enum not in valid_transitions[current_status]:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Invalid status transition from {current_status.name} to {status_enum.name}")

      # Update job status and message
      old_status = job.status
      job.status = status
      job.message = payload.get("message")

      await session.commit()
      await session.refresh(job)

      self.logger.info(
        f"Job {job_id} status updated from {JobStatus(old_status).name} to {status_enum.name} via internal service",
        extra={
          "job_id": str(job_id),
          "org_id": str(org_id),
          "old_status": old_status,
          "new_status": status,
          "auth_type": auth_context.get("auth_type"),
        },
      )

      # Prepare WebSocket broadcast data
      broadcast_data = {
        "job_id": str(job_id),
        "status": status,
        "status_name": status_enum.name,
        "message": job.message,
        "updated_at": job.updated_at.isoformat(),
        **(payload.get("data", {})),  # Merge additional data if provided
      }

      # Broadcast update to WebSocket clients
      await self.ws_manager.broadcast(
        org_id=str(org_id),
        data=broadcast_data,
        resource="kb",
        required_action="write",
      )

      return {
        "success": True,
        "message": "Job status updated successfully",
        "job_id": str(job_id),
        "status": status,
        "status_name": status_enum.name,
      }

    except HTTPException:
      # Re-raise HTTPExceptions without modification
      raise
    except Exception as e:
      self.logger.error(
        f"Unexpected error updating job status: {e}",
        extra={
          "job_id": payload.get("job_id") if "payload" in locals() else None,
          "org_id": payload.get("org_id") if "payload" in locals() else None,
          "error": str(e),
        },
      )
      raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Internal server error while updating job status")
