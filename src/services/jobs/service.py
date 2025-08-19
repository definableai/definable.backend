"""Job service for managing background jobs."""

from http import HTTPStatus
from typing import List, Optional
from uuid import UUID

from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC
from models.auth_model import UserModel
from models.job_model import JobModel, JobStatus
from models.org_model import OrganizationMemberModel
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
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
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

