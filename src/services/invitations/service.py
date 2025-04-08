import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from dependencies.security import RBAC
from services.__base.acquire import Acquire
from utils.email import EmailUtil

from models import OrganizationMemberModel, OrganizationModel, InvitationModel, RoleModel
from .schema import (
  InvitationCreate,
  InvitationResponse,
  InvitationResendRequest,
  InvitationListParams,
  InvitationListResponse,
  InvitationStatus,
  InvitationActionRequest,
)


class InvitationService:
  """Invitation service for managing organization invitations."""

  http_exposed = [
    "post=send",
    "post=resend",
    "post=accept",
    "post=reject",
    "get=list",
    "get=get",
    "get=reject",
  ]

  def __init__(self, acquire: Acquire):
    """Initialize service."""
    self.acquire = acquire
    self.utils = acquire.utils
    self.settings = acquire.settings
    self.logger = acquire.logger
    self.email_util = EmailUtil()  # Create an instance of EmailUtil

  async def post_send(
    self,
    org_id: UUID,
    invitation_data: InvitationCreate,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("invitations", "write")),
  ) -> InvitationResponse:
    self.logger.info(f"Creating invitation for email: {invitation_data.invitee_email}")

    # Validate inviter's role
    await self._validate_inviter_role(user["id"], org_id, invitation_data.role_id, session)

    # Check if invitation already exists
    existing_invitation = await self._get_pending_invitation(org_id, invitation_data.invitee_email, session)

    if existing_invitation:
      self.logger.error(f"Invitation already exists for email: {invitation_data.invitee_email}")
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invitation already sent to this email address",
      )

    # Generate invite token
    invite_token = secrets.token_urlsafe(32)

    # Set expiry time (48 hours from now)
    expiry_time = invitation_data.expiry_time or datetime.now(timezone.utc) + timedelta(hours=48)

    # Create invitation
    invitation = InvitationModel(
      organization_id=org_id,
      role_id=invitation_data.role_id,
      invitee_email=invitation_data.invitee_email,
      invited_by=user["id"],
      status=InvitationStatus.PENDING,
      expiry_time=expiry_time,
      invite_token=invite_token,
    )

    session.add(invitation)
    await session.flush()

    # Send invitation email
    try:
      await self._send_invitation_email(invitation, org_id, session)
    except Exception as e:
      await session.rollback()
      self.logger.error(f"Failed to send invitation email: {str(e)}")
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to send invitation email: {str(e)}",
      )

    await session.commit()
    await session.refresh(invitation)

    self.logger.info(f"Invitation created successfully: {invitation.id}")
    return InvitationResponse.model_validate(invitation)

  async def post_resend(
    self,
    org_id: UUID,
    resend_request: InvitationResendRequest,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("invitations", "write")),
  ) -> InvitationResponse:
    self.logger.info(f"Resending invitation: {resend_request.invitation_id}")

    # Get original invitation
    original_invitation = await self._get_invitation(resend_request.invitation_id, session)

    # Check if invitation exists
    if not original_invitation:
      self.logger.error(f"Original invitation not found: {resend_request.invitation_id}")
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Original invitation not found",
      )

    # Check if original invitation is pending
    if original_invitation.status != int(InvitationStatus.PENDING):
      self.logger.error(f"Cannot resend non-pending invitation: {original_invitation.id}")
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Only pending invitations can be resent",
      )

    # Validate inviter's role
    await self._validate_inviter_role(user["id"], org_id, original_invitation.role_id, session)

    # Generate new invite token
    invite_token = secrets.token_urlsafe(32)

    # Set expiry time (48 hours from now)
    expiry_time = datetime.now(timezone.utc) + timedelta(hours=48)

    # Create new invitation
    new_invitation = InvitationModel(
      organization_id=original_invitation.organization_id,
      role_id=original_invitation.role_id,
      invitee_email=original_invitation.invitee_email,
      invited_by=user["id"],  # Update to current user as inviter
      status=InvitationStatus.PENDING,
      expiry_time=expiry_time,
      invite_token=invite_token,
    )

    try:
      session.add(new_invitation)
      await session.flush()

      # Send invitation email
      try:
        await self._send_invitation_email(new_invitation, org_id, session)
      except Exception as e:
        await session.rollback()
        self.logger.error(f"Failed to send invitation email: {str(e)}")
        raise HTTPException(
          status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
          detail=f"Failed to send invitation email: {str(e)}",
        )

      await session.commit()
      await session.refresh(new_invitation)

      self.logger.info(f"New invitation created and sent successfully: {new_invitation.id}")
      return InvitationResponse.model_validate(new_invitation)

    except Exception as e:
      await session.rollback()
      self.logger.error(f"Failed to create new invitation: {str(e)}")
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to create new invitation: {str(e)}",
      )

  async def get_list(
    self,
    org_id: UUID,
    params: InvitationListParams = Depends(),
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("invitations", "read")),
  ) -> InvitationListResponse:
    # Build query
    query = select(InvitationModel)

    # Apply filters
    query = query.where(InvitationModel.organization_id == org_id)

    if params.status:
      query = query.where(InvitationModel.status == int(params.status))

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await session.execute(count_query)
    total_count = total.scalar() or 0

    # Apply pagination
    query = query.limit(params.size).offset((params.page - 1) * params.size)

    # Execute query
    result = await session.execute(query)
    invitations = list(result.scalars().all())

    return InvitationListResponse(
      items=[InvitationResponse.model_validate(invite) for invite in invitations],
      total=total_count,
      page=params.page,
      size=params.size,
    )

  async def get_get(
    self,
    org_id: UUID,
    invitation_id: UUID,
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(RBAC("invitations", "read")),
  ) -> InvitationResponse:
    # First get the invitation to get its organization_id
    invitation = await self._get_invitation(invitation_id, session)

    if not invitation:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Invitation not found",
      )

    # Check if user has access to this invitation
    if invitation.organization_id != org_id:
      raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Access denied",
      )

    return InvitationResponse.model_validate(invitation)

  async def post_accept(
    self,
    action_request: InvitationActionRequest,
    session: AsyncSession = Depends(get_db),
  ) -> dict:
    self.logger.info(f"Accepting invitation for email: {action_request.email} with token: {action_request.token}")

    # Find invitation
    query = select(InvitationModel).where(
      and_(
        InvitationModel.invite_token == action_request.token,
        InvitationModel.invitee_email == action_request.email,
      )
    )
    result = await session.execute(query)
    invitation = result.unique().scalar_one_or_none()

    if not invitation:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Invitation not found",
      )

    # Check if invitation is pending
    if invitation.status != int(InvitationStatus.PENDING):
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invitation is not in pending state (current status: {InvitationStatus(invitation.status).name})",
      )

    # Update invitation status
    invitation.status = int(InvitationStatus.ACCEPTED)
    session.add(invitation)
    await session.commit()

    return {"message": "Invitation accepted successfully"}

  async def post_reject(
    self,
    action_request: InvitationActionRequest,
    session: AsyncSession = Depends(get_db),
  ) -> dict:
    self.logger.info(f"Rejecting invitation for email: {action_request.email} with token: {action_request.token}")

    # Find invitation
    query = select(InvitationModel).where(
      and_(
        InvitationModel.invite_token == action_request.token,
        InvitationModel.invitee_email == action_request.email,
      )
    )
    result = await session.execute(query)
    invitation = result.unique().scalar_one_or_none()

    if not invitation:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Invitation not found",
      )

    # Check if invitation is pending
    if invitation.status != int(InvitationStatus.PENDING):
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invitation is not in pending state (current status: {InvitationStatus(invitation.status).name})",
      )

    # Update invitation status
    invitation.status = int(InvitationStatus.REJECTED)
    session.add(invitation)
    await session.commit()

    return {"message": "Invitation rejected successfully"}

  async def get_reject(
    self,
    token: str,
    email: str,
    session: AsyncSession = Depends(get_db),
  ) -> HTMLResponse:
    """
    Handle reject link click from email.

    Args:
        token: Invitation token
        email: Invitee email
        session: Database session

    Returns:
        HTML response
    """
    self.logger.info(f"Rejecting invitation via link for email: {email} with token: {token}")

    # Find invitation
    query = select(InvitationModel).where(
      and_(
        InvitationModel.invite_token == token,
        InvitationModel.invitee_email == email,
      )
    )
    result = await session.execute(query)
    invitation = result.unique().scalar_one_or_none()

    if not invitation:  # Add this null check first
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid invitation",
        )

    # Get organization name for the response
    org_name = "the organization"
    if invitation and invitation.organization_id:
      org = await OrganizationModel.read(invitation.organization_id)
      if org:
        org_name = org.name

    # Update invitation status to REJECTED
    invitation.status = int(InvitationStatus.REJECTED)
    session.add(invitation)
    await session.commit()

    # Return success message
    return HTMLResponse(
      content=f"""
            <html>
                <head>
                    <title>Invitation Rejected</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }}
                        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                        h1 {{ color: #e74c3c; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Invitation Rejected</h1>
                        <p>You have successfully rejected the invitation to join {org_name}.</p>
                    </div>
                </body>
            </html>
            """,
      status_code=200,
    )

  # Private methods

  async def _validate_inviter_role(
    self,
    user_id: UUID,
    org_id: UUID,
    role_id: UUID,
    session: AsyncSession,
  ) -> None:
    # Get user's role in the organization
    query = select(OrganizationMemberModel).where(
      and_(
        OrganizationMemberModel.user_id == user_id,
        OrganizationMemberModel.organization_id == org_id,
        OrganizationMemberModel.status == "active",
      )
    )
    result = await session.execute(query)
    member = result.unique().scalar_one_or_none()

    if not member:
      raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="User not active in organization",
      )

    # Get role being assigned
    role_query = select(RoleModel).where(
      and_(
        RoleModel.id == role_id,
        RoleModel.organization_id == org_id,
      )
    )
    role_result = await session.execute(role_query)
    role = role_result.unique().scalar_one_or_none()

    if not role:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Role not found",
      )

    # Get inviter's role
    inviter_role_query = select(RoleModel).where(RoleModel.id == member.role_id)
    inviter_role_result = await session.execute(inviter_role_query)
    inviter_role = inviter_role_result.unique().scalar_one_or_none()

    if not inviter_role:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Inviter's role not found",
      )

    # Validate hierarchy
    if inviter_role.hierarchy_level <= role.hierarchy_level:
      raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Cannot invite user with equal or higher role level",
      )

  async def _get_pending_invitation(
    self,
    organization_id: UUID,
    invitee_email: str,
    session: AsyncSession,
  ) -> Optional[InvitationModel]:
    """Get pending invitation by organization and email."""
    query = select(InvitationModel).where(
      and_(
        InvitationModel.organization_id == organization_id,
        InvitationModel.invitee_email == invitee_email,
        InvitationModel.status == int(InvitationStatus.PENDING),
      )
    )
    result = await session.execute(query)
    return result.unique().scalar_one_or_none()

  async def _get_invitation(
    self,
    invitation_id: UUID,
    session: AsyncSession,
  ) -> Optional[InvitationModel]:
    """Get invitation by ID."""
    query = select(InvitationModel).where(InvitationModel.id == invitation_id)
    result = await session.execute(query)
    return result.unique().scalar_one_or_none()

  async def _send_invitation_email(self, invitation: InvitationModel, org_id: UUID, session: AsyncSession) -> None:
    try:
      org = await OrganizationModel.read(org_id)
      if not org:
        raise ValueError("Organization not found")

      accept_url = f"{self.settings.frontend_url}/api/auth/signup/invite?token={invitation.invite_token}&email={invitation.invitee_email}"
      reject_url = f"{self.settings.frontend_url}/api/invitations/reject?token={invitation.invite_token}&email={invitation.invitee_email}"

      await self.email_util.send_invitation_email(
        email=invitation.invitee_email,
        organization_name=org.name,
        invite_token=invitation.invite_token,
        accept_url=accept_url,
        reject_url=reject_url,
      )
    except Exception as e:
      await session.rollback()
      self.logger.error(f"Failed to send invitation email: {str(e)}")
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to send invitation email: {str(e)}",
      )
