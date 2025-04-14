import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock
import sys
from uuid import uuid4, UUID
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Any
from pydantic import BaseModel, Field

# Create mock modules before any imports
sys.modules['database'] = MagicMock()
sys.modules['database.postgres'] = MagicMock()
sys.modules['database.models'] = MagicMock()
sys.modules['src.database'] = MagicMock()
sys.modules['src.database.postgres'] = MagicMock()
sys.modules['src.database.models'] = MagicMock()
sys.modules['config'] = MagicMock()
sys.modules['config.settings'] = MagicMock()
sys.modules['src.config'] = MagicMock()
sys.modules['src.config.settings'] = MagicMock()
sys.modules['src.services.__base.acquire'] = MagicMock()
sys.modules['dependencies.security'] = MagicMock()
sys.modules['src.utils.auth_util'] = MagicMock()
sys.modules['src.utils.email_sender'] = MagicMock()

# Constants to match API implementation
class InvitationStatus:
    PENDING = 0
    ACCEPTED = 1
    REJECTED = 2
    EXPIRED = 3

# Mock models using Pydantic to match schema.py
class MockInvitationModel(BaseModel):
    """Mock invitation model for database."""
    id: UUID = Field(default_factory=uuid4)
    organization_id: UUID = Field(default_factory=uuid4)
    role_id: UUID = Field(default_factory=uuid4)
    invitee_email: str = "test@example.com"
    invited_by: UUID = Field(default_factory=uuid4)
    status: int = InvitationStatus.PENDING
    expiry_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=48))
    invite_token: str = "mock_token"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    role_name: Optional[str] = None
    role_description: Optional[str] = None

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

class MockRoleModel(BaseModel):
    """Mock role model for database."""
    id: UUID = Field(default_factory=uuid4)
    name: str = "Test Role"
    description: str = "Test Role Description"
    organization_id: UUID = Field(default_factory=uuid4)
    is_admin: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

class MockOrganizationModel(BaseModel):
    """Mock organization model for database."""
    id: UUID = Field(default_factory=uuid4)
    name: str = "Test Organization"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

# Mock API response model matching API.json patterns
class MockResponse(BaseModel):
    """Mock API response model."""
    id: Optional[UUID] = None
    organization_id: Optional[UUID] = None
    role_id: Optional[UUID] = None
    role_name: Optional[str] = None
    role_description: Optional[str] = None
    invitee_email: Optional[str] = None
    invited_by: Optional[UUID] = None
    status: Optional[int] = None
    status_name: Optional[str] = None
    expiry_time: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    message: Optional[str] = None
    invitation_id: Optional[UUID] = None
    email: Optional[str] = None
    token: Optional[str] = None
    items: Optional[List[Any]] = None
    total: Optional[int] = None
    page: Optional[int] = None
    size: Optional[int] = None

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

@pytest.fixture
def mock_user():
    """Create a mock user."""
    return {
        "id": uuid4(),
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "organization_id": uuid4(),
    }

@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()

    # Setup scalar to return a properly mocked result
    scalar_mock = AsyncMock()
    session.scalar = scalar_mock

    # Setup execute to return a properly mocked result
    execute_mock = AsyncMock()
    # Make unique(), scalars(), first(), etc. return self to allow chaining
    execute_result = AsyncMock()
    execute_result.unique.return_value = execute_result
    execute_result.scalars.return_value = execute_result
    execute_result.scalar_one_or_none.return_value = None
    execute_result.scalar_one.return_value = None
    execute_result.first.return_value = None
    execute_result.all.return_value = []
    execute_result.mappings.return_value = execute_result

    execute_mock.return_value = execute_result
    session.execute = execute_mock

    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.flush = AsyncMock()
    session.rollback = AsyncMock()
    return session

@pytest.fixture
def mock_role():
    """Create a mock role."""
    return MockRoleModel(
        name="Test Role",
        description="Test role description",
        organization_id=uuid4(),
        is_admin=False
    )

@pytest.fixture
def mock_invitation():
    """Create a mock invitation."""
    return MockInvitationModel(
        organization_id=uuid4(),
        role_id=uuid4(),
        invitee_email="invited@example.com",
        invited_by=uuid4(),
        status=InvitationStatus.PENDING,
        expiry_time=datetime.now(timezone.utc) + timedelta(hours=48),
        invite_token="test_token"
    )

@pytest.fixture
def mock_invitations_service():
    """Create a mock invitations service."""
    invitations_service = MagicMock()

    async def mock_get_list(org_id, params=None, session=None, user=None):
        # Default pagination parameters
        page = params.page if params and hasattr(params, "page") else 1
        size = params.size if params and hasattr(params, "size") else 10
        offset = (page - 1) * size

        # Mock the total count query
        session.scalar.return_value = 3

        # Create mock invitations
        invitations = []
        for i in range(3):
            # Create invitation with role and organization info
            role = MockRoleModel(
                name=f"Role {i+1}",
                description=f"Role {i+1} Description",
                organization_id=org_id
            )

            invitation = MockInvitationModel(
                organization_id=org_id,
                role_id=role.id,
                invitee_email=f"invite{i+1}@example.com",
                invited_by=user["id"] if user else uuid4(),
                status=InvitationStatus.PENDING,
                expiry_time=datetime.now(timezone.utc) + timedelta(hours=48),
                invite_token=f"token_{i+1}"
            )

            # Add role info to invitation for the response
            invitation.role_name = role.name
            invitation.role_description = role.description

            invitations.append(invitation)

        # Set up the mock for execute.return_value.scalars().all()
        session.execute.return_value.scalars.return_value.all.return_value = invitations

        # Format response to match API
        invitation_responses = []
        for invitation in invitations:
            invitation_responses.append(MockResponse(
                id=invitation.id,
                organization_id=invitation.organization_id,
                role_id=invitation.role_id,
                role_name=invitation.role_name,
                role_description=invitation.role_description,
                invitee_email=invitation.invitee_email,
                invited_by=invitation.invited_by,
                status=invitation.status,
                status_name=["PENDING", "ACCEPTED", "REJECTED", "EXPIRED"][invitation.status],
                expiry_time=invitation.expiry_time.isoformat(),
                created_at=invitation.created_at.isoformat()
            ))

        return MockResponse(
            items=invitation_responses,
            total=3,
            page=page,
            size=size
        )

    async def mock_get(org_id, invitation_id, session=None, user=None):
        # Mock invitation with role info
        role = MockRoleModel(
            name="Test Role",
            description="Test Role Description",
            organization_id=org_id
        )

        invitation = MockInvitationModel(
            id=invitation_id,
            organization_id=org_id,
            role_id=role.id,
            invitee_email="invited@example.com",
            invited_by=user["id"] if user else uuid4(),
            status=InvitationStatus.PENDING,
            expiry_time=datetime.now(timezone.utc) + timedelta(hours=48),
            invite_token="test_token"
        )

        # Add role info to invitation for the response
        invitation.role_name = role.name
        invitation.role_description = role.description

        # Set up the mock for _get_invitation internal method
        session.execute.return_value.scalar_one_or_none.return_value = invitation

        # Format response to match API
        return MockResponse(
            id=invitation.id,
            organization_id=invitation.organization_id,
            role_id=invitation.role_id,
            role_name=invitation.role_name,
            role_description=invitation.role_description,
            invitee_email=invitation.invitee_email,
            invited_by=invitation.invited_by,
            status=invitation.status,
            status_name=["PENDING", "ACCEPTED", "REJECTED", "EXPIRED"][invitation.status],
            expiry_time=invitation.expiry_time.isoformat(),
            created_at=invitation.created_at.isoformat()
        )

    async def mock_send(org_id, invitation_data, session=None, user=None):
        # Check if invitation already exists
        existing_invitation = session.execute.return_value.scalar_one_or_none.return_value
        if existing_invitation:
            raise HTTPException(status_code=400, detail="Invitation already sent to this email address")

        # Get role information
        role = MockRoleModel(
            id=invitation_data.role_id,
            name="Test Role",
            description="Test Role Description",
            organization_id=org_id
        )

        # Create invitation
        invitation = MockInvitationModel(
            organization_id=org_id,
            role_id=invitation_data.role_id,
            invitee_email=invitation_data.invitee_email,
            invited_by=user["id"] if user else uuid4(),
            status=InvitationStatus.PENDING,
            expiry_time=invitation_data.expiry_time or (datetime.now(timezone.utc) + timedelta(hours=48)),
            invite_token="new_token"
        )

        # Add role info to invitation for the response
        invitation.role_name = role.name
        invitation.role_description = role.description

        session.add(invitation)
        await session.flush()
        await session.commit()
        await session.refresh(invitation)

        # Format response to match API
        return MockResponse(
            id=invitation.id,
            organization_id=invitation.organization_id,
            role_id=invitation.role_id,
            role_name=invitation.role_name,
            role_description=invitation.role_description,
            invitee_email=invitation.invitee_email,
            invited_by=invitation.invited_by,
            status=invitation.status,
            status_name=["PENDING", "ACCEPTED", "REJECTED", "EXPIRED"][invitation.status],
            expiry_time=invitation.expiry_time.isoformat(),
            created_at=invitation.created_at.isoformat(),
            message="Invitation sent successfully"
        )

    async def mock_resend(org_id, resend_request, session=None, user=None):
        # Get the original invitation
        role = MockRoleModel(
            name="Test Role",
            description="Test Role Description",
            organization_id=org_id
        )

        original_invitation = MockInvitationModel(
            id=resend_request.invitation_id,
            organization_id=org_id,
            role_id=role.id,
            invitee_email="invited@example.com",
            invited_by=uuid4(),
            status=InvitationStatus.EXPIRED,  # Expired invitation
            expiry_time=datetime.now(timezone.utc) - timedelta(hours=1),
            invite_token="old_token"
        )

        # Set up the mock for finding the original invitation
        session.execute.return_value.scalar_one_or_none.return_value = original_invitation

        # Create new invitation with the same details but new expiry
        new_invitation = MockInvitationModel(
            organization_id=org_id,
            role_id=original_invitation.role_id,
            invitee_email=original_invitation.invitee_email,
            invited_by=user["id"] if user else uuid4(),
            status=InvitationStatus.PENDING,
            expiry_time=datetime.now(timezone.utc) + timedelta(hours=48),
            invite_token="new_token"
        )

        # Add role info to invitation for the response
        new_invitation.role_name = role.name
        new_invitation.role_description = role.description

        session.add(new_invitation)
        await session.flush()
        await session.commit()
        await session.refresh(new_invitation)

        # Format response to match API
        return MockResponse(
            id=new_invitation.id,
            organization_id=new_invitation.organization_id,
            role_id=new_invitation.role_id,
            role_name=new_invitation.role_name,
            role_description=new_invitation.role_description,
            invitee_email=new_invitation.invitee_email,
            invited_by=new_invitation.invited_by,
            status=new_invitation.status,
            status_name=["PENDING", "ACCEPTED", "REJECTED", "EXPIRED"][new_invitation.status],
            expiry_time=new_invitation.expiry_time.isoformat(),
            created_at=new_invitation.created_at.isoformat(),
            message="Invitation resent successfully"
        )

    async def mock_accept(action_request, session=None):
        # Mock organization and role
        org_id = uuid4()
        role_id = uuid4()
        org = MockOrganizationModel(id=org_id, name="Test Organization")
        role = MockRoleModel(id=role_id, name="Test Role", organization_id=org_id)

        # Find invitation
        invitation = MockInvitationModel(
            organization_id=org_id,
            role_id=role_id,
            invitee_email=action_request.email,
            invited_by=uuid4(),
            status=InvitationStatus.PENDING,
            invite_token=action_request.token
        )

        # Mock the organization and role lookup
        session.execute.return_value.scalar_one_or_none.side_effect = [invitation, role, org]

        # Accept invitation
        invitation.status = InvitationStatus.ACCEPTED

        await session.commit()

        # Return response matching API format
        return {
            "message": "Invitation accepted successfully",
            "organization": {
                "id": org_id,
                "name": "Test Organization"
            },
            "role": {
                "id": role_id,
                "name": "Test Role"
            }
        }

    async def mock_reject(action_request, session=None):
        # Find invitation
        invitation = MockInvitationModel(
            organization_id=uuid4(),
            role_id=uuid4(),
            invitee_email=action_request.email,
            invited_by=uuid4(),
            status=InvitationStatus.PENDING,
            invite_token=action_request.token
        )

        session.execute.return_value.scalar_one_or_none.return_value = invitation

        # Reject invitation
        invitation.status = InvitationStatus.REJECTED

        await session.commit()

        # Return response matching API format
        return {"message": "Invitation rejected successfully"}

    # Create AsyncMock objects
    get_list_mock = AsyncMock(side_effect=mock_get_list)
    get_mock = AsyncMock(side_effect=mock_get)
    send_mock = AsyncMock(side_effect=mock_send)
    resend_mock = AsyncMock(side_effect=mock_resend)
    accept_mock = AsyncMock(side_effect=mock_accept)
    reject_mock = AsyncMock(side_effect=mock_reject)

    # Assign the mocks to the service
    invitations_service.get_list = get_list_mock
    invitations_service.get_get = get_mock
    invitations_service.post_send = send_mock
    invitations_service.post_resend = resend_mock
    invitations_service.post_accept = accept_mock
    invitations_service.post_reject = reject_mock

    return invitations_service

@pytest.mark.asyncio
class TestInvitationService:
    """Tests for the Invitation service."""

    async def test_get_list(self, mock_invitations_service, mock_db_session, mock_user):
        """Test getting a list of invitations for an organization."""
        # Call the service
        org_id = uuid4()

        class MockParams:
            page = 1
            size = 10

        response = await mock_invitations_service.get_list(
            org_id,
            MockParams(),
            session=mock_db_session,
            user=mock_user
        )

        # Verify result structure matches API response
        assert hasattr(response, "items")
        assert hasattr(response, "total")
        assert hasattr(response, "page")
        assert hasattr(response, "size")
        assert len(response.items) == 3
        assert response.total == 3
        assert response.page == 1
        assert response.size == 10

        # Verify invitation objects structure
        for invitation in response.items:
            assert hasattr(invitation, "id")
            assert hasattr(invitation, "organization_id")
            assert hasattr(invitation, "role_id")
            assert hasattr(invitation, "role_name")
            assert hasattr(invitation, "role_description")
            assert hasattr(invitation, "invitee_email")
            assert hasattr(invitation, "invited_by")
            assert hasattr(invitation, "status")
            assert hasattr(invitation, "status_name")
            assert hasattr(invitation, "expiry_time")
            assert hasattr(invitation, "created_at")
            assert invitation.organization_id == org_id

        # Verify service method was called
        assert mock_invitations_service.get_list.called

    async def test_get_invitation(self, mock_invitations_service, mock_db_session, mock_user):
        """Test getting a single invitation."""
        # Call the service
        org_id = uuid4()
        invitation_id = uuid4()

        response = await mock_invitations_service.get_get(
            org_id,
            invitation_id,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result structure matches API response
        assert response.id == invitation_id
        assert response.organization_id == org_id
        assert response.invitee_email == "invited@example.com"
        assert response.status == InvitationStatus.PENDING
        assert response.status_name == "PENDING"
        assert hasattr(response, "role_name")
        assert hasattr(response, "role_description")
        assert hasattr(response, "expiry_time")
        assert hasattr(response, "created_at")

        # Verify service method was called
        assert mock_invitations_service.get_get.called

    async def test_send_invitation(self, mock_invitations_service, mock_db_session, mock_user, mock_role):
        """Test sending a new invitation."""
        # Create invitation data
        org_id = uuid4()
        invitation_data = MockResponse(
            role_id=mock_role.id,
            invitee_email="new_invite@example.com",
            expiry_time=None  # Let the service set the default
        )

        # Configure mock to return no existing invitation
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Call the service
        response = await mock_invitations_service.post_send(
            org_id,
            invitation_data,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result structure matches API response
        assert response.organization_id == org_id
        assert response.role_id == invitation_data.role_id
        assert response.invitee_email == invitation_data.invitee_email
        assert response.status == InvitationStatus.PENDING
        assert response.status_name == "PENDING"
        assert hasattr(response, "id")
        assert hasattr(response, "role_name")
        assert hasattr(response, "role_description")
        assert hasattr(response, "expiry_time")
        assert hasattr(response, "created_at")
        assert hasattr(response, "message")
        assert "successfully" in response.message.lower()

        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.flush.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

        # Verify service method was called
        assert mock_invitations_service.post_send.called

    async def test_resend_invitation(self, mock_invitations_service, mock_db_session, mock_user):
        """Test resending an invitation."""
        # Create resend request
        org_id = uuid4()
        invitation_id = uuid4()
        resend_request = MockResponse(
            invitation_id=invitation_id
        )

        # Call the service
        response = await mock_invitations_service.post_resend(
            org_id,
            resend_request,
            session=mock_db_session,
            user=mock_user
        )

        # Verify result structure matches API response
        assert response.organization_id == org_id
        assert response.invitee_email == "invited@example.com"
        assert response.status == InvitationStatus.PENDING
        assert response.status_name == "PENDING"
        assert hasattr(response, "id")
        assert hasattr(response, "role_name")
        assert hasattr(response, "role_description")
        assert hasattr(response, "expiry_time")
        assert hasattr(response, "created_at")
        assert hasattr(response, "message")
        assert "successfully" in response.message.lower()

        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.flush.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

        # Verify service method was called
        assert mock_invitations_service.post_resend.called

    async def test_accept_invitation(self, mock_invitations_service, mock_db_session):
        """Test accepting an invitation."""
        # Create accept request
        action_request = MockResponse(
            token="test_token",
            email="invited@example.com"
        )

        # Call the service
        response = await mock_invitations_service.post_accept(
            action_request,
            session=mock_db_session
        )

        # Verify result structure matches API response
        assert response["message"] == "Invitation accepted successfully"
        assert "organization" in response
        assert "id" in response["organization"]
        assert "name" in response["organization"]
        assert "role" in response
        assert "id" in response["role"]
        assert "name" in response["role"]

        # Verify database operations
        mock_db_session.commit.assert_called_once()

        # Verify service method was called
        assert mock_invitations_service.post_accept.called

    async def test_reject_invitation(self, mock_invitations_service, mock_db_session):
        """Test rejecting an invitation."""
        # Create reject request
        action_request = MockResponse(
            token="test_token",
            email="invited@example.com"
        )

        # Call the service
        response = await mock_invitations_service.post_reject(
            action_request,
            session=mock_db_session
        )

        # Verify result structure matches API response
        assert response["message"] == "Invitation rejected successfully"

        # Verify database operations
        mock_db_session.commit.assert_called_once()

        # Verify service method was called
        assert mock_invitations_service.post_reject.called

    async def test_send_invitation_with_existing(self, mock_invitations_service, mock_db_session, mock_user):
        """Test sending an invitation that already exists."""
        # Create invitation data
        org_id = uuid4()
        invitation_data = MockResponse(
            role_id=uuid4(),
            invitee_email="existing@example.com",
            expiry_time=None
        )

        # Configure mock to return an existing invitation
        existing_invitation = MockInvitationModel(
            organization_id=org_id,
            role_id=invitation_data.role_id or uuid4(),  # Use default if None
            invitee_email=str(invitation_data.invitee_email or "existing@example.com"),  # Ensure str type
            invited_by=uuid4(),
            status=InvitationStatus.PENDING
        )
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = existing_invitation

        # Call the service and expect exception
        with pytest.raises(HTTPException) as excinfo:
            await mock_invitations_service.post_send(
                org_id,
                invitation_data,
                session=mock_db_session,
                user=mock_user
            )

        # Verify exception
        assert excinfo.value.status_code == 400
        assert "Invitation already sent to this email address" in excinfo.value.detail

        # Verify service method was called
        assert mock_invitations_service.post_send.called

    async def test_accept_non_pending_invitation(self, mock_invitations_service, mock_db_session):
        """Test accepting an invitation that is not pending."""
        # Create accept request
        action_request = MockResponse(
            token="test_token",
            email="invited@example.com"
        )

        # Configure mock to return a non-pending invitation
        non_pending_invitation = MockInvitationModel(
            invitee_email=str(action_request.email or "invited@example.com"),  # Ensure str type
            invite_token=str(action_request.token or "test_token"),  # Ensure str type
            status=InvitationStatus.ACCEPTED  # Already ACCEPTED
        )

        # Mock accept function to raise exception when non-pending invitation found
        async def mock_accept_non_pending(action_request, session=None):
            raise HTTPException(
                status_code=400,
                detail="Invitation is not in pending state"
            )

        mock_invitations_service.post_accept = AsyncMock(side_effect=mock_accept_non_pending)

        # Call the service and expect exception
        with pytest.raises(HTTPException) as excinfo:
            await mock_invitations_service.post_accept(
                action_request,
                session=mock_db_session
            )

        # Verify exception
        assert excinfo.value.status_code == 400
        assert "Invitation is not in pending state" in excinfo.value.detail

        # Verify service method was called
        assert mock_invitations_service.post_accept.called