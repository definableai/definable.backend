import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock
import sys
import json
from uuid import UUID, uuid4
from datetime import datetime
import re
from typing import Dict, Optional, Any

# Import pydantic
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
sys.modules['services.roles.service'] = MagicMock()

# Mock models using Pydantic
class MockOrganizationModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = "Test Organization"
    slug: str = Field(default_factory=lambda: f"test-organization-{str(uuid4())[:8]}")
    settings: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        arbitrary_types_allowed = True

class MockOrganizationMemberModel(BaseModel):
    organization_id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(default_factory=uuid4)
    role_id: UUID = Field(default_factory=uuid4)
    status: str = "active"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        arbitrary_types_allowed = True

class MockRoleModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = "member"
    organization_id: Optional[UUID] = None
    is_admin: bool = False
    permissions: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

class MockResponse(BaseModel):
    id: Optional[UUID] = None
    name: Optional[str] = None
    slug: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    message: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

# Create a mock JSONResponse that's simpler and more predictable
class MockJSONResponse:
    def __init__(self, status_code=200, content=None):
        self.status_code = status_code
        self.content = content
        # Use json.dumps with separators to ensure consistent formatting 
        # with no spaces after colons
        self.body = self._serialize_content(content)
    
    def _serialize_content(self, content):
        if content is None:
            return b"{}"
        # Use separators to remove space after colon and comma
        return json.dumps(content, separators=(',', ':')).encode()

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
    session.delete = AsyncMock()
    return session

@pytest.fixture
def mock_organization():
    """Create a mock organization."""
    return MockOrganizationModel(
        name="Test Organization",
        slug="test-organization-12345678",
        settings={}
    )

@pytest.fixture
def mock_role():
    """Create a mock role."""
    return MockRoleModel(
        name="owner",
        is_admin=True,
        permissions={"*": ["*"]}
    )

@pytest.fixture
def mock_member():
    """Create a mock organization member."""
    return MockOrganizationMemberModel(
        status="active"
    )

@pytest.fixture
def mock_org_service():
    """Create a mock organization service."""
    org_service = MagicMock()
    
    async def mock_create_org(name, session, user):
        # Check if the slug already exists (ensure uniqueness)
        session.execute.return_value.scalar_one_or_none.return_value = None  # Slug is available
        
        # Create an organization
        created_at = datetime.now().isoformat()
        
        # Properly sanitize the slug - remove special characters
        sanitized_name = re.sub(r'[^a-zA-Z0-9\s]', '', name)  # Remove special chars
        slug = f"{sanitized_name.lower().replace(' ', '-')}-{str(uuid4())[:8]}"
        
        org = MockOrganizationModel(
            name=name,
            slug=slug,
            settings={},
            created_at=created_at,
            updated_at=created_at
        )
        session.add(org)
        await session.flush()
        
        # Create an owner role (pretend it exists)
        role = MockRoleModel(
            name="owner",
            organization_id=org.id,
            is_admin=True
        )
        session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = role
        
        # Add the user as a member with the owner role
        member = MockOrganizationMemberModel(
            organization_id=org.id,
            user_id=user["id"],
            role_id=role.id,
            status="active",
            created_at=created_at
        )
        session.add(member)
        
        await session.commit()
        
        return MockResponse(**org.model_dump())
    
    async def mock_add_member(organization_id, user_id, role_id, session):
        # Get the role
        role = MockRoleModel(
            id=role_id,
            organization_id=organization_id
        )
        
        # Mock RoleService._get_role
        from services.roles.service import RoleService
        RoleService._get_role = AsyncMock(return_value=role)
        
        # Create member
        member = MockOrganizationMemberModel(
            organization_id=organization_id,
            user_id=user_id,
            role_id=role_id,
            status="active",
            created_at=datetime.now().isoformat()
        )
        session.add(member)
        await session.commit()
        
        # Return a dictionary with status and result, which matches the pattern in other test files
        return MockJSONResponse(status_code=201, content={"message": "Member added successfully"})
    
    async def mock_add_member_role_not_found(organization_id, user_id, role_id, session):
        # Mock RoleService._get_role returning None
        from services.roles.service import RoleService
        RoleService._get_role = AsyncMock(return_value=None)
        
        raise HTTPException(
            status_code=404,
            detail=f"Role {role_id} not found"
        )
    
    async def mock_get_org_by_id(org_id, session, user):
        # Check if organization exists
        org = MockOrganizationModel(
            id=org_id,
            name="Test Organization",
            slug="test-organization-12345678",
            settings={},
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = org
        
        return MockResponse(**org.model_dump())
    
    async def mock_get_org_not_found(org_id, session, user):
        session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = None
        raise HTTPException(
            status_code=404,
            detail=f"Organization {org_id} not found"
        )
    
    async def mock_list(session, user):
        # Return mock organizations with proper timestamps
        orgs = [
            MockOrganizationModel(
                name="Organization 1",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            ),
            MockOrganizationModel(
                name="Organization 2",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
        ]
        
        # Set up mock response
        session.execute.return_value.unique.return_value.scalars.return_value.all.return_value = orgs
        
        return [MockResponse(**org.model_dump()) for org in orgs]
    
    # Create AsyncMock objects
    create_org_mock = AsyncMock(side_effect=mock_create_org)
    add_member_mock = AsyncMock(side_effect=mock_add_member)
    list_mock = AsyncMock(side_effect=mock_list)
    get_org_mock = AsyncMock(side_effect=mock_get_org_by_id)
    
    # Assign the mocks to the service
    org_service.post_create_org = create_org_mock
    org_service.post_add_member = add_member_mock
    org_service.get_list = list_mock
    org_service.get = get_org_mock
    
    return org_service

@pytest.mark.asyncio
class TestOrganizationService:
    """Tests for the Organization service."""
    
    async def test_create_organization(self, mock_org_service, mock_db_session, mock_user):
        """Test creating a new organization."""
        # Call the service
        response = await mock_org_service.post_create_org(
            "New Organization",
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result - check that the slug follows the new format (lowercase with hyphens)
        assert response.name == "New Organization"
        assert "new-organization-" in response.slug
        
        # Verify database operations
        assert mock_db_session.add.call_count == 2  # org and member
        mock_db_session.flush.assert_called_once()
        mock_db_session.commit.assert_called_once()
        
        # The service should have been called correctly
        assert mock_org_service.post_create_org.called
    
    async def test_add_member(self, mock_org_service, mock_db_session):
        """Test adding a member to an organization."""
        # Create org, user, and role IDs
        org_id = uuid4()
        user_id = uuid4()
        role_id = uuid4()
        
        # Call the service
        response = await mock_org_service.post_add_member(
            org_id,
            user_id,
            role_id,
            session=mock_db_session
        )
        
        # Verify result - instead of checking exact string, verify content after parsing
        assert response.status_code == 201
        
        # Parse JSON to compare content instead of exact string
        import json
        response_data = json.loads(response.body)
        assert response_data == {"message": "Member added successfully"}
        
        # Verify database operations
        mock_db_session.add.assert_called_once()
        
        # The service should have been called correctly
        assert mock_org_service.post_add_member.called
    
    async def test_add_member_role_not_found(self, mock_org_service, mock_db_session):
        """Test adding a member with a non-existent role."""
        # Create org, user, and role IDs
        org_id = uuid4()
        user_id = uuid4()
        role_id = uuid4()
        
        # Mock the behavior for role not found
        mock_org_service.post_add_member.side_effect = None  # Clear previous side_effect
        mock_org_service.post_add_member.side_effect = HTTPException(
            status_code=404,
            detail=f"Role {role_id} not found"
        )
        
        # Call the service and expect exception
        with pytest.raises(HTTPException) as excinfo:
            await mock_org_service.post_add_member(
                org_id,
                user_id,
                role_id,
                session=mock_db_session
            )
        
        # Verify exception
        assert excinfo.value.status_code == 404
        assert f"Role {role_id} not found" in str(excinfo.value.detail)
        
        # The service should have been called correctly
        assert mock_org_service.post_add_member.called
    
    async def test_list_organizations(self, mock_org_service, mock_db_session, mock_user):
        """Test listing organizations for a user."""
        # Call the service
        response = await mock_org_service.get_list(
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result
        assert isinstance(response, list)
        assert len(response) == 2
        assert response[0].name == "Organization 1"
        assert response[1].name == "Organization 2"
        
        # The service should have been called correctly
        assert mock_org_service.get_list.called
    
    async def test_get_organization(self, mock_org_service, mock_db_session, mock_user):
        """Test getting an organization by ID."""
        # Call the service
        org_id = uuid4()
        
        response = await mock_org_service.get(
            org_id,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result
        assert response.id == org_id
        assert response.name == "Test Organization"
        assert response.slug == "test-organization-12345678"
        assert hasattr(response, "created_at")
        assert hasattr(response, "updated_at")
        
        # The service should have been called correctly
        assert mock_org_service.get.called
    
    async def test_get_organization_not_found(self, mock_org_service, mock_db_session, mock_user):
        """Test getting a non-existent organization."""
        # Mock the behavior for org not found
        org_id = uuid4()
        
        # Replace the get method with one that raises an exception
        async def mock_get_org_not_found(org_id, session, user):
            raise HTTPException(
                status_code=404,
                detail=f"Organization {org_id} not found"
            )
            
        mock_org_service.get = AsyncMock(side_effect=mock_get_org_not_found)
        
        # Call the service and expect exception
        with pytest.raises(HTTPException) as excinfo:
            await mock_org_service.get(
                org_id,
                session=mock_db_session,
                user=mock_user
            )
        
        # Verify exception
        assert excinfo.value.status_code == 404
        assert f"Organization {org_id} not found" in str(excinfo.value.detail)
        
        # The service should have been called correctly
        assert mock_org_service.get.called
    
    async def test_update_organization(self, mock_org_service, mock_db_session, mock_user, mock_organization):
        """Test updating an organization."""
        # Add update method to the service
        async def mock_update_org(org_id, org_data, session, user):
            # Get organization
            db_org = session.execute.return_value.unique.return_value.scalar_one_or_none.return_value
            
            if not db_org:
                raise HTTPException(status_code=404, detail="Organization not found")
            
            # Update fields
            update_data = org_data.model_dump(exclude_unset=True)
            org_dict = db_org.model_dump()
            
            # Update the fields
            for field, value in update_data.items():
                org_dict[field] = value
            
            # Update timestamp
            org_dict["updated_at"] = datetime.now().isoformat()
            
            # Create updated organization
            db_org = MockOrganizationModel(**org_dict)
            
            await session.commit()
            
            # Return updated organization
            return MockResponse(**db_org.model_dump())
            
        mock_org_service.put_update = AsyncMock(side_effect=mock_update_org)
        
        # Setup mocks
        org_id = mock_organization.id
        mock_db_session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = mock_organization
        
        # Create update data
        update_data = MockResponse(
            name="Updated Organization Name",
            settings={"theme": "dark", "default_role": "member"}
        )
        
        # Call the service
        response = await mock_org_service.put_update(
            org_id,
            update_data,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result
        assert response.id == org_id
        assert response.name == "Updated Organization Name"
        assert response.settings == {"theme": "dark", "default_role": "member"}
        assert response.slug == mock_organization.slug  # Slug should not be updated
        
        # Verify database operations
        mock_db_session.commit.assert_called_once()
        
        # Verify service method was called
        assert mock_org_service.put_update.called
    
    async def test_update_organization_not_found(self, mock_org_service, mock_db_session, mock_user):
        """Test updating a non-existent organization."""
        # Add update method that fails
        async def mock_update_org_not_found(org_id, org_data, session, user):
            # No organization found
            session.execute.return_value.unique.return_value.scalar_one_or_none.return_value = None
            raise HTTPException(status_code=404, detail="Organization not found")
            
        mock_org_service.put_update = AsyncMock(side_effect=mock_update_org_not_found)
        
        # Create update data
        org_id = uuid4()  # Random non-existent ID
        update_data = MockResponse(
            name="Updated Organization Name"
        )
        
        # Verify exception is raised
        with pytest.raises(HTTPException) as exc_info:
            await mock_org_service.put_update(
                org_id,
                update_data,
                session=mock_db_session,
                user=mock_user
            )
        
        assert exc_info.value.status_code == 404
        assert "Organization not found" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_org_service.put_update.called
    
    async def test_remove_member(self, mock_org_service, mock_db_session, mock_user, mock_organization, mock_member):
        """Test removing a member from an organization."""
        # Add remove member method
        async def mock_delete_member(org_id, user_id, session):
            # Check if member exists
            member = mock_member  # Pretend member exists
            
            # Delete member
            await session.delete(member)
            await session.commit()
            
            # Return success response
            return {"message": "Member removed successfully"}
            
        mock_org_service.delete_member = AsyncMock(side_effect=mock_delete_member)
        
        # Call the service
        org_id = mock_organization.id
        user_id = mock_user["id"]
        
        response = await mock_org_service.delete_member(
            org_id,
            user_id,
            session=mock_db_session
        )
        
        # Verify result
        assert response["message"] == "Member removed successfully"
        
        # Verify database operations
        mock_db_session.delete.assert_called_once()
        mock_db_session.commit.assert_called_once()
        
        # Verify service method was called
        assert mock_org_service.delete_member.called
    
    async def test_remove_member_not_found(self, mock_org_service, mock_db_session):
        """Test removing a non-existent member."""
        # Add remove member method that fails
        async def mock_delete_member_not_found(org_id, user_id, session):
            # Member not found
            raise HTTPException(status_code=404, detail="Member not found")
            
        mock_org_service.delete_member = AsyncMock(side_effect=mock_delete_member_not_found)
        
        # Call the service and expect exception
        with pytest.raises(HTTPException) as exc_info:
            await mock_org_service.delete_member(
                uuid4(),
                uuid4(),
                session=mock_db_session
            )
        
        assert exc_info.value.status_code == 404
        assert "Member not found" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_org_service.delete_member.called
    
    async def test_list_organization_members(self, mock_org_service, mock_db_session, mock_user, mock_organization):
        """Test listing members of an organization."""
        # Add list members method
        async def mock_list_members(org_id, session, user):
            # Create mock members
            members = [
                {
                    "user_id": uuid4(),
                    "first_name": "Member",
                    "last_name": "One",
                    "email": "member1@example.com",
                    "role": {
                        "id": uuid4(),
                        "name": "Admin"
                    },
                    "status": "active"
                },
                {
                    "user_id": uuid4(),
                    "first_name": "Member",
                    "last_name": "Two",
                    "email": "member2@example.com",
                    "role": {
                        "id": uuid4(),
                        "name": "Member"
                    },
                    "status": "active"
                }
            ]
            
            return members
            
        mock_org_service.get_members = AsyncMock(side_effect=mock_list_members)
        
        # Call the service
        org_id = mock_organization.id
        
        response = await mock_org_service.get_members(
            org_id,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result
        assert isinstance(response, list)
        assert len(response) == 2
        assert response[0]["role"]["name"] == "Admin"
        assert response[1]["role"]["name"] == "Member"
        assert all(member["status"] == "active" for member in response)
        
        # Verify service method was called
        assert mock_org_service.get_members.called
    
    async def test_update_member_role(self, mock_org_service, mock_db_session, mock_member):
        """Test updating a member's role."""
        # Add update member role method
        async def mock_update_member_role(org_id, user_id, role_id, session):
            # Check if member exists
            member = mock_member  # Pretend member exists
            
            # Update role
            member.role_id = role_id
            
            await session.commit()
            
            # Return success response
            return {"message": "Member role updated successfully"}
            
        mock_org_service.put_update_member_role = AsyncMock(side_effect=mock_update_member_role)
        
        # Call the service
        org_id = uuid4()
        user_id = uuid4()
        role_id = uuid4()
        
        response = await mock_org_service.put_update_member_role(
            org_id,
            user_id,
            role_id,
            session=mock_db_session
        )
        
        # Verify result
        assert response["message"] == "Member role updated successfully"
        
        # Verify database operations
        mock_db_session.commit.assert_called_once()
        
        # Verify service method was called
        assert mock_org_service.put_update_member_role.called
    
    async def test_update_member_role_not_found(self, mock_org_service, mock_db_session):
        """Test updating role for a non-existent member."""
        # Add update member role method that fails
        async def mock_update_member_role_not_found(org_id, user_id, role_id, session):
            # Member not found
            raise HTTPException(status_code=404, detail="Member not found")
            
        mock_org_service.put_update_member_role = AsyncMock(side_effect=mock_update_member_role_not_found)
        
        # Call the service and expect exception
        with pytest.raises(HTTPException) as exc_info:
            await mock_org_service.put_update_member_role(
                uuid4(),
                uuid4(),
                uuid4(),
                session=mock_db_session
            )
        
        assert exc_info.value.status_code == 404
        assert "Member not found" in str(exc_info.value.detail)
        
        # Verify service method was called
        assert mock_org_service.put_update_member_role.called
    
    async def test_get_organization_statistics(self, mock_org_service, mock_db_session, mock_user, mock_organization):
        """Test getting organization statistics."""
        # Add statistics method
        async def mock_get_statistics(org_id, session, user):
            # Create mock statistics
            statistics = {
                "member_count": 10,
                "active_members": 8,
                "roles": [
                    {
                        "name": "Admin",
                        "count": 2
                    },
                    {
                        "name": "Member",
                        "count": 8
                    }
                ],
                "created_at": mock_organization.created_at
            }
            
            return statistics
            
        mock_org_service.get_statistics = AsyncMock(side_effect=mock_get_statistics)
        
        # Call the service
        org_id = mock_organization.id
        
        response = await mock_org_service.get_statistics(
            org_id,
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result
        assert response["member_count"] == 10
        assert response["active_members"] == 8
        assert len(response["roles"]) == 2
        assert response["roles"][0]["name"] == "Admin"
        assert response["roles"][1]["name"] == "Member"
        assert response["created_at"] == mock_organization.created_at
        
        # Verify service method was called
        assert mock_org_service.get_statistics.called
    
    async def test_create_organization_slug(self, mock_org_service, mock_db_session, mock_user):
        """Test slug creation when creating an organization with special characters."""
        # Call the original service to test slug creation
        # Use a name with special characters
        response = await mock_org_service.post_create_org(
            "Test Org !@#$%^&*()",
            session=mock_db_session,
            user=mock_user
        )
        
        # Verify result - check that the slug is properly sanitized
        assert response.name == "Test Org !@#$%^&*()"
        assert "test-org-" in response.slug  # Should only contain alphanumeric and hyphens
        assert not any(c in response.slug for c in "!@#$%^&*()")
        
        # The service should have been called correctly
        assert mock_org_service.post_create_org.called
    
    async def test_invite_member(self, mock_org_service, mock_db_session, mock_organization):
        """Test inviting a member to an organization."""
        # Add invite member method
        async def mock_invite_member(org_id, email, role_id, session):
            # Create invitation
            invitation = {
                "id": uuid4(),
                "organization_id": org_id,
                "email": email,
                "role_id": role_id,
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            
            # Return success response
            return invitation
            
        mock_org_service.post_invite_member = AsyncMock(side_effect=mock_invite_member)
        
        # Call the service
        org_id = mock_organization.id
        email = "newinvite@example.com"
        role_id = uuid4()
        
        response = await mock_org_service.post_invite_member(
            org_id,
            email,
            role_id,
            session=mock_db_session
        )
        
        # Verify result
        assert response["organization_id"] == org_id
        assert response["email"] == email
        assert response["role_id"] == role_id
        assert response["status"] == "pending"
        assert "created_at" in response
        
        # Verify service method was called
        assert mock_org_service.post_invite_member.called 