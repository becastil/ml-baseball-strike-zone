"""
Tests for authentication endpoints
"""
import pytest
from httpx import AsyncClient
from fastapi import status

from app.main import app
from app.core.security import get_password_hash


@pytest.mark.asyncio
async def test_register_user():
    """Test user registration"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "username": "testuser",
                "password": "testpassword123",
                "confirm_password": "testpassword123",
                "full_name": "Test User"
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"
        assert "id" in data
        assert "hashed_password" not in data


@pytest.mark.asyncio
async def test_login():
    """Test user login"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # First register a user
        await client.post(
            "/api/v1/auth/register",
            json={
                "email": "login@example.com",
                "username": "loginuser",
                "password": "loginpassword123",
                "confirm_password": "loginpassword123"
            }
        )
        
        # Then try to login
        response = await client.post(
            "/api/v1/auth/login",
            data={
                "username": "login@example.com",
                "password": "loginpassword123"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_invalid_credentials():
    """Test login with invalid credentials"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/login",
            data={
                "username": "nonexistent@example.com",
                "password": "wrongpassword"
            }
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_get_current_user():
    """Test getting current user info"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Register and login
        await client.post(
            "/api/v1/auth/register",
            json={
                "email": "current@example.com",
                "username": "currentuser",
                "password": "currentpassword123",
                "confirm_password": "currentpassword123"
            }
        )
        
        login_response = await client.post(
            "/api/v1/auth/login",
            data={
                "username": "current@example.com",
                "password": "currentpassword123"
            }
        )
        
        token = login_response.json()["access_token"]
        
        # Get current user info
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == "current@example.com"
        assert data["username"] == "currentuser"