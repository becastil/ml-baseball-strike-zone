"""
User Pydantic schemas
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from uuid import UUID


class UserBase(BaseModel):
    """
    Base user schema
    """
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    country: Optional[str] = None
    timezone: str = "UTC"
    language: str = "en"
    theme: str = "light"
    notifications_enabled: bool = True
    email_notifications: bool = True


class UserCreate(UserBase):
    """
    Schema for creating a new user
    """
    password: str = Field(..., min_length=8, max_length=100)
    confirm_password: str
    
    def validate_passwords_match(self) -> "UserCreate":
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class UserUpdate(BaseModel):
    """
    Schema for updating user information
    """
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    country: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    theme: Optional[str] = None
    notifications_enabled: Optional[bool] = None
    email_notifications: Optional[bool] = None


class UserResponse(UserBase):
    """
    Schema for user response
    """
    id: UUID
    is_active: bool
    is_verified: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    subscription_tier: str
    subscription_expires: Optional[datetime] = None
    two_factor_enabled: bool
    
    model_config = ConfigDict(from_attributes=True)


class UserLogin(BaseModel):
    """
    Schema for user login
    """
    username: str  # Can be email or username
    password: str


class Token(BaseModel):
    """
    Schema for authentication token response
    """
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    """
    Schema for token payload
    """
    sub: str
    exp: int
    user_id: Optional[str] = None
    type: Optional[str] = None


class PasswordChange(BaseModel):
    """
    Schema for changing password
    """
    old_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
    confirm_new_password: str
    
    def validate_passwords_match(self) -> "PasswordChange":
        if self.new_password != self.confirm_new_password:
            raise ValueError("New passwords do not match")
        return self


class PasswordReset(BaseModel):
    """
    Schema for password reset
    """
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)
    confirm_password: str
    
    def validate_passwords_match(self) -> "PasswordReset":
        if self.new_password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class EmailVerification(BaseModel):
    """
    Schema for email verification
    """
    email: EmailStr
    token: str