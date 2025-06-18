"""
User service for database operations
"""
from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update
from datetime import datetime

from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from app.core.security import get_password_hash, verify_password


class UserService:
    """
    Service class for user-related database operations
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_user(self, user_data: UserCreate) -> User:
        """
        Create a new user
        """
        # Hash the password
        hashed_password = get_password_hash(user_data.password)
        
        # Create user instance
        user = User(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            phone_number=user_data.phone_number,
            country=user_data.country,
            timezone=user_data.timezone,
            language=user_data.language,
            theme=user_data.theme,
            notifications_enabled=user_data.notifications_enabled,
            email_notifications=user_data.email_notifications
        )
        
        # Add to database
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        
        return user
    
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """
        Get user by ID
        """
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email
        """
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username
        """
        result = await self.db.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()
    
    async def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user by username/email and password
        """
        # Try to get user by email first, then username
        user = await self.get_by_email(username)
        if not user:
            user = await self.get_by_username(username)
        
        if not user:
            return None
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            return None
        
        # Verify password
        if not verify_password(password, user.hashed_password):
            # Increment failed login attempts
            user.failed_login_attempts += 1
            
            # Lock account after 5 failed attempts
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.utcnow().replace(
                    minute=datetime.utcnow().minute + 30
                )
            
            await self.db.commit()
            return None
        
        # Reset failed login attempts on successful login
        user.failed_login_attempts = 0
        user.last_login = datetime.utcnow()
        await self.db.commit()
        
        return user
    
    async def update_user(self, user_id: UUID, user_data: UserUpdate) -> Optional[User]:
        """
        Update user information
        """
        # Get user
        user = await self.get_by_id(user_id)
        if not user:
            return None
        
        # Update fields
        update_data = user_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(user)
        
        return user
    
    async def update_password(self, user_id: UUID, new_password: str) -> bool:
        """
        Update user password
        """
        hashed_password = get_password_hash(new_password)
        
        result = await self.db.execute(
            update(User)
            .where(User.id == user_id)
            .values(
                hashed_password=hashed_password,
                updated_at=datetime.utcnow()
            )
        )
        
        await self.db.commit()
        return result.rowcount > 0
    
    async def verify_email(self, user_id: UUID) -> bool:
        """
        Mark user email as verified
        """
        result = await self.db.execute(
            update(User)
            .where(User.id == user_id)
            .values(
                is_verified=True,
                updated_at=datetime.utcnow()
            )
        )
        
        await self.db.commit()
        return result.rowcount > 0
    
    async def update_subscription(
        self, 
        user_id: UUID, 
        tier: str, 
        expires_at: Optional[datetime] = None
    ) -> bool:
        """
        Update user subscription
        """
        result = await self.db.execute(
            update(User)
            .where(User.id == user_id)
            .values(
                subscription_tier=tier,
                subscription_expires=expires_at,
                updated_at=datetime.utcnow()
            )
        )
        
        await self.db.commit()
        return result.rowcount > 0
    
    async def enable_two_factor(self, user_id: UUID, secret: str) -> bool:
        """
        Enable two-factor authentication
        """
        result = await self.db.execute(
            update(User)
            .where(User.id == user_id)
            .values(
                two_factor_enabled=True,
                two_factor_secret=secret,
                updated_at=datetime.utcnow()
            )
        )
        
        await self.db.commit()
        return result.rowcount > 0
    
    async def disable_two_factor(self, user_id: UUID) -> bool:
        """
        Disable two-factor authentication
        """
        result = await self.db.execute(
            update(User)
            .where(User.id == user_id)
            .values(
                two_factor_enabled=False,
                two_factor_secret=None,
                updated_at=datetime.utcnow()
            )
        )
        
        await self.db.commit()
        return result.rowcount > 0
    
    async def delete_user(self, user_id: UUID) -> bool:
        """
        Delete a user (soft delete by deactivating)
        """
        result = await self.db.execute(
            update(User)
            .where(User.id == user_id)
            .values(
                is_active=False,
                updated_at=datetime.utcnow()
            )
        )
        
        await self.db.commit()
        return result.rowcount > 0