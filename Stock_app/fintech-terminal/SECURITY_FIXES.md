# Critical Security Fixes Needed

## Immediate Actions Required

### 1. Fix Secret Key Management
```python
# backend/app/core/config.py
# BEFORE (INSECURE):
SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")

# AFTER (SECURE):
SECRET_KEY: str = os.getenv("SECRET_KEY")  # No default value
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is required")
```

### 2. Implement Proper Authentication
```python
# backend/app/core/security.py
async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # CRITICAL: Actually fetch user from database
    user = await get_user_from_db(username)  # Implement this
    if user is None:
        raise credentials_exception
    return user
```

### 3. Add Input Validation
```python
# backend/app/schemas/market.py
from pydantic import BaseModel, validator
import re

class SymbolRequest(BaseModel):
    symbols: str
    
    @validator('symbols')
    def validate_symbols(cls, v):
        # Only allow alphanumeric, commas, and dots
        if not re.match(r'^[A-Za-z0-9,.-]+$', v):
            raise ValueError('Invalid symbol format')
        
        # Limit length
        if len(v) > 200:
            raise ValueError('Symbols string too long')
            
        return v.upper()
```

### 4. Environment Variables
```bash
# .env.example
SECRET_KEY=your-super-secret-key-here-min-32-chars
DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379/0
CORS_ORIGINS=https://yourapp.com,http://localhost:3000

# Frontend .env.example  
REACT_APP_API_URL=https://api.yourapp.com
REACT_APP_WS_URL=wss://api.yourapp.com
```

### 5. Replace Print Statements with Logging
```python
# backend/app/core/logging.py
import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# In services, replace print() with:
logger = logging.getLogger(__name__)
logger.error(f"Error fetching quote for {symbol}: {e}")
```

## Testing Requirements

### Minimum Test Coverage Needed
```bash
# Backend (Python)
pytest backend/tests/ --cov=app --cov-report=html --cov-fail-under=70

# Frontend (TypeScript)
npm test -- --coverage --coverageThreshold.global.lines=70
```

### Critical Test Cases
1. Authentication flow tests
2. Input validation tests  
3. WebSocket connection tests
4. Market data service tests
5. Error handling tests

## Security Checklist Before Deployment

- [ ] All secrets moved to environment variables
- [ ] Input validation on all endpoints
- [ ] Proper error handling with logging
- [ ] HTTPS/WSS in production
- [ ] Database connection pooling configured
- [ ] Rate limiting implemented
- [ ] CORS properly configured
- [ ] Dependencies updated and audited
- [ ] Test coverage >70%
- [ ] Security headers configured