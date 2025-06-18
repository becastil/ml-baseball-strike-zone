# Fintech Terminal Backend

A professional FastAPI backend for a financial terminal application.

## Features

- **Authentication**: JWT-based authentication with refresh tokens
- **Market Data**: Real-time stock quotes, historical data, and company information
- **User Management**: User registration, login, and profile management
- **Security**: Password hashing, CORS configuration, and secure token handling
- **Database**: Async PostgreSQL with SQLAlchemy ORM
- **API Documentation**: Auto-generated Swagger/OpenAPI documentation

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── auth.py         # Authentication endpoints
│   │       └── market_data.py  # Market data endpoints
│   ├── core/
│   │   ├── config.py          # Application settings
│   │   └── security.py        # Security utilities
│   ├── db/
│   │   └── database.py        # Database configuration
│   ├── models/
│   │   └── user.py           # SQLAlchemy models
│   ├── schemas/
│   │   ├── user.py           # User schemas
│   │   └── market.py         # Market data schemas
│   ├── services/
│   │   ├── user_service.py   # User business logic
│   │   └── market_service.py # Market data fetching
│   └── main.py               # FastAPI application
├── tests/
│   └── test_auth.py          # Authentication tests
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables example
└── README.md               # This file
```

## Setup

1. **Clone the repository**
   ```bash
   cd fintech-terminal/backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Set up PostgreSQL database**
   - Create a PostgreSQL database
   - Update DATABASE_URL in .env

6. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

## API Documentation

Once the application is running, you can access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login user
- `POST /api/v1/auth/refresh` - Refresh access token
- `GET /api/v1/auth/me` - Get current user info
- `POST /api/v1/auth/change-password` - Change password
- `POST /api/v1/auth/logout` - Logout user

### Market Data
- `GET /api/v1/market/quote/{symbol}` - Get stock quote
- `GET /api/v1/market/quotes` - Get multiple quotes
- `GET /api/v1/market/historical/{symbol}` - Get historical data
- `GET /api/v1/market/search` - Search symbols
- `GET /api/v1/market/company/{symbol}` - Get company info
- `GET /api/v1/market/market-overview` - Get market overview
- `GET /api/v1/market/watchlist` - Get user watchlist
- `POST /api/v1/market/watchlist/{symbol}` - Add to watchlist
- `DELETE /api/v1/market/watchlist/{symbol}` - Remove from watchlist

## Testing

Run tests with pytest:
```bash
pytest
```

## Development

### Database Migrations

Using Alembic for database migrations:
```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head
```

### Code Style

This project follows PEP 8 style guidelines. Format code with:
```bash
black app/
isort app/
```

## Deployment

For production deployment:
1. Use environment variables for sensitive data
2. Set up proper database with connection pooling
3. Use a process manager like gunicorn
4. Set up reverse proxy with nginx
5. Enable HTTPS with SSL certificates
6. Configure proper CORS origins

## License

MIT License