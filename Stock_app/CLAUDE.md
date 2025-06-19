# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Frontend Development (React + TypeScript + Vite)
```bash
cd fintech-terminal/frontend
npm install                  # Install dependencies
npm run dev                  # Start development server (port 3000)
npm run build                # Build for production
npm run lint                 # Run ESLint
npm run preview              # Preview production build
```

### Backend Development (FastAPI + Python)
```bash
cd fintech-terminal/backend
pip install -r requirements.txt    # Install Python dependencies
uvicorn main:app --reload          # Run development server (port 8000)
uvicorn main:app --reload --host 0.0.0.0 --port 8000  # Run with external access
python -m pytest tests/            # Run tests
python -m pytest tests/ --cov=app  # Run tests with coverage
```

### ML Engine Development
```bash
cd fintech-terminal/ml_engine
pip install -r requirements.txt    # Install ML dependencies
python src/predictors/price_predictor.py    # Run price predictions
jupyter notebook notebooks/        # Start Jupyter for experiments
```

### Docker Development
```bash
# Full stack with Docker Compose
docker-compose up -d               # Start all services in background
docker-compose up --build         # Rebuild and start services
docker-compose down               # Stop all services
docker-compose logs backend      # View backend logs
docker-compose exec backend bash # Shell into backend container
```

### Quick Start Options
```bash
# Windows Quick Start
./fintech-terminal/QUICK_START.bat        # One-click frontend start
./fintech-terminal/START_APP.bat          # Alternative launcher
./fintech-terminal/START_WITH_REAL_DATA.bat # Start with live data
```

## Architecture Overview

This is a comprehensive financial technology terminal (Bloomberg/TradingView alternative) with three main components:

### 1. Frontend (`fintech-terminal/frontend/`)
- **Framework**: React 18 + TypeScript + Vite
- **State Management**: Redux Toolkit
- **UI Components**: Tailwind CSS + Lucide React icons
- **Charts**: Chart.js, React-Chartjs-2, Recharts
- **Routing**: React Router DOM
- **Key Features**: Real-time market data, interactive charts, portfolio management

### 2. Backend (`fintech-terminal/backend/`)
- **Framework**: FastAPI with async/await support
- **Database**: PostgreSQL with SQLAlchemy ORM, Redis for caching
- **Authentication**: JWT-based with python-jose
- **APIs**: RESTful endpoints for market data, analytics, user management
- **External Integrations**: Yahoo Finance (yfinance), Alpha Vantage
- **Architecture**: Layered with API routes, services, models, schemas

### 3. ML Engine (`fintech-terminal/ml_engine/`)
- **Core Libraries**: TensorFlow, PyTorch, scikit-learn, XGBoost
- **Data Processing**: pandas, numpy, yfinance, pandas-ta
- **Analysis Types**: Price prediction, technical analysis, sentiment analysis
- **Backtesting**: Backtrader, QuantStats
- **Model Management**: MLflow for experiment tracking

## Key Development Patterns

### Backend Service Architecture
- API routes in `app/api/v1/` (versioned endpoints)
- Business logic in `app/services/` (market_service.py, user_service.py)
- Data models in `app/models/` with SQLAlchemy
- Pydantic schemas in `app/schemas/` for request/response validation
- Database layer in `app/db/` with repository pattern

### Frontend Component Structure
- Pages in `src/pages/` (Dashboard, MarketView, Portfolio, Analytics)
- Reusable components in `src/components/` organized by domain
- Custom hooks in `src/hooks/` (useMarketData.ts, useAuth.ts)
- API services in `src/services/` with axios
- Redux slices in `src/store/slices/` for state management

### Real-time Data Flow
- Backend fetches data from external APIs (Yahoo Finance, Alpha Vantage)
- Frontend polls backend REST endpoints for market data
- WebSocket connections planned for real-time streaming
- Redis caching layer for performance optimization

## Environment Configuration

### Required Environment Variables
```bash
# Backend (.env)
DATABASE_URL=postgresql://user:pass@localhost:5432/fintech_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-jwt-secret-key
CORS_ORIGINS=http://localhost:3000
API_KEY=your-external-api-key

# Frontend (.env)
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

## Testing Strategy

### Backend Testing
- Unit tests with pytest in `backend/tests/`
- Async test support with pytest-asyncio
- Test authentication in `tests/test_auth.py`

### Frontend Testing
- ESLint for code quality and consistency
- TypeScript for compile-time error checking

## Data Sources & External APIs

### Market Data Providers
- **Yahoo Finance**: Primary data source via yfinance library
- **Alpha Vantage**: Alternative API for market data
- **News APIs**: For sentiment analysis and market news

### Database Schema
- **PostgreSQL**: Primary database for persistent data
- **Redis**: Caching layer for frequently accessed data
- **Time-series data**: Optimized for financial market data storage

## Deployment Architecture

### Docker Compose Services
- `postgres`: PostgreSQL database (port 5432)
- `redis`: Redis cache (port 6379)  
- `backend`: FastAPI application (port 8000)
- `frontend`: React development server (port 3000)
- `ml_engine`: Machine learning services
- `nginx`: Reverse proxy (optional, production profile)

### Development vs Production
- Development: Direct service access with hot reloading
- Production: Nginx reverse proxy, optimized builds, health checks

## Security Considerations

- JWT-based authentication with secure secret keys
- CORS configuration for cross-origin requests
- Input validation with Pydantic schemas
- Environment variable management for sensitive data
- Database connection security with proper credentials

## Performance Optimizations

- Redis caching for frequently accessed market data  
- Async/await patterns in FastAPI for concurrent request handling
- React lazy loading for code splitting
- PostgreSQL indexing for time-series data queries
- Docker multi-stage builds for optimized container sizes