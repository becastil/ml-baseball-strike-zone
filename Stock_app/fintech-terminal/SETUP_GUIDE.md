# Fintech Terminal - Complete Setup Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- PostgreSQL 14+
- Redis 6+
- Docker & Docker Compose (optional but recommended)

### Option 1: Docker Setup (Recommended)

```bash
# Clone the repository
cd fintech-terminal

# Create environment file
cp .env.example .env

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Option 2: Manual Setup

#### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup database
createdb fintech_terminal

# Run migrations
alembic init alembic
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head

# Copy environment file
cp .env.example .env

# Edit .env with your settings
# - Add your database URL
# - Add API keys for market data providers
# - Set JWT secret key

# Run the backend
python run.py
```

#### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Run the frontend
npm run dev
```

#### 3. ML Engine Setup

```bash
cd ml_engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (optional)
python scripts/download_models.py
```

## 📋 Configuration

### Environment Variables

#### Backend (.env)
```env
# Database
DATABASE_URL=postgresql://user:password@localhost/fintech_terminal
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Market Data Providers
YAHOO_FINANCE_API_KEY=your-key
ALPHA_VANTAGE_API_KEY=your-key
POLYGON_API_KEY=your-key

# Email (for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Feature Flags
ENABLE_ML_PREDICTIONS=true
ENABLE_REALTIME_DATA=true
```

#### Frontend (.env)
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_ENABLE_MOCK_DATA=false
```

### Market Data Providers

1. **Yahoo Finance** (Free)
   - No API key required for basic usage
   - Rate limited

2. **Alpha Vantage** (Free tier available)
   - Sign up at: https://www.alphavantage.co/support/#api-key
   - Free tier: 5 API requests per minute

3. **Polygon.io** (Free tier available)
   - Sign up at: https://polygon.io/
   - Free tier: 5 API calls per minute

## 🏗️ Architecture Overview

### Tech Stack

#### Backend
- **Framework**: FastAPI (async Python web framework)
- **Database**: PostgreSQL (primary), Redis (caching)
- **ORM**: SQLAlchemy with async support
- **Authentication**: JWT tokens
- **WebSocket**: For real-time data
- **Task Queue**: Celery with Redis broker

#### Frontend
- **Framework**: React 18 with TypeScript
- **State Management**: Redux Toolkit
- **Styling**: TailwindCSS
- **Charts**: Chart.js, Recharts, Lightweight Charts
- **Build Tool**: Vite
- **HTTP Client**: Axios

#### ML/AI Engine
- **Deep Learning**: TensorFlow, PyTorch
- **ML Libraries**: scikit-learn, XGBoost
- **Time Series**: Prophet, statsmodels
- **NLP**: Transformers, NLTK
- **Data Processing**: pandas, numpy
- **Technical Analysis**: ta-lib, pandas-ta

## 🛠️ Development Workflow

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test

# ML tests
cd ml_engine
pytest tests/
```

### Code Quality

```bash
# Backend linting
cd backend
flake8 app/
black app/

# Frontend linting
cd frontend
npm run lint
npm run format
```

### Making Changes

1. Create a feature branch
2. Make your changes
3. Run tests
4. Update documentation
5. Submit a pull request

## 📊 Features Overview

### Current Features
- ✅ Real-time market data streaming
- ✅ Interactive financial charts
- ✅ Technical indicators (50+ indicators)
- ✅ Portfolio management
- ✅ Watchlist functionality
- ✅ Price alerts
- ✅ Basic ML predictions
- ✅ User authentication
- ✅ Dark mode

### Upcoming Features
- 🔄 Advanced ML models
- 🔄 Backtesting engine
- 🔄 Social sentiment analysis
- 🔄 Options chain analysis
- 🔄 Custom indicators
- 🔄 Mobile app

## 🔧 Common Issues & Solutions

### Issue: Database connection error
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Create database if missing
createdb fintech_terminal

# Check connection string in .env
DATABASE_URL=postgresql://user:password@localhost/fintech_terminal
```

### Issue: Redis connection error
```bash
# Start Redis
redis-server

# Or with Docker
docker run -d -p 6379:6379 redis
```

### Issue: Missing market data
- Check API keys in .env
- Verify rate limits haven't been exceeded
- Try using mock data: `VITE_ENABLE_MOCK_DATA=true`

### Issue: CORS errors
- Ensure backend is running on http://localhost:8000
- Check CORS settings in backend/app/main.py

## 📚 API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

#### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login
- `POST /api/v1/auth/refresh` - Refresh token

#### Market Data
- `GET /api/v1/market/quote/{symbol}` - Get stock quote
- `GET /api/v1/market/history/{symbol}` - Historical data
- `WS /ws/market` - Real-time market data stream

#### Portfolio
- `GET /api/v1/portfolio` - Get user portfolios
- `POST /api/v1/portfolio` - Create portfolio
- `POST /api/v1/portfolio/{id}/transaction` - Add transaction

## 🚢 Deployment

### Using Docker Compose (Production)

```bash
# Build and start services
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
```

### Manual Deployment

1. **Backend**:
   - Use Gunicorn with Uvicorn workers
   - Set up Nginx as reverse proxy
   - Use systemd for process management

2. **Frontend**:
   - Build: `npm run build`
   - Serve with Nginx or CDN

3. **Database**:
   - Use managed PostgreSQL (AWS RDS, Google Cloud SQL)
   - Set up regular backups

## 📞 Support

- **Documentation**: See `/docs` folder
- **Issues**: GitHub Issues
- **Community**: Discord/Slack (coming soon)

## 📄 License

MIT License - feel free to use for personal or commercial projects.

---

Happy Trading! 📈🚀