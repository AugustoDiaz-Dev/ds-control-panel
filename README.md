# DS Control Panel

An interactive Data Science control panel for real-time experimentation with a FastAPI backend. This project focuses on a financial classification problem and compares multiple ML algorithms.

## Features

- **FastAPI Backend**: Production-ready REST API
- **Multiple ML Models**: LightGBM, XGBoost, Random Forest, Extra Trees, CatBoost
- **Real-time Training**: Train models with custom hyperparameters
- **Model Comparison**: Compare performance across different algorithms
- **Prediction API**: Make predictions using trained models
- **MLflow Integration**: Experiment tracking, model versioning, and performance comparison
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **SHAP Values**: Model interpretability with SHAP (SHapley Additive exPlanations) values
- **Model Ensembles**: Voting, Stacking, and Weighted ensemble methods
- **Basic Monitoring**: Structured logging, request/response metrics, health checks, and error tracking
- **Docker Support**: Containerized deployment with Docker Compose
- **Interactive Dashboard**: Real-time visualizations (ROC curves, confusion matrices, feature importance)
- **Comprehensive Testing**: Unit, integration, and model validation tests
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions

## Project Structure

```
v-3/
├── app/
│   ├── controllers/      # API endpoints (MVC Controllers)
│   │   └── api.py
│   ├── models/           # ML models and data processing (MVC Models)
│   │   └── ml_models/
│   ├── views/            # Frontend templates (MVC Views)
│   │   ├── index.html
│   │   └── dashboard.html
│   ├── config/           # Configuration files
│   ├── monitoring/       # Monitoring and logging
│   │   ├── logger.py     # Structured logging configuration
│   │   ├── metrics.py     # Metrics collection
│   │   └── middleware.py  # Request/response logging middleware
│   └── utils/            # Utility functions
├── tests/                # Test suite
│   ├── unit/
│   ├── integration/
│   └── model_validation/
├── data/                 # Dataset storage
├── models/               # Saved model files
├── .github/workflows/    # CI/CD pipelines
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick Start

### Option 1: Docker (Recommended)

```bash
docker-compose up -d
```

This starts:
- **Backend API** at `http://localhost:8000`
- **MLflow UI** at `http://localhost:5001`
- **Jupyter Notebook** at `http://localhost:8888`

### Option 2: Local Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start the backend**:
```bash
python -m app.main
# or
uvicorn app.controllers.api:app --reload --host 0.0.0.0 --port 8000
```

3. **Start MLflow UI** (optional):
```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5001
```

4. **Open the dashboard**:
```bash
python -m http.server 8080
# Then open http://localhost:8080/app/views/index.html
```

## API Endpoints

### Authentication Endpoints
- `POST /auth/login` - Login to get access token (form data: username, password)
- `POST /auth/register` - Register a new user (JSON body: username, password, email, full_name)
- `GET /auth/me` - Get current user information (requires authentication)
- `GET /auth/users` - List all users (admin only)
- `PUT /auth/users/{user_id}` - Update a user (admin only)
- `DELETE /auth/users/{user_id}` - Delete a user (admin only)

**Note:** All endpoints except `/`, `/auth/login`, `/auth/register`, `/dashboard`, and `/index` require authentication.

### Core Endpoints (Require Authentication)
- `GET /` - API information
- `POST /train` - Train all models
- `POST /optimize` - Optimize hyperparameters
- `GET /metrics` - Get model metrics
- `GET /models` - List trained models
- `POST /predict` - Make predictions
- `GET /visualizations/*` - Get visualization data

### SHAP Interpretability
- `GET /shap/values/{model_name}` - Calculate SHAP values for a model (with optional `sample_size` parameter)
- `GET /shap/explain/{model_name}` - Explain a single prediction using SHAP values (requires `features` query parameter)

### Model Ensembles
- `POST /ensemble/train` - Train an ensemble model (voting, stacking, or weighted)
- `GET /ensemble/list` - List all trained ensemble models
- `POST /ensemble/predict` - Make predictions using an ensemble model

### MLflow Integration
- `GET /mlflow/experiments` - List MLflow experiments
- `GET /mlflow/runs` - List MLflow runs

### Monitoring Endpoints
- `GET /health` - Health check endpoint (no authentication required)
- `GET /model/status` - Get status of all trained models (requires authentication)
- `GET /monitoring/metrics` - Get API performance metrics and statistics (requires authentication)
- `GET /monitoring/errors` - Get recent errors from the API (requires authentication)

See API documentation at `http://localhost:8000/docs`

### Authentication Usage

1. **Login to get access token:**
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

2. **Use token in API requests:**
```bash
curl -X GET "http://localhost:8000/models" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

3. **Register a new user:**
```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "newuser",
    "email": "user@example.com",
    "password": "securepassword",
    "full_name": "New User"
  }'
```

**Default Admin User:**
- Username: `admin`
- Password: `admin123`
- ⚠️ **Change this password in production!**

### Example: Training an Ensemble

```bash
# First, login to get token
TOKEN=$(curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123" | jq -r '.access_token')

# Then use the token for authenticated requests
curl -X POST "http://localhost:8000/ensemble/train" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "ensemble_type": "voting",
    "model_names": ["random_forest", "xgboost", "lightgbm"]
  }'
```

### Example: Getting SHAP Values

```bash
curl "http://localhost:8000/shap/values/random_forest?sample_size=100" \
  -H "Authorization: Bearer $TOKEN"
```

### Example: Explaining a Prediction

```bash
curl "http://localhost:8000/shap/explain/random_forest?features=750,80000,15000,35,60000,8,25000" \
  -H "Authorization: Bearer $TOKEN"
```

### Example: Monitoring

```bash
# Health check (no auth required)
curl "http://localhost:8000/health"

# Get model status
curl "http://localhost:8000/model/status" \
  -H "Authorization: Bearer $TOKEN"

# Get monitoring metrics
curl "http://localhost:8000/monitoring/metrics" \
  -H "Authorization: Bearer $TOKEN"

# Get recent errors
curl "http://localhost:8000/monitoring/errors?limit=10" \
  -H "Authorization: Bearer $TOKEN"
```

## Monitoring

The application includes comprehensive monitoring capabilities:

### Logging
- **Structured Logging**: All logs are written to `logs/` directory
  - `logs/app.log` - Application logs (all levels)
  - `logs/errors.log` - Error logs only
  - `logs/access.log` - HTTP access logs
- **Log Rotation**: Logs are automatically rotated when they reach 10MB (keeps 5 backups)
- **Log Levels**: Configurable via `setup_logging()` function

### Metrics Collection
The system automatically tracks:
- **Request Metrics**: Total requests, requests per second, response times (avg, p50, p95, p99)
- **Error Tracking**: Error rates, recent errors with details
- **Model Metrics**: Training times, prediction times, usage counts per model
- **Endpoint Statistics**: Request counts and error rates per endpoint

### Health Monitoring
- **Health Check**: `/health` endpoint provides system health status
- **Model Status**: `/model/status` shows which models are loaded and available
- **Performance Metrics**: `/monitoring/metrics` provides detailed performance statistics

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=html
```

## CI/CD

The project includes GitHub Actions workflows for automated testing and deployment. See `.github/workflows/` for details.

## License

MIT License
