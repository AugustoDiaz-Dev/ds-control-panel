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

- `GET /` - API information
- `POST /train` - Train all models
- `POST /optimize` - Optimize hyperparameters
- `GET /metrics` - Get model metrics
- `GET /models` - List trained models
- `POST /predict` - Make predictions
- `GET /visualizations/*` - Get visualization data

See API documentation at `http://localhost:8000/docs`

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
