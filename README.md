# Delivery Time Prediction System

A comprehensive machine learning system for predicting delivery times using data cleaning, feature engineering, and advanced ML models with a production-ready API.

## Overview

This project implements an end-to-end solution for delivery time prediction, featuring:
- Data processing and cleaning pipeline for customer, product, and order data
- Machine learning model training and evaluation
- RESTful API for model deployment and management
- Docker containerization for easy deployment

## Project Structure

```
.
├── app.py                      # FastAPI application for model serving
├── solution.py                 # Data cleaning and processing pipeline
├── solution.ipynb             # Jupyter notebook with ML model development
├── export_clean_model.py      # Script to export trained ML models
├── ml_evaluator.py            # Model evaluation utilities
├── test_revenue.py            # Revenue calculation tests
├── docker-compose.yml         # Docker compose configuration
├── Dockerfile                 # Docker image definition
├── pyproject.toml            # Python project dependencies
├── data/                     # Data directory
│   ├── customers.csv         # Raw customer data
│   ├── products.csv          # Raw product data
│   ├── orders.csv            # Raw order data
│   ├── dim_customers_clean.csv  # Cleaned customer dimension
│   ├── dim_products_clean.csv   # Cleaned product dimension
│   ├── fact_orders_clean.csv    # Cleaned orders fact table
│   ├── kpi_revenue_by_month.csv # Revenue KPI by month
│   └── ml_dataset.csv           # Prepared ML training dataset
├── models/                    # Trained model storage
│   └── *.joblib              # Serialized ML models
└── predictions.csv           # Model predictions output
```

## Features

### Data Processing Pipeline
- **Column Standardization**: Normalizes column names across all datasets
- **Data Cleaning**: Handles missing values, standardizes formats, and validates data types
- **Deduplication**: Removes duplicate records based on business logic
- **Data Integrity Checks**: Validates referential integrity between datasets
- **Feature Engineering**: Creates calculated fields and enriches data with timezone conversions
- **KPI Calculation**: Generates business metrics including revenue by month, category, and top customers

### Machine Learning Components
- **Feature Preparation**: Extracts temporal features, normalizes numerical values
- **Model Training**: Implements Random Forest regressor with hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics including MAE, RMSE, R², and MAPE
- **Feature Importance Analysis**: Identifies key predictors for delivery time

### API Endpoints

#### Health Check
- `GET /health/`: Service health status

#### Model Management
- `POST /model/`: Upload and activate a new model
- `GET /model/`: List all available models
- `PUT /model/{name}/activate`: Activate a specific model
- `DELETE /model/{name}`: Delete a model

#### Prediction
- `POST /predict/`: Get delivery time prediction

## Installation

### Prerequisites
- Python 3.11+
- Docker and Docker Compose (for containerized deployment)

### Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-engineer-test
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Process data and train model:
```bash
# Clean and process raw data
python solution.py

# Train ML model (run Jupyter notebook or script)
jupyter notebook solution.ipynb

# Export trained model
python export_clean_model.py
```

### Docker Deployment

1. Build and run the container:
```bash
docker-compose up --build
```

2. The API will be available at `http://localhost:8000`

## API Usage

### Upload a Model
```bash
curl -X POST "http://localhost:8000/model/" \
  -H "Content-Type: multipart/form-data" \
  -F "name=delivery_model_v1" \
  -F "file=@model.joblib" \
  -F "activate=true"
```

### Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "qty": 2,
    "weight_kg": 0.8,
    "unit_price_usd": 12.5,
    "distance_km": 350.0,
    "hour_of_day": 14,
    "weekday": 2,
    "ship_country": "IT",
    "category": "Widgets",
    "carrier": "DHL"
  }'
```

### List Available Models
```bash
curl -X GET "http://localhost:8000/model/"
```

## Model Performance

The trained Random Forest model achieves:
- **MAE**: ~1.2 days
- **RMSE**: ~1.6 days
- **R² Score**: ~0.85
- **MAPE**: ~15%

Key predictive features:
1. Distance (km)
2. Carrier type
3. Weight (kg)
4. Hour of day
5. Weekday

## Data Schema

### Input Features for Prediction
- `qty`: Quantity of items (integer)
- `weight_kg`: Total weight in kilograms (float)
- `unit_price_usd`: Unit price in USD (float)
- `distance_km`: Shipping distance in kilometers (float)
- `hour_of_day`: Hour when order was placed (0-23)
- `weekday`: Day of week (0=Monday, 6=Sunday)
- `ship_country`: Destination country code
- `category`: Product category
- `carrier`: Shipping carrier name

### Output
- `delivery_days`: Predicted delivery time in days (float)

```

## Development

### Adding New Features
1. Update the feature engineering in `solution.ipynb`
2. Retrain the model with new features
3. Export the updated model using `export_clean_model.py`
4. Update the API request schema in `app.py` if needed

### Model Versioning
Models are stored with versioned names (e.g., `model_v1.joblib`, `model_v2.joblib`) to maintain backward compatibility and enable A/B testing.

## Architecture

The system follows a modular architecture:
1. **Data Layer**: Raw CSV files processed through cleaning pipeline
2. **ML Layer**: Feature engineering and model training in Jupyter environment
3. **API Layer**: FastAPI service for model serving
4. **Storage Layer**: File-based model storage with activation management

## Environment Variables

- `PYTHONUNBUFFERED`: Set to 1 for real-time logging in Docker