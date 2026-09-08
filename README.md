# CI/CD Pipeline for Flask & Machine Learning Applications

## Overview

This project demonstrates the implementation of automated CI/CD workflows for Python-based applications using **GitHub Actions, Docker, Flask, Pytest, and Scikit-learn**.

The project contains two application workflows:

1. A basic Flask web application
2. A Flask application that serves predictions from a trained Random Forest machine learning model

The objective is to demonstrate how automated testing and containerization can be integrated into the development workflow.

## Architecture

```text
                    GitHub Repository
                           │
                           ▼
                    GitHub Actions
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
          Flask Application      ML Flask Application
                 │                   │
                 │             Random Forest Model
                 │                   │
                 └─────────┬─────────┘
                           ▼
                     Automated Tests
                         Pytest
                           │
                           ▼
                       Docker Build
                           │
                           ▼
                       Docker Image
```

## Tech Stack

| Category | Technology |
|---|---|
| Programming Language | Python 3.10 |
| Web Framework | Flask |
| Machine Learning | Scikit-learn |
| ML Model | Random Forest |
| Testing | Pytest |
| Containerization | Docker |
| Local Containers | Docker Compose |
| CI/CD | GitHub Actions |
| Version Control | Git & GitHub |
| Model Serialization | Joblib |

## Project Features

- Flask web application
- Machine learning prediction application
- Random Forest model training
- Automated unit testing
- Docker containerization
- Docker Compose support
- GitHub Actions CI/CD workflow
- Automated testing with Pytest
- Docker image build and deployment workflow

## Project Structure

```text
ci-cd-aids/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── app.py
├── train_model.py
├── random_forest_model.joblib
├── data.csv
│
├── test_app.py
├── test_model.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Machine Learning Workflow

The machine learning component follows this workflow:

```text
Dataset
  │
  ▼
data.csv
  │
  ▼
train_model.py
  │
  ▼
Random Forest Model
  │
  ▼
random_forest_model.joblib
  │
  ▼
Flask Application
  │
  ▼
Prediction API
```

The trained Random Forest model is serialized using Joblib and loaded by the Flask application for prediction.

## Testing

The project contains automated tests using **Pytest**.

### Application Tests

`test_app.py` is used to test the Flask application's routes and behavior.

### Machine Learning Tests

`test_model.py` verifies that the trained model can be loaded and used to generate predictions.

Example:

```python
from joblib import load
import numpy as np

def test_model_prediction():
    model = load("random_forest_model.joblib")
    sample_input = np.array([[1, 2, 3, 4]])
    prediction = model.predict(sample_input)

    assert prediction is not None
```

## CI/CD Pipeline

The CI/CD workflow is implemented using **GitHub Actions**.

The workflow performs automated steps such as:

```text
Git Push
   │
   ▼
Checkout Repository
   │
   ▼
Set Up Python
   │
   ▼
Install Dependencies
   │
   ▼
Run Pytest
   │
   ▼
Build Docker Image
   │
   ▼
Docker Hub Authentication
   │
   ▼
Push Docker Image
```

The workflow configuration is located at:

```text
.github/workflows/ci-cd.yml
```

## Docker

The application can be containerized using the included `Dockerfile`.

### Build the Docker Image

```bash
docker build -t ci-cd-app .
```

### Run the Container

```bash
docker run -p 5000:5000 ci-cd-app
```

The application can then be accessed through:

```text
http://localhost:5000
```

## Docker Compose

The project also includes a Docker Compose configuration.

Start the application using:

```bash
docker compose up --build
```

Stop the application using:

```bash
docker compose down
```

## Key Learning Outcomes

Through this project, I gained practical experience in:

- Building Flask applications
- Training and serving machine learning models
- Writing automated tests
- Using Pytest for application and model testing
- Containerizing Python applications with Docker
- Managing containers with Docker Compose
- Creating CI/CD workflows with GitHub Actions
- Automating testing and Docker image builds
- Integrating machine learning workflows with DevOps practices

## Challenges Faced

During development, I worked through issues involving:

- Git branch configuration
- Docker authentication
- Docker image push configuration
- CI/CD workflow configuration
- Dockerfile configuration

These challenges helped me understand the practical aspects of implementing an automated development and deployment workflow.

## Project Outcome

The completed project demonstrates a complete development workflow in which:

**Code → Automated Testing → Docker Build → Containerized Application**

This provides a practical foundation for applying DevOps practices to Python and machine learning applications.

## Future Improvements

- Add more comprehensive unit and integration tests
- Add API documentation
- Add model performance monitoring
- Add automated model retraining
- Add security scanning to the CI/CD pipeline
- Deploy the containerized application to a cloud platform
- Add monitoring and logging

## Author

**Dinesh Kumar**

GitHub: [Lucky5683](https://github.com/Lucky5683)

## License

This project is licensed under the MIT License.
