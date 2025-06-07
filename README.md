

## 1. Objective

The objective of this project is to implement two fully automated CI/CD pipelines:

1. A simple Flask web application
2. A Flask application serving a trained machine learning model

Both pipelines are executed using GitHub Actions and Docker, aiming to simulate real-world DevOps workflows in a cost-effective, cloud-free environment.

---

## 2. Tools and Technologies Used

* Python 3.10 — Backend language for both applications
* Flask — Web framework
* Scikit-learn — Machine learning model training
* Git & GitHub — Version control and CI/CD pipeline hosting
* GitHub Actions — Automation of testing and deployment
* Docker — Containerization of the application
* Docker Hub (Optional) — Container registry (if enabled)
* Pytest — Unit testing framework

---

## 3. Project Structure

```
.
├── app.py
├── train_model.py
├── random_forest_model.joblib
├── data.csv
├── test_app.py
├── test_model.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/
    └── ci-cd.yml
```

---

## 4. CI/CD Pipeline Description

### 4.1 Pipeline for Basic Flask App

* Triggered on push to `main` branch
* Steps:

  * Set up Python environment
  * Install dependencies
  * Run unit tests (`test_app.py`)
  * Build Docker image
  * (Optional) Push Docker image to Docker Hub

### 4.2 Pipeline for ML Model Flask App

* ML model trained using `train_model.py` and saved as `random_forest_model.joblib`
* Flask app loads this model and provides predictions via API
* CI/CD steps:

  * Install dependencies
  * Run tests (`test_model.py`)
  * Build Docker image
  * (Optional) Push Docker image to Docker Hub

---

## 5. GitHub Actions Workflow

The workflow is defined in `.github/workflows/ci-cd.yml` and includes:

* Trigger on push to `main` branch
* Setup Python 3.10
* Install dependencies from `requirements.txt`
* Run tests with `pytest`
* Docker login to Docker Hub using GitHub secrets
* Build and optionally push Docker image

---

## 6. Unit Testing

Two test files included:

* `test_app.py` — Tests basic Flask routes
* `test_model.py` — Tests ML model loading and prediction

Example test snippet:

```python
from joblib import load
import numpy as np

def test_model_prediction():
    model = load('random_forest_model.joblib')
    sample_input = np.array([[1, 2, 3, 4]])
    prediction = model.predict(sample_input)
    assert prediction is not None
```

---

## 7. Docker Integration

### Build Docker Image Locally

```bash
docker build -t ci-cd-app .
```

### Run Docker Container

```bash
docker run -p 5000:5000 ci-cd-app
```

### Run with Docker Compose

```bash
docker-compose up
```

---

## 8. Challenges Faced

| Challenge                        | Solution                                 |
| -------------------------------- | ---------------------------------------- |
| Git error: `src refspec main...` | Created initial commit, confirmed branch |
| Docker login issues in Actions   | Used GitHub Secrets for credentials      |
| Docker push errors               | Verified Dockerfile and credentials      |

---

## 9. Outcome and Deliverables

* CI/CD pipeline for Flask app: Completed
* CI/CD pipeline for ML app: Completed
* Dockerfile and Docker Compose support: Included
* GitHub Actions workflow: Operational
* Unit tests: Passing successfully

---

## 10. Conclusion

This project successfully demonstrates the application of professional CI/CD practices to AI and Data Science projects. By integrating automated testing, containerization, and deployment workflows, it simulates a real-world DevOps environment that enhances development efficiency and reliability.

---
