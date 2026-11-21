# Student Academic Performance Predictor — End-to-End ML App

<p align="center">
  <img src="notebook/img/confusion_matrix.png" width="500">
</p>

> **Live Demo:** [Try the deployed app on Render](https://student-tier-prediction.onrender.com/)

## 1. The Problem

Educational institutions and instructors need an interpretable way to identify student performance tiers early enough to offer timely support without relying on guesswork or brittle rules.

---

## 2. Objective

The objective of this project is to build a reproducible, end-to-end machine learning pipeline that identifies students likely to perform above or below the median average score—and deploy it as a containerized web application so educators can test predictions interactively.

---

## 3. Methodology

### Data & Target

* **Source:** Kaggle — [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)
* **Features:** Student demographic and background attributes (gender, race/ethnicity, parental level of education, lunch type, and test preparation course).
* **Target Reframing:** Rather than predicting individual exam scores (`math_score`, `reading_score`, and `writing_score`), which felt less practical since exams often take place close together with little time in between, the initial goal was predicting the continuous average score (`avg_score`). After regression experiments showed weak predictive signal, the task was reframed into a **binary classification target** indicating whether `avg_score` is above (`High Performer`) or below (`At-Risk / Standard`) the median.

### Data Exploration & Cleaning

The dataset was inspected for missing values, duplicate records, distributional patterns, and potential data leakage. All preprocessing transformations (encoding and target threshold rules) were fitted strictly on the training split before being applied to test or inference inputs.

### Modeling & Problem Reframing

1. **Regression Baseline & Iteration:** Initial regression models yielded low explanatory power (`R²` around `0.16` at best, and negative `R²` for more complex models). Feature engineering and regularization did not improve regression performance.
2. **Shift to Classification:** Reframing the target into binary performance tiers with **Logistic Regression** provided a cleaner, interpretable baseline suited to the available signal in the data.

### Deployment Pipeline

The fitted preprocessor and `LogisticRegression` model were serialized using `pickle`, wrapped in a **Flask** web application served by **Gunicorn**, containerized with **Docker**, and deployed live on **Render**.

---

## 4. Findings

### Model Performance

The final classification model achieved **67.5% accuracy** on the test set (`200` samples, `100` per class):

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (At-Risk / Standard)** | 0.69 | 0.63 | 0.66 | 100 |
| **1 (High Performer)** | 0.66 | 0.72 | 0.69 | 100 |
| **Accuracy / Macro Avg** | **0.68** | **0.68** | **0.67** | **200** |

### Key Takeaways

* **Honest Evaluation Over Forced Complexity:** A clean, leak-free pipeline and a straightforward `LogisticRegression` model delivered balanced precision and recall across both classes without unnecessary complexity.
* **Reframing Based on Signal & Business Context:** Machine learning models can perform poorly in a regression setup when features lack fine-grained signal, yet reframing the problem as classification can uncover actionable patterns aligned with stakeholder needs.
* **Early Intervention Timing:** While using `reading_score` and `writing_score` to predict `math_score` would yield higher regression metrics, real-world educators need an early signal *before* exam season begins. Predicting one exam after others have already been taken can be too late for timely support.

---

## 5. Next Steps

* **Richer Behavioral Features:** In a real-world deployment, collaborate with stakeholders to collect higher-signal variables such as study hours, attendance, stress levels, or whether the student works, to capture stronger predictive patterns and feature interactions.
* **Revisit Granular Score Prediction:** With stronger input features, revisit regression modeling to estimate continuous scores with higher accuracy.
* **Production Enhancements:** Add request logging, input validation alerts, and threshold calibration based on academic intervention costs.

---

## 6. Technologies

* **Language:** Python 3.11
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`LogisticRegression`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`)
* **Visualization:** Matplotlib, Seaborn
* **Web & Deployment:** Flask, Gunicorn, Docker, Render

---

## Project Structure

```text
├── artifacts/                     # Serialized artifacts and train/test data splits               
├── logs/                          # Execution and error logs
├── notebook/                      # Jupyter notebooks, raw datasets, and figures
├── src/                           # Modular source code package
├── static/                        # Custom styling for the Flask UI            
├── templates/                     # HTML templates for the web app                  
├── app.py                         # Flask application entry point
├── Dockerfile                     # Container build instructions for production
├── README.md                      # Project documentation
├── requirements-dev.txt           # Full dependencies for local development and EDA
├── requirements.txt               # Lightweight runtime dependencies for Docker
└── setup.py                       # Package installation configuration
```

---

## Installation & Local Setup

### 1. Clone the repository

```bash
git clone [https://github.com/fiorellatrigo/student-performance-prediction-app.git](https://github.com/fiorellatrigo/student-performance-prediction-app.git)
cd student-performance-prediction-app

```

### 2. Run with Docker (Recommended)

```bash
docker build -t student-performance-prediction-app .
docker run -p 8000:8000 student-performance-prediction-app

```

Then open `http://127.0.0.1:8000` in your browser.

### 3. Or Run Locally with Python

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements-dev.txt
python app.py

```

---

## References & Data Sources

* **Dataset:** Kaggle — [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977).
