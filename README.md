# 🏥 Hospital Bed Occupancy Predictor

A Machine Learning-based web application that predicts **hospital bed occupancy rates** using hospital capacity, ICU, demographic, and regional information.

The project combines a trained machine learning model with a Flask web application to provide an easy-to-use prediction interface for hospital resource planning and decision support.

---

## 📌 Project Overview

Efficient hospital bed management is essential for reducing overcrowding, improving patient flow, and optimizing healthcare resources. Unpredictable changes in patient admissions and hospital capacity can make manual occupancy estimation difficult.

This project develops a Machine Learning solution for predicting **All Bed Occupancy Rate** using historical hospital and regional data. The project aims to support proactive hospital resource planning and data-driven decision-making.

The research behind the project explores multiple machine learning and forecasting approaches, including Regression, Random Forest, Gradient Boosting, ARIMA, XGBoost, and LSTM.

---

## 🎯 Objectives

* Predict hospital bed occupancy rates using Machine Learning.
* Analyze hospital capacity and demographic factors affecting occupancy.
* Handle missing numerical data and categorical variables during preprocessing.
* Build a reusable ML pipeline for prediction.
* Provide a simple web interface for entering hospital information.
* Display the predicted occupancy rate visually.
* Support better hospital resource allocation and planning.
* Develop a scalable approach that can potentially be adapted to different hospitals and regions.

---

## 🚀 Features

### Machine Learning

* Gradient Boosting Regression
* Numerical feature imputation
* StandardScaler normalization
* One-Hot Encoding for categorical features
* Scikit-learn Pipeline
* Pre-trained model saved using Joblib

### Web Application

* Flask-based backend
* Responsive HTML interface
* Hospital information input form
* Bed occupancy prediction
* Interactive occupancy visualization
* Dark-themed user interface
* Chart.js doughnut chart

The Flask application loads the trained `model.pkl` file and creates a prediction from the submitted hospital information.

---

## 🧠 Machine Learning Workflow

```text
                ┌──────────────────┐
                │   Dataset.csv    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Data Preprocessing│
                └────────┬─────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Numerical Features    Categorical Features
              │                     │
              ▼                     ▼
        Median Imputation     One-Hot Encoding
              │                     │
              ▼                     │
       Standard Scaling             │
              └──────────┬──────────┘
                         ▼
                ┌──────────────────┐
                │ Gradient Boosting│
                │    Regressor     │
                └────────┬─────────┘
                         │
                         ▼
                   model.pkl
                         │
                         ▼
                ┌──────────────────┐
                │   Flask Web App  │
                └────────┬─────────┘
                         │
                         ▼
             Predicted Bed Occupancy
```

The implemented training pipeline uses `GradientBoostingRegressor`, with separate numerical and categorical preprocessing pipelines.

---

## 📊 Input Features

The application accepts the following information:

| Feature                | Description                     |
| ---------------------- | ------------------------------- |
| Staffed All Beds       | Number of staffed hospital beds |
| Staffed ICU Beds       | Number of staffed ICU beds      |
| Licensed All Beds      | Number of licensed beds         |
| ICU Bed Occupancy Rate | ICU occupancy rate              |
| Population             | Total population                |
| Population (20+)       | Population aged 20 and above    |
| Population (65+)       | Population aged 65 and above    |
| State                  | State/region                    |
| County Name            | County name                     |
| ICU Bed Source         | Source of ICU bed information   |

These fields correspond to the features used by the implemented preprocessing and prediction pipeline.

---

## 🎯 Prediction Target

The model predicts:

**All Bed Occupancy Rate**

## During training, rows without a target occupancy value are removed and the trained pipeline is saved as `model.pkl`.

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Gradient Boosting Regression
* Pandas
* NumPy
* Joblib

### Web Development

* Flask
* HTML
* CSS
* JavaScript
* Chart.js

### Development Tools

* Jupyter Notebook
* Python IDE / VS Code
* Git & GitHub

The project's requirements file includes Flask, Pandas, Scikit-learn, and Joblib.

---

## 📁 Project Structure

```text
Hospital-Bed-Occupancy-Predictor/
│
├── app.py
├── model_pipeline.py
├── model.pkl
├── Dataset.csv
├── requirements.txt
├── ML_Project.ipynb
│
├── templates/
│   └── index.html
│
└── README.md
```

> If `index.html` is currently in the project root, move it into a `templates/` directory because Flask's `render_template("index.html")` expects the template there.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Hospital-Bed-Occupancy-Predictor.git
cd Hospital-Bed-Occupancy-Predictor
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

The application will run locally at:

```text
http://127.0.0.1:5000/
```

Open the address in your browser.

---

## 🧪 Train the Model

If you want to retrain the model using `Dataset.csv`, run:

```bash
python model_pipeline.py
```

The script will:

1. Load `Dataset.csv`
2. Remove unnecessary data
3. Handle missing target values
4. Separate numerical and categorical features
5. Impute missing numerical values
6. Standardize numerical features
7. Encode categorical features
8. Train the Gradient Boosting Regressor
9. Save the trained model as `model.pkl`

## The implemented training script saves the final pipeline using Joblib.

## 🖥️ Web Interface

The application provides a form where users enter hospital and regional information and click **Predict Now**.

The interface displays:

```text
Predicted Bed Occupancy Rate: XX%
```

It also generates a doughnut chart showing:

* Occupied Beds
* Available Beds

The HTML interface uses Chart.js for the occupancy visualization.

---

## 🔬 Project Methodology

The overall project methodology includes:

### 1. Data Collection

Hospital and healthcare-related data can contain information about bed availability, occupancy, admissions, discharges, demographics, and external factors.

### 2. Data Preprocessing

The project applies:

* Missing-value handling
* Numerical normalization
* Categorical encoding
* Feature preparation

The current implementation specifically uses median imputation, standard scaling, and one-hot encoding.

### 3. Model Development

The current web application uses a **Gradient Boosting Regressor** for occupancy prediction.

### 4. Prediction

The Flask backend receives user inputs, constructs a Pandas DataFrame, passes it to the trained model, and returns the prediction.

### 5. Visualization

The predicted occupancy is displayed with an interactive Chart.js doughnut chart.

---

## 📈 Evaluation Metrics

The project documentation identifies the following metrics for evaluating occupancy prediction models:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error
* **R² Score** — Coefficient of Determination

These metrics are intended to measure prediction error and model performance.

The research documentation also compares different approaches such as Linear Regression, Decision Trees, Random Forest, XGBoost, and LSTM.

---

## 💡 Applications

The proposed system can support:

* 🏥 Hospital capacity planning
* 🛏️ Bed allocation
* 👨‍⚕️ Staff planning
* 🚑 Emergency preparedness
* 📊 Healthcare resource management
* 📈 Hospital occupancy monitoring
* 🦠 Surge and outbreak preparedness

The project's documented motivation is to move hospital planning from reactive decision-making toward proactive, data-driven resource management.

---

## 🔮 Future Enhancements

Possible future improvements include:

* Real-time hospital data integration
* Integration with EHR/EMR systems
* Automatic API-based prediction updates
* Weather and epidemiological data integration
* Advanced time-series forecasting
* LSTM-based forecasting
* XGBoost and hybrid models
* Occupancy threshold alerts
* Interactive hospital dashboards
* Model monitoring and retraining
* Deployment using Docker and cloud platforms

The project documentation specifically identifies real-time APIs, additional external data, hybrid models, and interactive dashboards as future directions.

---

## ⚠️ Limitations

* Model performance depends on the quality and availability of data.
* A model trained on one hospital's data may not generalize directly to another hospital.
* Real-time integration requires continuously updated data.
* More complex deep-learning models require greater computational resources.
* Clinical deployment would require appropriate healthcare data privacy, security, and regulatory considerations.

---

## 👨‍💻 Project Team

**Prashant Mani Tripathi**
**Naveen Sindhu**
**G Sharath**

Lovely Professional University
Department of Computer Science and Engineering

The project report identifies the three students as the project contributors.

---

## 📚 Research

This project is supported by project documentation and research work titled:

> **"Predicting Hospital Bed Occupancy Rates Using Machine Learning"**

The research discusses the use of predictive analytics for hospital resource planning and evaluates multiple supervised learning approaches.

---

## 📜 Patent / Intellectual Property

The project documentation describes an invention titled:

**"ML Model Which Predicts Hospital Bed Occupancy Rate"**

The proposed invention focuses on healthcare informatics, predictive analytics, and machine-learning-based hospital resource management.

---

## ⚕️ Disclaimer

This project is intended for **educational, research, and prototype purposes**.

Predictions generated by the application should not be treated as medical advice or as a replacement for decisions made by qualified healthcare professionals or hospital administrators.

Real-world clinical deployment would require appropriate validation, data governance, privacy protections, security controls, and regulatory compliance.

---

## ⭐ Project Highlights

```text
Machine Learning       → Gradient Boosting Regression
Data Processing        → Pandas + Scikit-learn
Model Pipeline         → Imputation + Scaling + Encoding
Backend                → Flask
Frontend               → HTML + CSS + JavaScript
Visualization          → Chart.js
Model Storage          → Joblib
Target                 → All Bed Occupancy Rate
```

---

## 📄 License

This project is intended for educational and academic purposes. Add an appropriate open-source license before distributing the project commercially or as open-source software.

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub!
