# 🌦️ Weather Prediction System

## 📌 Overview

The **Weather Prediction System** is a modular, machine learning–based platform designed to predict **multiple weather parameters** using historical climatic and geographical data.

Instead of being a single-purpose model, the system is structured as a **collection of independent subsystems**, where each subsystem is responsible for predicting a specific weather condition.

One such subsystem is the **Rainfall Prediction System**, which operates independently and can be developed, trained, and deployed without affecting other subsystems.

---

## 🧩 System Architecture (Modular Design)

The overall system follows a **modular architecture**, where:

- Each weather parameter is handled by a **separate ML subsystem**
- Subsystems share common preprocessing logic but have **independent models**
- New subsystems can be added without modifying existing ones

### Example Subsystems

- 🌧️ Rainfall Prediction Subsystem  
- 🌡️ Temperature Prediction Subsystem  
- 💧 Humidity Prediction Subsystem  
- 🌬️ Wind Speed Prediction Subsystem  
- ⛈️ Extreme Weather / Storm Prediction Subsystem  

---

## 🌧️ Rainfall Prediction Subsystem

### Description

The **Rainfall Prediction Subsystem** is an independent machine learning module that predicts the **amount of rainfall (in mm)** based on climatic and geographical inputs.

This subsystem:
- Can function **standalone**
- Has its **own trained model**
- Uses the same input interface pattern as other subsystems
- Can be deployed separately using Streamlit

---

### 🎯 Objective

To predict rainfall accurately using historical data and user-provided climatic inputs such as:
- State
- Month
- Temperature
- Humidity
- Wind speed

---

### 🧠 ML Details (Rainfall Subsystem)

- **Problem Type:** Regression  
- **Target Variable:** Rainfall (mm)  
- **Model Used:** Linear Regression / Logistic Regression (as implemented)  

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Libraries:**
  - pandas
  - numpy
  - scikit-learn
  - streamlit

---

## 📂 Project Structure

weather-prediction-system/
│
├── rainfall_subsystem/
│ ├── rainfall_model.pkl
│ ├── rainfall_app.py
│ └── rainfall_preprocessing.py
