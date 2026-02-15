# Breast-Cancer-Detection

# 🩺 Breast Cancer Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Tools & Technologies](#-tools--technologies)
- [Project Structure](#-project-structure)
- [Data Preprocessing](#-data-preprocessing)
- [Model Development](#-model-development)
- [Model Evaluation](#-model-evaluation)
- [Feature Importance](#-feature-importance)
- [Key Insights](#-key-insights)
- [How to Run](#-how-to-run)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 📖 Overview

This project focuses on detecting whether a breast tumor is **malignant (cancerous)** or **benign (non-cancerous)** using Machine Learning techniques.

The system analyzes medical features such as tumor radius, texture, perimeter, and area to classify the diagnosis with high accuracy.

Developed using **Python in Kaggle Jupyter Notebook**, this project demonstrates complete Data Science workflow including preprocessing, EDA, model training, evaluation, and prediction.

---

## 💼 Business Problem

Breast cancer is one of the most common cancers worldwide. Early detection significantly increases survival rates.

Manual diagnosis based on multiple measurements can be complex and error-prone.

### 🎯 Objective:

- Automatically classify tumors as malignant or benign  
- Assist healthcare professionals in diagnosis  
- Reduce diagnostic time  
- Improve decision-making accuracy  

---

## 📊 Dataset

**Source:** Breast Cancer Wisconsin Dataset (Kaggle)  
**Records:** 569  
**Features:** 30 numerical features  
**Target Variable:** `diagnosis`

- `M` → Malignant  
- `B` → Benign  

The dataset contains characteristics of cell nuclei extracted from breast mass images.

---

## 🛠 Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- StandardScaler
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Cross Validation
- ROC Curve
- Confusion Matrix

---

## 📂 Project Structure
breast-cancer-detection/
│
├── breast-cancer-detection.ipynb
├── requirements.txt
├── README.md
└── .gitignore


---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

- Removed duplicate records  
- Checked and handled missing values  
- Encoded target variable (`M` → 1, `B` → 0)  
- Feature scaling using StandardScaler  
- Train-Test Split (80/20 ratio)  
- Outlier detection using boxplots and IQR method  

---

## 🤖 Model Development

### 🔎 Problem Type:
Binary Classification

### 🧠 Models Implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. Random Forest Classifier  

### ⚙ Training Strategy:

- Train-Test Split (80/20)  
- 5-Fold Cross Validation  
- Pipeline-based preprocessing  

---

## 📈 Model Evaluation

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve

### 🎯 Performance Comparison

| Model | Mean Accuracy |
|--------|---------------|
| Logistic Regression | ~97% |
| Decision Tree | ~92% |
| Random Forest | ~98% |

Random Forest achieved the highest accuracy.

---

## 📊 Feature Importance

Random Forest identified the most influential features:

- radius_mean  
- perimeter_mean  
- area_mean  
- texture_mean  

These tumor size-related features strongly influence malignancy prediction.

---

## 🔍 Key Insights

- Tumor size measurements are highly correlated with cancer diagnosis.
- Feature scaling significantly improves model performance.
- Logistic Regression performs very well due to linear separability.
- Random Forest provides robust performance and reduces overfitting.
- Machine Learning can assist in early cancer detection.

---
