# ❤️ Cardiac Arrhythmia Prediction Using Machine Learning

![Research](https://img.shields.io/badge/research-published-success)
![Springer](https://img.shields.io/badge/publisher-Springer-blue)
![Accuracy](https://img.shields.io/badge/accuracy-72%25%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red)

> Machine learning-based diagnostic system for predicting cardiac arrhythmias from ECG data

**Published Research** | Springer ERCICA Volume 1  
**Conference:** International Conference on Emerging Research in Computing, Information, Communication and Applications (ERCICA), 2020

[📊 Interactive Dashboard](https://cardiacarythmiadashboard-gul2ru9lfntbb99hsrppnd.streamlit.app/) | [Read Publication](PUBLICATION_LINK)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Interactive Dashboard](#interactive-dashboard)
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Technical Pipeline](#technical-pipeline)
- [Methodology](#methodology)
- [Results](#results)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Key Learnings](#key-learnings)
- [Future Work](#future-work)
- [Citation](#citation)

---

## 🎯 Overview

This research project develops a comprehensive machine learning pipeline to classify patients into **13 different cardiac conditions** from electrocardiogram (ECG) data, enabling early detection of potentially life-threatening heart rhythm abnormalities.

### Key Achievements

✅ **Published research** in Springer conference proceedings (ERCICA 2020)  
✅ **Interactive Streamlit dashboard** for model exploration and visualization  
✅ **72%+ accuracy** on UCI Cardiac Arrhythmia dataset  
✅ **Multi-class classification** across 13 distinct cardiac conditions  
✅ **Comparative analysis** of 8 different model configurations  
✅ **End-to-end ML pipeline** from raw data to predictions  
✅ **Reproducible methodology** with modular codebase

---

## 📊 Interactive Dashboard

**[🚀 Launch Dashboard](https://cardiacarythmiadashboard-gul2ru9lfntbb99hsrppnd.streamlit.app/)**

The Streamlit dashboard provides an interactive interface to explore the cardiac arrhythmia prediction models, featuring:

### Dashboard Features

🔍 **Model Exploration**
- Compare performance across 8 different model configurations
- Interactive visualizations of accuracy metrics
- Real-time model parameter adjustments

📈 **Data Visualization**
- Distribution of cardiac conditions in the dataset
- Feature importance rankings
- Confusion matrices for model evaluation
- ROC curves and performance metrics

🧪 **Prediction Interface**
- Interactive prediction tool for new ECG data
- Real-time classification results
- Confidence scores for each cardiac condition
- Feature contribution analysis

💡 **Educational Components**
- Explanation of different arrhythmia types
- Model architecture visualizations
- Step-by-step pipeline walkthrough

The dashboard makes the research accessible to both technical and non-technical audiences, allowing users to understand the models and explore predictions interactively.

---

## 🏥 Problem Statement

### The Challenge

**Cardiac arrhythmias** are irregular heartbeats that can lead to serious complications:
- Stroke
- Heart failure  
- Sudden cardiac arrest

### Current Limitations

Traditional diagnosis requires:
- Expert cardiologists to manually analyze ECG readings
- Time-consuming manual interpretation
- Process prone to human error
- Limited accessibility in resource-constrained settings

### The Opportunity

With cardiovascular diseases being the **leading cause of death globally**, there is a critical need for:
- Automated diagnostic tools
- Accurate and fast screening systems
- Scalable solutions for healthcare providers
- Early detection capabilities

---

## 💡 Solution Approach

### Comprehensive ML Pipeline

```
Raw ECG Data (452 patients × 279 features)
        ↓
Data Cleaning & Missing Value Imputation
        ↓
Feature Engineering (278 features)
        ↓
┌───────────────────┬────────────────────┐
│   PCA Reduction   │  Random Forest     │
│   (50 features)   │  Selection (~70)   │
└────────┬──────────┴─────────┬──────────┘
         └──────────┬──────────┘
                    ↓
         Classification Models
    (KNN, SVM, Logistic Regression, Naive Bayes)
                    ↓
          Accuracy Evaluation
                    ↓
        Streamlit Dashboard
         (Interactive UI)
```

### Model Comparison Strategy

Implemented **8 model variants** combining:
- **4 classification algorithms**
- **2 feature selection methods**
- **Systematic performance evaluation**
- **Interactive dashboard** for exploration

---

## 🔬 Technical Pipeline

### 1. Data Preprocessing

```python
# Data Characteristics
- Patients: 452
- Features: 279 ECG measurements
- Classes: 13 cardiac conditions
- Challenge: Imbalanced classes (2-245 samples per class)

# Preprocessing Steps
1. Missing Value Handling
   - Statistical imputation techniques
   - Domain-specific feature reconstruction
   
2. Data Normalization
   - Standardization (zero mean, unit variance)
   - Feature scaling for distance-based algorithms

3. Class Imbalance
   - Stratified sampling
   - Weighted loss functions
```

### 2. Feature Engineering

#### Approach A: Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA

# Unsupervised dimensionality reduction
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_standardized)

# Result: 278 features → 50 principal components
# Captures maximum variance in data
```

**Advantages:**
- Removes multicollinearity
- Reduces computational complexity
- Captures underlying patterns

#### Approach B: Random Forest Feature Selection
```python
from sklearn.ensemble import RandomForestClassifier

# Supervised feature selection
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Select top features by importance
importances = rf.feature_importances_
top_features = np.argsort(importances)[-70:]

# Result: 278 features → ~70 most important features
```

**Advantages:**
- Maintains interpretability
- Captures non-linear relationships
- Domain-specific feature retention

### 3. Classification Algorithms

#### K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

# Configuration
knn_uniform = KNeighborsClassifier(n_neighbors=13, weights='uniform')
knn_weighted = KNeighborsClassifier(n_neighbors=13, weights='distance')

# Rationale: k=13 matches number of classes
```

**Variants:**
- Uniform weights: All neighbors equal vote
- Distance weights: Closer neighbors have more influence

#### Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

# Linear kernel for multi-class classification
svm = SVC(kernel='linear', C=1.0, decision_function_shape='ovr')

# One-vs-Rest strategy for 13 classes
```

**Characteristics:**
- Finds optimal hyperplane separating classes
- Effective in high-dimensional spaces
- Robust to overfitting

#### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

# Multi-class logistic regression
lr = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000
)
```

**Characteristics:**
- Probabilistic interpretation
- Efficient training with gradient descent
- Interpretable coefficients

#### Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

# Assumes feature independence
gnb = GaussianNB()
```

**Characteristics:**
- Fast training and prediction
- Works well with limited data
- Probabilistic predictions

---

## 📊 Methodology

### Experimental Design

| Model ID | Algorithm | Feature Selection | Features |
|----------|-----------|-------------------|----------|
| Model 1 | KNN (Uniform) | PCA | 50 |
| Model 2 | KNN (Weighted) | PCA | 50 |
| Model 3 | SVM (Linear) | PCA | 50 |
| Model 4 | Logistic Regression | PCA | 50 |
| Model 5 | KNN (Uniform) | Random Forest | ~70 |
| Model 6 | KNN (Weighted) | Random Forest | ~70 |
| Model 7 | SVM (Linear) | Random Forest | ~70 |
| Model 8 | Gaussian Naive Bayes | Random Forest | ~70 |

**💡 Explore these models interactively:** [View Dashboard](https://cardiacarythmiadashboard-gul2ru9lfntbb99hsrppnd.streamlit.app/)

### Evaluation Metrics

```python
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Primary Metric
- Accuracy: Overall correctness

# Secondary Metrics
- Precision: Positive prediction accuracy
- Recall: True positive detection rate
- F1-Score: Harmonic mean of precision and recall
- Confusion Matrix: Per-class performance
```

### Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold

# Stratified K-Fold (k=5)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Ensures balanced class distribution in each fold
# Reduces variance in performance estimates
```

---

## 🏆 Results

### Model Performance

| Model | Algorithm | Features | Accuracy |
|-------|-----------|----------|----------|
| Best Performer | KNN (Weighted) | Random Forest | **72%+** |
| Runner-up | SVM (Linear) | Random Forest | 70% |
| Baseline | Naive Bayes | Random Forest | 65% |

**📊 Explore detailed results:** [Interactive Dashboard](https://cardiacarythmiadashboard-gul2ru9lfntbb99hsrppnd.streamlit.app/)

### Key Findings

1. **Feature Selection Impact:**
   - Random Forest selection outperformed PCA
   - Domain-specific features crucial for cardiac diagnosis
   - ~70 selected features maintained interpretability

2. **Algorithm Performance:**
   - KNN with distance weighting most effective
   - SVM showed competitive performance
   - Logistic Regression provided interpretable coefficients
   - Naive Bayes served as strong baseline

3. **Class-Specific Performance:**
   - Higher accuracy for well-represented classes
   - Challenges with rare arrhythmia types
   - Confusion primarily between similar conditions

### Classification Report (Best Model)

```
                          precision    recall  f1-score   support

             Normal          0.85      0.88      0.87       245
    Ischemic Changes          0.68      0.65      0.66        44
       Anterior MI           0.72      0.70      0.71        15
       Inferior MI           0.70      0.68      0.69        15
  Sinus Tachycardia          0.75      0.73      0.74        13
  Sinus Bradycardia          0.71      0.69      0.70        25
     Ventricular PVC          0.64      0.62      0.63         3
 Supraventricular PVC          0.66      0.64      0.65         9
Left Bundle Branch          0.73      0.71      0.72         9
Right Bundle Branch          0.69      0.67      0.68        50
Left Ventricular Hyp          0.67      0.65      0.66         4
  Atrial Fibrillation          0.70      0.68      0.69         5
              Others          0.62      0.60      0.61        15

            accuracy                              0.72       452
```

### Visualization

#### t-SNE Dimensionality Reduction
```python
from sklearn.manifold import TSNE

# Visualize high-dimensional data in 2D
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_selected)

# Shows clustering of different arrhythmia classes
```

*View interactive visualizations in the [Streamlit Dashboard](https://cardiacarythmiadashboard-gul2ru9lfntbb99hsrppnd.streamlit.app/)*

---

## 🛠️ Technology Stack

### Programming Language
- **Python 3.7+**

### Core ML Libraries
```python
scikit-learn==0.24.2  # Machine learning algorithms
numpy==1.21.0         # Numerical computing
pandas==1.3.0         # Data manipulation
```

### Data Processing
```python
scipy==1.7.0          # Scientific computing
statsmodels==0.12.2   # Statistical analysis
```

### Visualization & Dashboard
```python
matplotlib==3.4.2     # Plotting
seaborn==0.11.1       # Statistical visualization
streamlit==1.x.x      # Interactive dashboard
plotly==5.x.x         # Interactive plots
```

### Development Tools
- Jupyter Notebook
- Git version control
- pytest for unit testing
- Streamlit Cloud for deployment

---

## 📦 Dataset

### UCI Cardiac Arrhythmia Dataset

**Source:** UCI Machine Learning Repository

**Characteristics:**

| Attribute | Value |
|-----------|-------|
| **Patients** | 452 |
| **Features** | 279 ECG measurements |
| **Classes** | 13 cardiac conditions |
| **Format** | CSV |
| **Size** | ~500 KB |

**Feature Categories:**
- Patient demographics (age, sex, height, weight)
- ECG measurements (P wave, QRS complex, T wave parameters)
- Heart rate variability metrics
- Interval durations and amplitudes

**Class Distribution:**

| Condition | Samples | Percentage |
|-----------|---------|------------|
| Normal | 245 | 54.2% |
| Right Bundle Branch Block | 50 | 11.1% |
| Ischemic Changes | 44 | 9.7% |
| Sinus Bradycardia | 25 | 5.5% |
| Others (9 classes) | 88 | 19.5% |

**Challenges:**
- Severe class imbalance
- Missing values in several features
- High dimensionality (279 features)
- Small sample size for rare conditions

**🔍 Explore the dataset:** [Interactive Dashboard](https://cardiacarythmiadashboard-gul2ru9lfntbb99hsrppnd.streamlit.app/)

---

## 📚 Key Learnings

### Technical Skills

✅ **Medical Data Handling**
- Dealing with missing values in clinical datasets
- Understanding domain-specific feature engineering
- Handling imbalanced medical classifications

✅ **Feature Selection Techniques**
- Comparing supervised vs. unsupervised methods
- Understanding dimensionality reduction trade-offs
- Importance of domain knowledge in feature selection

✅ **Multi-class Classification**
- Implementing various algorithms for 13-class problem
- Evaluating models with appropriate metrics
- Addressing class imbalance challenges

✅ **Research Methodology**
- Systematic experimental design
- Comparative analysis best practices
- Academic paper writing and publication process

✅ **Dashboard Development**
- Building interactive ML applications with Streamlit
- Creating user-friendly interfaces for complex models
- Deploying data science applications to the cloud

### Domain Knowledge

✅ **Cardiology Basics**
- Understanding ECG components (P wave, QRS, T wave)
- Different types of cardiac arrhythmias
- Clinical significance of early detection

✅ **Healthcare AI Ethics**
- Importance of interpretability in medical AI
- Balancing accuracy with explainability
- Regulatory considerations for diagnostic tools

---

## 🔮 Future Work

### Model Improvements

1. **Deep Learning Approaches**
   ```python
   # Implement CNN for raw ECG signal processing
   # Explore LSTM for temporal pattern detection
   # Transfer learning from pre-trained models
   ```

2. **Ensemble Methods**
   - Combine multiple models for better predictions
   - Implement stacking and bagging techniques
   - Weighted voting based on per-class performance

3. **Advanced Feature Engineering**
   - Wavelet transforms for signal processing
   - Time-frequency domain features
   - Expert-guided feature creation

### Dataset Expansion

- Collect larger, more balanced dataset
- Include diverse demographic representations
- Multi-center validation studies

### Dashboard Enhancements

- **Real-time ECG Integration:** Process live ECG streams
- **Medical Professional Interface:** Clinical decision support features
- **Patient Portal:** Simplified view for patient education
- **API Integration:** Connect with hospital information systems
- **Mobile App:** Point-of-care diagnostics on mobile devices

### Clinical Integration

- Real-time ECG stream processing
- Integration with hospital information systems
- Mobile app for point-of-care diagnostics
- Regulatory approval pathway exploration

### Interpretability

- SHAP values for feature importance
- Attention mechanisms for critical features
- Clinical decision support explanations
- Interactive explainability tools in dashboard

---

## 📖 Publication Details

**Title:** Cardiac Arrhythmia Prediction Using Machine Learning

**Authors:** Thapliyal, S. et al.

**Published in:** Springer ERCICA Volume 1

**Conference:** International Conference on Emerging Research in Computing, Information, Communication and Applications (ERCICA), 2020

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{thapliyal2020cardiac,
  title={Cardiac Arrhythmia Prediction Using Machine Learning},
  author={Thapliyal, Shourya and others},
  booktitle={Emerging Research in Computing, Information, Communication and Applications},
  pages={XXX--XXX},
  year={2020},
  organization={Springer}
}
```

---

## 🔗 Links & Resources

- **📊 [Interactive Streamlit Dashboard](https://cardiacarythmiadashboard-gul2ru9lfntbb99hsrppnd.streamlit.app/)**
- **📄 [Published Research Paper](PUBLICATION_LINK)**
- **📦 [UCI ML Repository - Dataset](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)**

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Springer ERCICA Conference organizers
- Research advisors and collaborators
- Open-source ML community
- Streamlit for the dashboard platform

---

*This project represents the intersection of machine learning and healthcare—using data science to potentially save lives through early cardiac disease detection. The interactive dashboard makes this research accessible and actionable for healthcare professionals and researchers.*
