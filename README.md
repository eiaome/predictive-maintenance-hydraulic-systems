# Predictive Maintenance of Hydraulic Systems Using Machine Learning

## Overview

This capstone project applies machine learning techniques to predict hydraulic system conditions using sensor data from the ZeMA Hydraulic System Condition Monitoring Dataset.

The study compares the performance of three machine learning classification models:

- Decision Tree
- Random Forest
- XGBoost

The primary goal is to support predictive maintenance by identifying potential hydraulic system faults before failure occurs. The project also includes sensor importance analysis, model explainability, and a diagnostic decision-support console for maintenance recommendations.

---

## Authors

- Eurika Innah Alyza Estocapio
- Jemimah Bless Bayer
- Marvin Bontigao
- Kurtdane Dela Cruz
- Gian Joshua Lagunero
- Jathniel Manangbao
- Jeff Kervi Quilang
- Murphy Tangilag

**BS Marine Engineering Capstone Project**

---

## Research Objectives

- Develop machine learning models for hydraulic condition monitoring.
- Compare the performance of Decision Tree, Random Forest, and XGBoost classifiers.
- Identify the most influential sensors affecting system condition predictions.
- Evaluate sensor importance and feature contributions.
- Investigate the effects of data leakage on model performance.
- Develop a practical diagnostic decision-support console.
- Demonstrate the application of predictive maintenance in engineering systems.

---

## Dataset

This study utilizes the **ZeMA Hydraulic System Condition Monitoring Dataset**, a widely used benchmark dataset for predictive maintenance research.

### Dataset Sources

**UCI Machine Learning Repository**

http://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems

**GitHub Reference Repository**

https://github.com/Machine-Learning-FGA/Hydraulic-systems

### Sensor Categories

The dataset contains measurements from:

- Temperature Sensors (TS1–TS4)
- Pressure Sensors (PS1–PS6)
- Flow Sensors (FS1–FS2)
- Vibration Sensor (VS1)
- Motor Power Measurements
- Cooling Efficiency Measurements

### Target Conditions

The models predict the operating condition of:

- Cooler
- Valve
- Pump
- Hydraulic Accumulator

---

## Machine Learning Workflow

### Data Preparation

- Data cleaning
- Feature extraction
- Data normalization
- Train-test splitting
- Cross-validation

### Feature Engineering

The following statistical features were extracted from sensor signals:

- Mean
- Standard Deviation
- Minimum
- Maximum
- Percentiles
- Range
- Root Mean Square (RMS)
- Signal Drift

### Models Evaluated

#### Decision Tree

A simple and interpretable classification model used as the baseline algorithm.

#### Random Forest

An ensemble learning approach that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

#### XGBoost

An advanced gradient-boosting algorithm known for high predictive performance and efficiency.

---

## Evaluation Metrics

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Accuracy

Measures the proportion of correctly classified observations.

### Precision

Measures the proportion of positive predictions that are correct.

### Recall

Measures the proportion of actual positive cases successfully identified.

### F1 Score

Provides a balance between Precision and Recall.

---

## Results and Visualizations

### Model Performance Comparison

screenshots/model_comparison.png

---

### Confusion Matrix

screenshots/confusion_matrix.png

---

### Feature Importance Analysis

screenshots/feature_importance.png

---

### Sensor Ablation Analysis

The study investigated the impact of removing individual sensors and sensor groups to identify the most critical measurements for predictive maintenance.

Key objectives:

- Determine essential sensors
- Evaluate sensor redundancy
- Support cost-effective monitoring system design

---

### Data Leakage Investigation

A leakage analysis was performed to determine whether derived or virtual sensor variables artificially inflated model performance.

This analysis helps ensure that the final predictive model reflects realistic operational deployment conditions.

---

## Diagnostic Decision-Support Console

A diagnostic console was developed to convert machine learning predictions into practical maintenance recommendations.

### Console Demonstration

screenshots/console_demo.png

### Console Outputs

The system categorizes equipment conditions as:

- NORMAL
- WARNING
- CRITICAL

Based on predicted conditions, the console generates maintenance recommendations to support engineering decision-making.

---

## Repository Contents

```text
predictive-maintenance-hydraulic-systems/
│
├── MOL_capstone.ipynb
├── diagnostic_console.py
├── requirements.txt
├── README.md
│
└── screenshots/
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── console_demo.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/eiaome/predictive-maintenance-hydraulic-systems.git
```

Navigate to the project folder:

```bash
cd predictive-maintenance-hydraulic-systems
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

Open the notebook using:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

Then run:

```text
MOL_capstone.ipynb
```

---

## Running the Diagnostic Console

```bash
python diagnostic_console.py
```

---

## Software and Libraries

The project was developed using:

- Python
- NumPy
- Pandas
- Scikit-Learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- Gradio

---

## Data Attribution

The dataset used in this study is publicly available through the UCI Machine Learning Repository and related research publications.

All rights to the dataset remain with the original creators.

This repository contains the **code, analyses, visualizations, and documentation developed as part of the capstone project.**

---

## Capstone Paper

The complete capstone manuscript is available upon request.

---

## Acknowledgments

Special thanks to:

- ZeMA Hydraulic System Dataset Developers
- UCI Machine Learning Repository
- Scikit-Learn Development Team
- XGBoost Development Team
- Open-Source Software Community

---

## Disclaimer

This repository was developed for academic and educational purposes as part of a Bachelor of Science in Marine Engineering capstone project.

**The machine learning workflows were implemented using established open-source libraries and adapted methodologies from the predictive maintenance and condition monitoring literature.**

---

## Contact

For academic inquiries regarding this project, please contact the project authors through their respective institutional affiliations. 

## Citation
If you use this repository, please cite:

Tangilag, M. et al, (2026). Predictive Maintenance of Hydraulic Systems Using Machine Learning: A Comparative Analysis of Decision Tree, Random Forest, and XGBoost.
