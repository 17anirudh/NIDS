# Network Intrusion Detection System (NSL-KDD)

[![Deployment](https://img.shields.io/badge/Deployment-Render-blue)](https://intrusion-detection-yt2w.onrender.com/)

Machine learning application that detects intrusions using the NSL-KDD dataset. This project uses various classification algorithms to recognizes abnormalities.

## Overview

This application uses machine learning models trained on the NSL-KDD dataset to classify network traffic as either normal or intrusive. Users can input network parameters through a web interface, and the system returns a binary prediction indicating whether the traffic is benign or malicious.

## Features

- Web-based interface for easy interaction
- Multiple machine learning models for improved accuracy:
  - Decision Trees
  - Random Forest
  - Naive Bayes
  - Stacking
  - Bagging
  - Gradient Boosting
  - XGBoost
  - CatBoost
- Model persistence using pickle/joblib
- Build backend using Flask
- Deployed on Render for accessibility

## Technology Stack

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn, XGBoost, CatBoost
- **Model Serialization**: pickle, joblib
- **Backend**: Flask (Python)
- **Deployment**: Render

## Installation and Usage

### Option 1: Use the Deployed Application

Access the application directly at: [Network Intrusion Detection System](https://intrusion-detection-yt2w.onrender.com/)

### Option 2: Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/17anirudh/nsl-kdd.git
   cd nsl-kdd
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## How to Use

1. Fill in the network parameters in the HTML form.
2. Submit the form to get a prediction.
3. The system will display whether the network traffic is normal or an intrusion.

## Dataset

This project uses the NSL-KDD dataset, an improved version of the KDD Cup 1999 dataset. It contains various network connection features and is labeled for different types of intrusions.
```
cd docs
```

## Model Training

The models were trained using various algorithms:
  - Decision Trees
  - Random Forest
  - Naive Bayes
  - Stacking
  - Bagging
  - Gradient Boosting
  - XGBoost
  - CatBoost

## Contact

Anirudh - [@17anirudh](https://github.com/17anirudh)

Project Link: [https://github.com/17anirudh/nsl-kdd](https://github.com/17anirudh/nsl-kdd)
