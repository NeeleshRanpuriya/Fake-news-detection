# Fake News Prediction App

This project is a Streamlit app that predicts whether a news article is **real** or **fake** using an ensemble of trained machine learning models.

## Project Files

- `app.py` - Streamlit web app
- `news.ipynb` - notebook used for model development/training workflow
- `True.csv` - real news dataset
- `Fake.csv` - fake news dataset
- `vectorizer.pkl` - text vectorizer
- `logistic_model.pkl` - Logistic Regression model
- `decisiontree_model.pkl` - Decision Tree model
- `gradientboost_model.pkl` - Gradient Boosting model
- `randomforest_model.pkl` - Random Forest model

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## How It Works

1. User enters a news article.
2. Text is cleaned and transformed with the saved vectorizer.
3. Four models generate predictions.
4. Final output is based on the majority/percentage of model votes.
