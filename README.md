# Fake News Prediction App

This project detects whether a news article is REAL or FAKE using multiple machine learning models trained on news text data. It includes:

- A full training workflow in `news.ipynb`
- Saved model artifacts (`.pkl` files)
- A Streamlit web app in `app.py` for interactive prediction

## Dataset Details

The project uses two CSV files:

- `Fake.csv` -> fake news articles
- `True.csv` -> real news articles

Columns in raw data:

- `title`
- `text`
- `subject`
- `date`

Label encoding used in training:

- Fake news -> `class = 0`
- True news -> `class = 1`

Original dataset sizes from notebook:

- Fake: 23,481 rows
- True: 21,417 rows

For manual testing samples, last 10 rows were set aside from each class before training:

- Training pool after removal:
	- Fake: 23,471
	- True: 21,407
	- Total: 44,878

## Preprocessing Pipeline

The text-cleaning function used in both notebook and app performs:

- Lowercasing
- URL removal
- HTML tag removal
- Punctuation removal
- Newline removal
- Removal of tokens containing digits

Data processing flow:

1. Merge fake and true dataframes
2. Shuffle merged dataset
3. Keep only `text` and `class` (drop `title`, `subject`, `date`)
4. Handle nulls and apply text cleaning
5. Split data with `train_test_split(test_size=0.25)`
6. Convert text to features using `TfidfVectorizer`

## Models Trained

The following classifiers were trained on TF-IDF vectors:

- Logistic Regression (`LogisticRegression()`)
- Decision Tree (`DecisionTreeClassifier()`)
- Gradient Boosting (`GradientBoostingClassifier(random_state=0)`)
- Random Forest (`RandomForestClassifier(random_state=0)`)

## Training Results

Test set size: 11,220 samples

Model scores recorded in notebook (`.score` on test set):

- Logistic Regression: `0.9868983957`
- Decision Tree: `0.9960784314`
- Gradient Boosting: `0.9954545455`
- Random Forest: `0.9843137255`

Classification report summaries (test set):

- Logistic Regression
	- Accuracy: `0.99`
	- Class 0 F1: `0.99`
	- Class 1 F1: `0.99`

- Decision Tree
	- Accuracy: `1.00`
	- Class 0 F1: `1.00`
	- Class 1 F1: `1.00`

- Gradient Boosting
	- Accuracy: `1.00`
	- Class 0 F1: `1.00`
	- Class 1 F1: `1.00`

- Random Forest
	- Accuracy: `0.98`
	- Class 0 F1: `0.99`
	- Class 1 F1: `0.98`

## Saved Artifacts

The notebook saves these files for inference:

- `vectorizer.pkl`
- `logistic_model.pkl`
- `decisiontree_model.pkl`
- `gradientboost_model.pkl`
- `randomforest_model.pkl`

## Project Files

- `app.py` - Streamlit app for prediction
- `news.ipynb` - end-to-end training notebook
- `Fake.csv` - fake news data
- `True.csv` - true news data
- `.pkl` files - trained models + TF-IDF vectorizer

## Installation

1. Create and activate a virtual environment (recommended)
2. Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

## Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## How App Prediction Works

1. User enters a news article
2. App applies same text cleaning as training
3. Text is transformed using saved `vectorizer.pkl`
4. All 4 models predict (`0` or `1`)
5. Final output is shown using vote percentage

## Retraining Workflow

To retrain from scratch:

1. Open `news.ipynb`
2. Run all cells in order
3. Confirm evaluation metrics
4. Re-save model artifacts using the final `joblib.dump(...)` cell
5. Run Streamlit app again to use updated models
