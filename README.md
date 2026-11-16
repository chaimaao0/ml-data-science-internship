# ml-data-science-internship

Machine Learning & Data Science internship projects (Datheon, Standard Internship, 4 weeks).
Each task is a self-contained Jupyter notebook: data preparation, model training, evaluation and plots.

## Projects

| # | Task | Notebook | Models | Main metrics |
|---|------|----------|--------|--------------|
| 1 | Multi-class classification (Iris / Wine) | `iris_classification.ipynb` | Logistic Regression, Decision Tree | Accuracy, precision, recall, F1, confusion matrix |
| 2 | Housing price prediction | `housing_price_prediction.ipynb` | OLS, Ridge, Lasso (GridSearchCV) | RMSE, MAE, R² |
| 3 | Customer churn prediction | `churn_prediction.ipynb` | Random Forest, XGBoost | ROC-AUC, confusion matrix, feature importance |
| 4 | NLP sentiment classification | `NLP_Sentiment_Classification.ipynb` | Multinomial Naive Bayes, Logistic Regression | Accuracy, precision, recall, F1 |

## Installation

The notebooks run directly in Google Colab. Task 4 downloads the NLTK data (`stopwords`, `wordnet`) on first run.

## Architecture

All notebooks follow the same flow:

```
Load data -> Clean / preprocess -> Feature engineering -> Train models -> Evaluate -> Plots
```

### Task 4: NLP sentiment classification

```
Raw review -> clean_text() -> TfidfVectorizer(max_features=5000) -> MultinomialNB / LogisticRegression -> predict_sentiment()
```
<img width="527" height="635" alt="image" src="https://github.com/user-attachments/assets/8d70b1be-52bf-490a-b697-0806b7d856a4" />

- **Cleaning:** lowercase, HTML tag and punctuation removal, stopword removal (keeping *not / no / nor* so negation survives), lemmatization (NLTK).
- **Features:** TF-IDF matrix.
- **Models:** Multinomial Naive Bayes and Logistic Regression.
- **Evaluation:** 80/20 split, stratified cross-validation and leave-one-out.
- **Interactive helper:** `predict_sentiment("your review")` returns the prediction of both models with confidence. The notebook also has an optional input loop (`RUN_INTERACTIVE = True`).

**Limitation:** the provided starter dataset has only 10 reviews (9 after removing the neutral one), so accuracy scores are low and unstable. They show that the pipeline works, not that the models are good. A larger corpus (IMDB, Amazon reviews) would give much better results.

## Author

Omrani Chaima, Datheon Machine Learning & Data Science Internship
