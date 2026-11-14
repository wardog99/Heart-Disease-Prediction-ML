# Heart Disease Prediction

This project explores a binary classification task: predicting the `target` label in a heart disease dataset from patient measurements. It uses exploratory data analysis and scikit-learn models, with a focus on K-nearest neighbors (KNN).

> **Disclaimer:** This is an educational machine-learning project, not a medical diagnostic tool.

## Project contents

- [`heart-disease-pred.ipynb`](./heart-disease-pred.ipynb) — data exploration, preprocessing, train/test splitting, and model evaluation.
- [`datasets/heart.csv`](./datasets/heart.csv) — dataset used by the notebook (303 rows and 14 columns, including the target).

The dataset includes fields such as age, sex, chest-pain type, resting blood pressure, cholesterol, maximum heart rate, and the binary `target` label.

## Workflow

The notebook:

1. Loads the CSV file and inspects its structure, summary statistics, and missing values.
2. Visualizes the data and converts categorical columns to dummy variables.
3. Standardizes selected numeric features.
4. Splits the features and target into training and test sets (80/20).
5. Evaluates KNN with 10-fold cross-validation while exploring different neighbor counts.

The notebook reports cross-validation scores for model selection. The held-out test set should be used for a final evaluation after selecting a model.

## Setup and run

From the project directory, install the required packages:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Start Jupyter and open the notebook:

```bash
jupyter notebook
```

Run the notebook cells from top to bottom. The CSV-loading cell currently uses a local absolute path; if your project is in a different location, update it to point to `datasets/heart.csv` (for example, `pd.read_csv("datasets/heart.csv")` when running from the project directory).
