### 📄 Full `README.md` Content (copy/paste this)

````markdown
# 🕵️ Predict Credit Card Fraud with Logistic Regression

This project uses logistic regression to build a model that predicts fraudulent credit card transactions. You’ll work with a simulated dataset and walk step-by-step through loading data, engineering features, training the model, and making predictions.

---

## 📁 Dataset

- **transactions_modified.csv**: A simplified dataset of 1,000 simulated transactions.
- Each row includes:
  - `type`, `amount`, account balances before and after the transaction
  - A label `isFraud` (1 for fraud, 0 for not fraud)

---

## ⚙️ Requirements

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
````

---

## ✅ Step-by-Step Instructions

### 📦 Load the Data

#### **Step 1. Load and Explore the Data**

* Load `transactions_modified.csv` into a DataFrame called `transactions`.
* Use `.head()` to view the first few rows.
* Use `.info()` to inspect row count and data types.
* Print the total number of fraudulent transactions (`isFraud == 1`).

---

### 🧹 Clean the Data

#### **Step 2. Summarize the `amount` Column**

* Use `.describe()` to see the distribution of transaction amounts.
* Consider how outliers or skewness could affect modeling.

#### **Step 3. Create `isPayment` Column**

* Add a new column `isPayment`:

  * Set to `1` if `type` is `"PAYMENT"` or `"DEBIT"`
  * Otherwise set to `0`

#### **Step 4. Create `isMovement` Column**

* Add a new column `isMovement`:

  * Set to `1` if `type` is `"CASH_OUT"` or `"TRANSFER"`
  * Otherwise set to `0`

#### **Step 5. Create `accountDiff` Column**

* Create a new column `accountDiff`:

  * The absolute difference between `oldbalanceOrg` and `oldbalanceDest`:

  ```python
  transactions['accountDiff'] = abs(transactions['oldbalanceOrg'] - transactions['oldbalanceDest'])
  ```

---

### 🧪 Select and Split the Data

#### **Step 6. Define Features and Label**

* Create a variable `features` containing:

  * `amount`, `isPayment`, `isMovement`, `accountDiff`
* Create a variable `label` that references the `isFraud` column.

#### **Step 7. Split into Train/Test Sets**

* Use `train_test_split()` from `sklearn.model_selection`
* Use `test_size=0.3` to keep 30% for testing.

---

### ⚖️ Normalize the Data

#### **Step 8. Standardize the Features**

* Create a `StandardScaler` object.
* Use `.fit_transform()` on training features.
* Use `.transform()` on test features.

---

### 🤖 Create and Evaluate the Model

#### **Step 9. Train the Logistic Regression Model**

* Use `LogisticRegression()` from `sklearn.linear_model`.
* Fit it to the scaled training data using `.fit()`.

#### **Step 10. Evaluate Accuracy on Training Data**

* Use `.score()` on the training data.
* Print the result to see training accuracy.

#### **Step 11. Evaluate Accuracy on Test Data**

* Use `.score()` on the test data.
* Compare with training accuracy to assess overfitting or underfitting.

#### **Step 12. Print Feature Coefficients**

* Print `model.coef_` to view the importance of each feature.
* Which feature had the highest weight? Which was least impactful?

---

### 🔍 Predict New Transactions

#### **Step 13. Define New Transactions**

```python
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])
```

* Create your own `your_transaction` array with 4 float values:

  * `amount`, `isPayment`, `isMovement`, `accountDiff`

#### **Step 14. Combine into One Array**

* Combine all four transactions using:

```python
sample_transactions = np.array([transaction1, transaction2, transaction3, your_transaction])
```

#### **Step 15. Scale New Transaction Data**

* Use the same `StandardScaler` to transform `sample_transactions`.

```python
sample_transactions = scaler.transform(sample_transactions)
```

#### **Step 16. Make Predictions**

* Use `model.predict(sample_transactions)` to see which transactions are fraud.
* Optionally, use `model.predict_proba(sample_transactions)` to see the probability of each prediction.

---

### 🏁 Step 17. Reflect and Try the Full Dataset

* Try rerunning the project using the full dataset `transactions.csv`.
* You may notice that most rows are non-fraud.
* That’s called an **imbalanced class problem**, and it's a challenge in real-world fraud detection.
* You’ll explore this more in the **Logistic Regression II** module.

---

## 🙌 Project Summary

This beginner-friendly machine learning project introduces:

* Basic EDA and feature engineering
* Splitting and scaling data
* Training a logistic regression model
* Interpreting model accuracy and predictions

---

## 👤 Author

Created by Ben Castillo
[GitHub Repo](https://github.com/becastil/credit-card-fraud-prediction)


