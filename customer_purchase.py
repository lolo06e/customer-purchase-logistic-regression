"""
TASK 4 - Customer Purchase (Logistic Regression)
Predict whether a customer will BUY (1) or NOT BUY (0) a product
based on age, income, product price, and previous purchases.

Run in VS Code:  python3 customer_purchase.py
Needs: numpy, pandas, scikit-learn, matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "customer_purchase.csv")

# ---------------------------------------------------------------
# 1. PROBLEM STATEMENT
# ---------------------------------------------------------------
print("=" * 60)
print("1. PROBLEM STATEMENT")
print("=" * 60)
print("Predict whether a customer will Buy or Not Buy a product,")
print("using the customer's age, income, the product price and")
print("how many times they have bought from the shop before.\n")

# ---------------------------------------------------------------
# Dataset: the task gives no data, so we build a realistic one
# (500 customers). Fixed seed -> same data every run.
# Rule behind it: higher income and more previous purchases make
# buying more likely; a higher price makes it less likely.
# Random noise is added so the model can't be 100% perfect.
# ---------------------------------------------------------------
if not os.path.exists(CSV):
    rng = np.random.default_rng(42)
    n = 500
    age      = rng.integers(18, 66, n)                          # years
    income   = rng.normal(3500, 1200, n).clip(800, 8000).round()  # USD / month
    price    = rng.uniform(20, 800, n).round(2)                  # USD
    previous = rng.poisson(3, n)                                 # past purchases
    score = (0.0012 * (income - 3500)
             - 0.0060 * (price - 400)
             + 0.45 * (previous - 3)
             - 0.015 * (age - 40)
             + rng.normal(0, 1.0, n))
    purchased = (score > 0).astype(int)
    pd.DataFrame({"age": age, "income": income, "product_price": price,
                  "previous_purchases": previous, "purchased": purchased}).to_csv(CSV, index=False)

df = pd.read_csv(CSV)
print("First 5 rows of the dataset:")
print(df.head(), "\n")
print(f"Rows: {len(df)}   Buy: {df.purchased.sum()}   Not Buy: {(df.purchased == 0).sum()}\n")

# Picture of the data: price vs income, coloured by Buy / Not Buy
fig, ax = plt.subplots(figsize=(7, 5))
for cls, colour, name in [(0, "#d62728", "Not Buy"), (1, "#1f77b4", "Buy")]:
    part = df[df.purchased == cls]
    ax.scatter(part.income, part.product_price, s=18, alpha=0.7, c=colour, label=name)
ax.set_xlabel("Income (USD / month)")
ax.set_ylabel("Product price (USD)")
ax.set_title("Customers: income vs product price")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(HERE, "data_overview.png"), dpi=150)

# ---------------------------------------------------------------
# 2. FEATURES (X)   3. TARGET (y)
# ---------------------------------------------------------------
features = ["age", "income", "product_price", "previous_purchases"]
X = df[features]
y = df["purchased"]
print("=" * 60)
print("2. FEATURES (X):", features)
print("3. TARGET  (y): purchased -> 1 = Buy, 0 = Not Buy")
print("=" * 60, "\n")

# ---------------------------------------------------------------
# 4. SPLIT: 80% train / 20% test (stratify keeps the Buy/Not Buy ratio)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"4. SPLIT -> train: {len(X_train)} rows, test: {len(X_test)} rows\n")

# ---------------------------------------------------------------
# 5. CREATE AND TRAIN the Logistic Regression model
# StandardScaler first, because income (~thousands) and
# previous purchases (~0-10) are on very different scales.
# ---------------------------------------------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression()),
])
model.fit(X_train, y_train)
coefs = model.named_steps["logreg"].coef_[0]
print("5. MODEL TRAINED. Coefficients (on scaled features):")
for f, c in zip(features, coefs):
    print(f"   {f:<20} {c:+.3f}  ({'raises' if c > 0 else 'lowers'} chance of buying)")
print(f"   intercept            {model.named_steps['logreg'].intercept_[0]:+.3f}\n")

# Picture of the coefficients: which feature pushes towards Buy / Not Buy
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(features, coefs, color=["#1f77b4" if c > 0 else "#d62728" for c in coefs])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient (scaled features)  ->  + means more likely to Buy")
ax.set_title("What drives a purchase?")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "coefficients.png"), dpi=150)

# ---------------------------------------------------------------
# 6. PREDICT on the test set
# ---------------------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]          # P(Buy)

# ---------------------------------------------------------------
# 7. ACCURACY
# ---------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
print("=" * 60)
print(f"7. ACCURACY: {acc:.3f}  ({acc*100:.1f}% of test customers predicted correctly)")
print("=" * 60, "\n")

# ---------------------------------------------------------------
# 8. CONFUSION MATRIX
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("8. CONFUSION MATRIX")
print("                 Predicted Not Buy   Predicted Buy")
print(f"Actual Not Buy        {tn:>5}             {fp:>5}")
print(f"Actual Buy            {fn:>5}             {tp:>5}\n")
print(f"   TN={tn} (correct Not Buy)  FP={fp} (said Buy, didn't)")
print(f"   FN={fn} (missed a buyer)   TP={tp} (correct Buy)\n")
print(classification_report(y_test, y_pred, target_names=["Not Buy", "Buy"]))

disp = ConfusionMatrixDisplay(cm, display_labels=["Not Buy", "Buy"])
disp.plot(cmap="Blues")
plt.title(f"Customer Purchase - Logistic Regression (acc = {acc:.2f})")
plt.tight_layout()
png = os.path.join(HERE, "confusion_matrix.png")
plt.savefig(png, dpi=150)
print(f"Confusion matrix image saved -> {png}\n")

# ---------------------------------------------------------------
# 9. AT LEAST 5 PREDICTED EXAMPLES
# ---------------------------------------------------------------
label = {0: "Not Buy", 1: "Buy"}
examples = X_test.head(8).copy()
examples["P(Buy)"] = y_prob[:8].round(3)
examples["predicted"] = [label[v] for v in y_pred[:8]]
examples["actual"] = [label[v] for v in y_test.iloc[:8]]
examples["correct?"] = np.where(examples.predicted == examples.actual, "yes", "NO")
print("9. PREDICTED EXAMPLES (from the test set)")
print(examples.to_string(index=False), "\n")

# Bonus: two brand-new customers
new = pd.DataFrame({"age": [25, 50], "income": [6000, 2000],
                    "product_price": [150, 700], "previous_purchases": [6, 0]})
new["P(Buy)"] = model.predict_proba(new[features])[:, 1].round(3)
new["prediction"] = [label[v] for v in model.predict(new[features])]
print("New customers:")
print(new.to_string(index=False), "\n")

# ---------------------------------------------------------------
# 10. EXPLANATION
# ---------------------------------------------------------------
print("=" * 60)
print("10. EXPLANATION")
print("=" * 60)
print(f"The Logistic Regression model predicted Buy / Not Buy correctly for "
      f"{acc*100:.1f}% of the {len(y_test)} test customers.")
print("Product price has the strongest negative effect: expensive products are "
      "much less likely to be bought, while higher income and more previous "
      "purchases raise the chance of buying; older customers are slightly less likely to buy.")
wrong = y_pred != y_test.values
unsure = int(((y_prob > 0.2) & (y_prob < 0.8) & wrong).sum())
print(f"The confusion matrix shows {fp + fn} mistakes ({fp} false Buy, {fn} missed buyers); "
      f"{unsure} of them had a predicted probability between 0.2 and 0.8, i.e. borderline customers.")
print("Because the classes are balanced and errors are split fairly evenly, "
      "accuracy is a fair summary here, and the model is useful for deciding "
      "which customers to target with offers.")

if matplotlib.get_backend().lower() not in ("agg",):
    plt.show()
