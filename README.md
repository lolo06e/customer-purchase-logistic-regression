# Task 4 – Customer Purchase (Logistic Regression)

Predict whether a customer will **Buy (1)** or **Not Buy (0)** a product from
age, income, product price and previous purchases.

## Run in VS Code
1. Open the `machine-learning-class` folder in VS Code.
2. Pick your conda interpreter (bottom-right), install if needed:
   `pip install numpy pandas scikit-learn matplotlib`
3. Open `task4_customer_purchase/customer_purchase.py` and press ▶ (Run Python File).

## Files
| File | What it is |
|---|---|
| `customer_purchase.py` | Full solution – all 10 requirements, printed in order |
| `customer_purchase.csv` | 500-customer dataset (the task gives no data; generated with a fixed seed, so it's the same every run) |
| `EXPLANATION.md` / `EXPLANATION.pdf` | Presentation write-up: all 10 requirements answered, with code and pictures |
| `confusion_matrix.png`, `coefficients.png`, `data_overview.png` | Plots made by the script |
| `output.txt` | Full printed output of the script |

## Results
- Split: 400 train / 100 test (80/20, stratified)
- **Accuracy: 0.86**
- Confusion matrix: TN 45, FP 7, FN 7, TP 41
- Price and income matter most; previous purchases raise the chance of buying; age slightly lowers it.
