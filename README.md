# Logistic Regression with Preprocessing

A Python/Jupyter notebook that builds an end-to-end **Logistic Regression** classification pipeline — from raw CSV to accuracy score — including preprocessing steps like label encoding and feature scaling.

## 📁 Files

| File | Description |
|---|---|
| `Model.ipynb` | Main Jupyter notebook |
| `Dataset.csv` | Input dataset (insurance response data) |
| `Prompt_.docx` | Original task requirements |

## 📊 Dataset

Insurance customer data with the following columns:

`id, Gender, Age, Driving_License, Region_Code, Previously_Insured, Vehicle_Age, Vehicle_Damage, Annual_Premium, Policy_Sales_Channel, Vintage, Response`

- **Target column:** `Response` (binary — 0 or 1)
- **Identifier column:** `id` (dropped before training)

## ⚙️ Requirements

- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

```bash
pip install pandas numpy matplotlib scikit-learn
```

## ▶️ How to Run

1. Keep `Model.ipynb` and `Dataset.csv` in the same folder.
2. Launch Jupyter:
   ```bash
   jupyter notebook Model.ipynb
   ```
3. Run all cells in order (**Run → Run All Cells**).
4. Enter the CSV filename when prompted (e.g. `Dataset.csv`).

## 🧠 Pipeline Overview

1. **Import Libraries** — pandas, numpy, matplotlib, scikit-learn.
2. **Get Filename from User** — prompts for the dataset's CSV file name.
3. **Load & Preview** — reads the CSV and displays the first 5 rows.
4. **Drop ID Column** — removes the identifier column (`id`), which holds no predictive value.
5. **Encode Categorical Variables** — applies `LabelEncoder` to all object-type (text) columns, e.g. `Gender`, `Vehicle_Age`, `Vehicle_Damage`.
6. **Feature/Target Split** — separates `X` (features) and `y` (`Response`).
7. **Standardize Features** — scales `X` using `StandardScaler` for stable model convergence.
8. **Train/Test Split** — 80% training, 20% testing (`random_state=1`).
9. **Train Model** — fits a `LogisticRegression` classifier.
10. **Predict** — generates predictions on the test set.
11. **Evaluate** — prints the **Accuracy Score**.

## ✅ Output

```
First 5 Rows:
   id  Gender  Age  ...  Vintage  Response
0   1    Male   44  ...      217         1
...

Model Accuracy: 0.87   (example — actual value depends on data/run)
```

## 📝 Notes

- Only columns with `object` dtype are label-encoded; numeric columns pass through unchanged.
- Feature scaling is applied to **all** features equally via `StandardScaler`.
- Accuracy is the sole reported metric, as specified in the task prompt.
