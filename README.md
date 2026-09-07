# Bearing Fault Classification

A simple, end-to-end notebook that reads your CSV of vibration statistics and trains a
machine learning model to predict the bearing fault type.

## What's in this project

```
bearing_fault_classification.ipynb   # the main notebook (EDA + ML)
requirements.txt                     # Python packages needed to run it
README.md                            # this file
```

## Expected data format

The notebook expects a CSV file with:

- Numeric feature columns, e.g.: `max`, `min`, `mean`, `sd`, `rms`, `skewness`,
  `kurtosis`, `crest`, `form`
- One label column called `fault` containing the class name, e.g. `Ball_007_1`,
  `Normal`, `IR_014_2`, etc.

If your column names differ slightly, just update the `TARGET_COLUMN` variable in the
second cell of the notebook (the "Setup" cell) to match your label column's name — the
rest of the numeric columns are picked up automatically.

## How to run it

1. **Install the requirements** (ideally inside a virtual environment):

   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Put your CSV file** in the same folder as the notebook. By default the notebook
   looks for a file named `data.csv` — either rename your file to `data.csv`, or open
   the notebook and change the `DATA_PATH` variable in the "Setup" cell to your file's
   name/path.

3. **Launch Jupyter**:

   ```bash
   jupyter notebook
   ```

   Then open `bearing_fault_classification.ipynb` and run all the cells (Cell → Run All).

## What the notebook does, step by step

1. **Load the data** — reads the CSV, checks its shape, data types, missing values, and
   removes duplicate rows.
2. **Exploratory Data Analysis (EDA)**
   - Bar chart of how many samples exist per fault class (checks for class imbalance).
   - Histograms of every numeric feature's distribution.
   - Boxplots showing how each feature varies across the different fault classes.
   - A correlation heatmap between all numeric features.
3. **Prepare the data** — encodes the text fault labels into numbers, splits the data
   into training (75%) and test (25%) sets (keeping the class balance the same in both,
   via stratification), and scales the features so all models can use them fairly.
4. **Try several models** — trains and compares:
   - Logistic Regression
   - K-Nearest Neighbors
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - Support Vector Machine (RBF kernel)

   Each is scored on the test set using accuracy and weighted F1-score, shown in a
   table and a bar chart.
5. **Pick the best model and validate it properly** — the model with the best F1-score
   is re-evaluated using **5-fold stratified cross-validation**, so the result isn't
   just a fluke of one particular train/test split. Cross-validation scores are shown
   as a boxplot.
6. **Final evaluation** — the best model is evaluated one last time on the untouched
   test set, producing:
   - A confusion matrix (which classes get mixed up with which)
   - A full classification report (precision, recall, F1 per class)
   - A feature importance chart (which measurements matter most for the prediction)
7. **Summary** — a short, auto-generated recap of the best model and its key numbers,
   printed at the end of the notebook.

## Notes

- The notebook uses only widely-used, well-documented libraries (`pandas`, `numpy`,
  `matplotlib`, `seaborn`, `scikit-learn`), so it's easy to read and modify.
- All settings you're likely to want to change (file path, target column name, test
  size, random seed, list of models) sit near the top of the notebook or in clearly
  labeled cells — no need to hunt through the code.
- If your dataset is very large, some models (especially SVM) may take longer to train
  — you can comment out models you don't need in the `models` dictionary.
