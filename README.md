# Machine Learning Practical Labs

A collaborative university coursework repository for the **Machine Learning** practical track. This project brings together a series of applied lab exercises focused on real-world data preparation, exploratory analysis, feature engineering, and regression modeling. It reflects the full introductory machine learning workflow, from cleaning raw datasets to evaluating predictive performance with standard metrics.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![Labs](https://img.shields.io/badge/labs-1%20%E2%80%93%203-success)
![License](https://img.shields.io/badge/use-coursework-lightgrey)

---

## Overview

This repository is maintained by a five-member team as part of the Machine Learning practical course. Each lab applies the concepts covered in the lectures — the learning paradigm, preprocessing, feature scaling, and the supervised learning workflow — to real-world datasets.

The work is organized by lab, with each notebook documenting its own problem statement, dataset, methodology, and evaluation.

> **Progression at a glance** — the labs build on each other, moving from raw cleaning to increasingly expressive regression models:

```mermaid
flowchart LR
    A["Lab 01<br/>Data Preprocessing<br/><i>Titanic</i>"] --> B["Lab 02<br/>Linear Regression<br/><i>Housing · Bike</i>"]
    B --> C["Lab 03<br/>Multiple &amp; Polynomial<br/>Regression<br/><i>Housing · Insurance</i>"]
```

## Course Information

| | |
|---|---|
| **Course** | Machine Learning |
| **Instructor** | Dr. Marwa Mamdouh |
| **Program** | Third Year — First Semester |
| **Credit Hours** | 3 |
| **Type** | University practical (labs) |

## Repository Structure

```
ML-Project/
├── Lab01/
│   ├── Lab 1.ipynb                              # Data preprocessing on the Titanic dataset
│   └── Titanic.csv
├── Lab02/
│   ├── lab2-HOUSEPRICE_LinearRegression.ipynb   # King County house price model
│   ├── kc_house_data.csv
│   └── team/
│       ├── pre_processing.ipynb                 # Motorcycle sales preprocessing
│       ├── USA_Housing_LinearRegression.ipynb   # USA housing price model
│       ├── USA_Housing.csv
│       └── BIKE DETAILS.csv
├── Lab03/
│   ├── HOUSEPRICE_multiple_Regeression.ipynb    # Multiple linear regression on King County
│   ├── Polynomial_Regression2.ipynb             # Polynomial regression on medical insurance
│   ├── kc_house_data.csv
│   └── insurance.csv
├── requirements.txt
└── README.md
```

## Labs

### Lab 01 — Data Preprocessing (Titanic)

Foundations of cleaning a raw dataset before any model is fitted:

- Handling missing values (median imputation, mode imputation, column removal)
- Encoding categorical variables with one-hot encoding (`get_dummies`)
- Splitting data into training and testing sets
- Standardizing numeric features with `StandardScaler`

### Lab 02 — Linear Regression

Applied regression modeling across three datasets:

| Notebook | Dataset | Task |
|---|---|---|
| `lab2-HOUSEPRICE_LinearRegression.ipynb` | King County house sales | Predict house `price` from features such as living area and grade |
| `team/USA_Housing_LinearRegression.ipynb` | USA Housing | Predict `Price` from average area income and related features |
| `team/pre_processing.ipynb` | Motorcycle sales | Clean and explore the dataset ahead of sales prediction |

Common workflow across the lab:

1. **Data loading & inspection** — `shape`, `head`, `info`, `describe`
2. **Cleaning** — dropping unnecessary columns, checking nulls
3. **Exploratory Data Analysis** — distributions, boxplots, scatter plots, correlation heatmaps
4. **Feature engineering** — outlier removal via IQR, extracting month/year from dates
5. **Train/test split** — held-out evaluation sets
6. **Modeling** — `LinearRegression` from scikit-learn
7. **Evaluation** — MAE, MSE, RMSE, and R² (`score`)

### Lab 03 — Multiple & Polynomial Regression

Extends the regression work with richer feature handling and non-linear models.

#### `HOUSEPRICE_multiple_Regeression.ipynb` — Multiple Linear Regression

Predicts King County house `price` from **19 features** (bedrooms, bathrooms, square footage, grade, location, etc.) on `kc_house_data.csv` (21,613 rows × 21 columns).

- Drops the `date` column, then splits 70/30 (`random_state=101`)
- Fits `LinearRegression` and inspects the intercept and per-feature coefficients
- Compares **Actual vs. Predicted** prices side by side

| Metric | Value |
|---|---|
| R² (train) | 0.70 |
| R² (test) | 0.71 |
| MAE | 126,051.39 |
| MSE | 41,054,725,150.16 |

#### `Polynomial_Regression2.ipynb` — Polynomial Regression

Predicts medical `charges` from the insurance dataset (`insurance.csv`, 1,338 rows) using a non-linear model.

- Encodes categorical columns (`sex`, `smoker`, `region`) to numeric
- Splits 75/25 (`random_state=1`) and scales with `StandardScaler`
- Expands features with `PolynomialFeatures(degree=3)`, then fits `LinearRegression`

| Metric | Value |
|---|---|
| R² (train) | 0.856 |
| R² (test) | 0.836 |
| MAE (train) | 2,765.40 |
| MAE (test) | 2,767.14 |

#### Lab 03 workflow

1. **Data loading** — read the CSV and inspect structure
2. **Encoding** — map categorical values to numeric codes
3. **Split & scale** — train/test split followed by `StandardScaler`
4. **Feature expansion** — polynomial features to capture non-linearity
5. **Modeling** — `LinearRegression` over the transformed feature space
6. **Evaluation** — MAE and R² for both train and test sets

## Models & Metrics

| Model | Lab | Dataset (rows) | Test R² | Test MAE |
|---|---|---|---|---|
| Single-feature Linear Regression | 02 | King County | — | — |
| Multiple Linear Regression | 03 | King County (21,613) | 0.71 | 126,051.39 |
| Polynomial Regression (deg. 3) | 03 | Insurance (1,338) | 0.84 | 2,767.14 |

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas / NumPy | Data manipulation and numerics |
| Matplotlib / seaborn | Visualization and EDA |
| scikit-learn | Preprocessing, splitting, and models |
| Jupyter Notebook | Experimentation and reporting |

Dependencies are pinned in [`requirements.txt`](requirements.txt).

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/asaerxd6767/ML-Project.git
cd ML-Project

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

Then open any notebook under `Lab01/`, `Lab02/`, or `Lab03/`. Datasets are stored alongside their notebooks, so run cells from the notebook's own directory (or update the file paths) to load the CSVs.

## Evaluation

Models across Labs 02 and 03 are assessed with standard regression metrics:

- **MAE** — mean absolute error
- **MSE / RMSE** — mean squared error and its square root
- **R² score** — proportion of variance explained, reported for both train and test sets

---

## Collaborators

This project was designed, implemented, and reviewed by a team of six contributors. The contributions below reflect shared ownership across the different lab workstreams rather than individual ranking.

| Member | GitHub | Focus |
|---|---|---|
| **Asser Mohammed** | [@asaerxd6767](https://github.com/asaerxd6767) | Repository setup, Lab 02 house-price notebooks, Lab 03 modeling, version control |
| **Habiba Mahmoud** | [@habibamahmoud13](https://github.com/habibamahmoud13) | Dataset sourcing and preprocessing |
| **Mostafa Aiman** | [@mostafaelwazzany](https://github.com/mostafaelwazzany) | Exploratory data analysis and visualization |
| **Sara Ahmed** | [@Sarsoour](https://github.com/Sarsoour) | Lab 03 modeling and preprocessing |
| **Akaber Ahmed** | — | Data cleaning and documentation |
| **Steven Mounir** | — | Model evaluation and reporting |


---

### Commit History

The project’s commit history is summarized below by contributor, linked GitHub profile, and total number of commits recorded in the repository.

| Contributor | GitHub Profile | Number of Commits |
|---|---|---:|
| <img src="https://github.com/mostafaelwazzany.png" width="80" alt="Mostafa Aiman's GitHub avatar" /> | [@mostafaelwazzany](https://github.com/mostafaelwazzany) | ![Mostafa's commits](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github.com%2Fsearch%2Fcommits%3Fq%3Drepo%253Aasaerxd6767%252FML-Project%2Bauthor%253Amostafaelwazzany&query=%24.total_count&label=commits) |
| <img src="https://github.com/habibamahmoud13.png" width="80" alt="Habiba Mahmoud's GitHub avatar" /> | [@habibamahmoud13](https://github.com/habibamahmoud13) | ![Habiba's commits](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github.com%2Fsearch%2Fcommits%3Fq%3Drepo%253Aasaerxd6767%252FML-Project%2Bauthor%253Ahabibamahmoud13&query=%24.total_count&label=commits) |
| <img src="https://github.com/asaerxd6767.png" width="80" alt="Asser Mohammed's GitHub avatar" /> | [@asaerxd6767](https://github.com/asaerxd6767) | ![Asser's commits](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github.com%2Fsearch%2Fcommits%3Fq%3Drepo%253Aasaerxd6767%252FML-Project%2Bauthor%253Aasaerxd6767&query=%24.total_count&label=commits) |
| <img src="https://github.com/Sarsoour.png" width="80" alt="Sara Ahmed's GitHub avatar" /> | [@Sarsoour](https://github.com/Sarsoour) | ![Sara's commits](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github.com%2Fsearch%2Fcommits%3Fq%3Drepo%253Aasaerxd6767%252FML-Project%2Bauthor%253ASarsoour&query=%24.total_count&label=commits) |
| **Akaber Ahmed** | — | No linked GitHub profile to attribute commits |
| **Steven Mounir** | — | No linked GitHub profile to attribute commits |

*Coursework repository for the Machine Learning practical track.*
