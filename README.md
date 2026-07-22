# Ames Housing Price Prediction

## Project Overview

This project develops machine learning models to predict residential sale prices using the Ames Housing dataset. The dataset contains 1,460 home sales and approximately 80 variables describing property quality, size, condition, location, and amenities.

The project compares two regression approaches:

- **LASSO Regression**, which provides regularization and interpretable feature coefficients
- **XGBoost Regression**, which captures nonlinear relationships and interactions between housing characteristics

The goal was to compare model performance while identifying the property features most strongly associated with sale price.

## Results

| Model | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|
| LASSO Regression | 0.148 | 0.098 | 0.871 |
| XGBoost Regression | 0.131 | 0.087 | 0.898 |

XGBoost achieved the strongest predictive performance with a test R² of approximately **0.90**. LASSO also performed well and offered greater interpretability through its feature coefficients.

The difference between XGBoost's training and testing performance suggests some overfitting, but the model still generalized effectively to unseen data.

## Key Findings

- Overall material and finish quality was one of the strongest predictors of sale price.
- Above-ground living area was strongly associated with higher home values.
- Garage capacity and garage area contributed meaningful predictive information.
- Newer homes and recently remodeled homes generally had higher predicted sale prices.
- Neighborhood and location-related variables helped explain differences between otherwise similar properties.
- XGBoost provided better predictive accuracy, while LASSO provided a more interpretable baseline.

These relationships should be interpreted as predictive associations rather than evidence that any individual feature directly causes a higher sale price.

## Project Workflow

### 1. Data Exploration

The dataset was reviewed for:

- Variable types
- Missing values
- Sale price distribution
- Correlations between numerical variables
- Relationships between important housing features and sale price

Because sale prices were right-skewed, the target variable was log-transformed before model training.

### 2. Data Preparation

The preprocessing workflow included:

- Handling missing values
- Encoding ordinal housing features
- Converting categorical variables into model-ready values
- Separating predictors from the target variable
- Creating training and testing datasets
- Standardizing predictors for LASSO regression

### 3. LASSO Regression

LASSO regression was used as a regularized linear baseline. The model reduces overfitting by shrinking less useful coefficients toward zero and can perform feature selection when coefficients are reduced entirely to zero.

LASSO achieved a test R² of approximately **0.87**.

### 4. XGBoost Regression

XGBoost was used to model nonlinear relationships and interactions between housing characteristics. The model combines many sequential decision trees, with each tree attempting to correct errors made by earlier trees.

XGBoost achieved the strongest test performance with a test R² of approximately **0.90**.

### 5. Model Evaluation

Model performance was evaluated using:

- Root Mean Squared Error
- Mean Absolute Error
- Coefficient of Determination
- Predicted versus actual sale price plots
- Residual analysis
- Feature importance

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## Repository Structure

```text
ames-housing-price-prediction/
├── README.md
├── Housing_Prices.ipynb
├── requirements.txt
├── data/
│   └── README.md
├── images/
└── .gitignore
```

## Running the Project

1. Clone the repository.

```bash
git clone https://github.com/your-username/ames-housing-price-prediction.git
cd ames-housing-price-prediction
```

2. Install the required packages.

```bash
pip install -r requirements.txt
```

3. Add the Ames Housing dataset to the `data` folder.

```text
data/train.csv
```

4. Launch the notebook.

```bash
jupyter notebook Housing_Prices.ipynb
```

## Dataset

This project uses the Ames Housing dataset, which includes detailed information about residential properties in Ames, Iowa.

The data includes variables related to:

- Property size
- Construction quality
- Neighborhood
- Basement characteristics
- Garage features
- Bathrooms and bedrooms
- Year built and remodeled
- Exterior and interior condition
- Final sale price

Dataset download instructions can be added to the `data/README.md` file.

## Limitations and Future Improvements

Potential improvements include:

- Building a complete preprocessing pipeline with `ColumnTransformer`
- Using cross-validation for all model comparisons
- Performing more extensive XGBoost hyperparameter tuning
- Comparing additional models such as Random Forest and Elastic Net
- Evaluating performance using repeated cross-validation
- Applying SHAP values for more detailed model interpretation
- Deploying the final model through a web application or dashboard

## Author

Russell Wagner  
B.S. Statistics, Minor in Mathematics  
Texas A&M University
