# Food Price Prediction using Machine Learning

This project aims to predict future food prices based on historical data and various influencing factors like weather conditions, inflation rate, and supply/demand. The goal is to develop a machine learning model that can generate synthetic data, preprocess and clean the data, explore key trends and correlations, and predict future food prices.

## Project Overview

The main objectives of this project are:
- **Generate Synthetic Data**: Create a synthetic dataset with food price and related factors.
- **Data Preprocessing and Cleaning**: Handle missing values, encode categorical features, and scale numerical data.
- **Exploratory Data Analysis (EDA)**: Perform in-depth analysis of trends, correlations, outliers, and distributions of features.
- **Model Training and Prediction**: Train machine learning models (e.g., Linear Regression, Random Forest) to predict future food prices.

## Features of the Project

1. **Synthetic Data Generation**: The synthetic dataset includes columns like date, region, food item, historical price, rainfall, temperature, inflation rate, fuel cost, supply, and demand.
2. **Data Preparation**: Cleaning, handling missing values, encoding categorical data, and scaling features.
3. **Exploratory Data Analysis (EDA)**: Analysis of correlations between features and their impact on historical prices.
4. **Modeling**: Regression models are used to predict future prices and evaluate model performance using metrics like MAE, RMSE, and R².

## Project Structure

```
food-price-prediction/
│
├── data/                    # Directory to store datasets
│   ├── synthetic_food_data.csv  # Synthetic data generated
│   └── processed_food_data.csv  # Processed and cleaned data
│
├── scripts/                 # Python scripts
│   ├── generate_data.py         # Script for synthetic data generation
│   ├── data_preparation.py      # Script for preprocessing and cleaning data
│   ├── eda.py                   # Script for exploratory data analysis
│   └── model_training.py        # Script for training and predicting prices
│
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Requirements

To run the project, you'll need to install the required libraries. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

The required dependencies are:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## How to Use

### 1. **Generate Synthetic Data**

Run the `generate_data.py` script to generate the synthetic dataset. The data will be saved as `synthetic_food_data.csv` in the `data/` folder.

```bash
python scripts/generate_data.py
```

### 2. **Data Preprocessing**

Once the synthetic data is generated, preprocess and clean it by running the `data_preparation.py` script.

```bash
python scripts/data_preparation.py
```

This will save the processed data as `processed_food_data.csv` in the `data/` folder.

### 3. **Exploratory Data Analysis (EDA)**

Run the Jupyter notebook `eda.ipynb` for visualizations and insights about the data. The notebook performs various analyses, including:

- Historical price trends
- Correlation heatmaps
- Outlier analysis
- Supply vs demand scatterplots

```bash
jupyter notebook scripts/eda.ipynb
```

### 4. **Train and Evaluate the Model**

Finally, use the `model_training.py` script to train a machine learning model and predict future food prices.

```bash
python scripts/model_training.py
```

This will evaluate the model performance and provide predicted prices.

## Insights and Results

The project provides insights into:
- The influence of factors like rainfall, temperature, inflation, and supply/demand on food prices.
- How to predict future food prices using historical data.
- Model evaluation metrics (e.g., MAE, RMSE, R²) to assess prediction accuracy.

