import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path):
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path, encoding='ansi')
        logger.info("Data loaded successfully.")
    except UnicodeDecodeError as e:
        logger.error(f"Error reading the CSV file: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)
    return df


def define_models():
    """Define a dictionary of models to evaluate."""
    models = {
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Support Vector Regression': SVR(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'Lasso': Lasso(random_state=42),
        'Ridge': Ridge(random_state=42),
        'ElasticNet': ElasticNet(random_state=42)
    }
    logger.info("Models defined.")
    return models


def tune_random_forest(X_train, y_train):
    """Tune hyperparameters for Random Forest using GridSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                               param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def preprocess_data(df):
    """Preprocess data: handle missing values, one-hot encode categorical features, and normalize."""
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    # Handle missing values: fill numeric columns with their mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Handle missing values in non-numeric columns if needed (e.g., mode imputation)
    for col in non_numeric_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Drop the target column and get features and labels
    X = df.drop('user_of_ibbwifi', axis=1)
    y = df['user_of_ibbwifi']

    # One-hot encode categorical features
    X = pd.get_dummies(X)

    # Get feature names before normalization
    feature_names = X.columns

    # Normalize features
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info("Data preprocessing completed.")
    return X_train, X_test, y_train, y_test, feature_names


def tune_svr(X_train, y_train):
    """Tune hyperparameters for SVR using GridSearchCV."""
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2, 0.5]
    }
    grid_search = GridSearchCV(estimator=SVR(),
                               param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def tune_gradient_boosting(X_train, y_train):
    """Tune hyperparameters for Gradient Boosting using GridSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                               param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def tune_xgboost(X_train, y_train):
    """Tune hyperparameters for XGBoost using GridSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    grid_search = GridSearchCV(estimator=XGBRegressor(random_state=42),
                               param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def plot_feature_importances(model, feature_names):
    """Plot the top 10 feature importances and save as PNG."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Select top 10 features
        plt.figure(figsize=(12, 8))  # Adjust the figure size
        plt.title("Top 10 Feature Importances")
        plt.bar(range(10), importances[indices], align="center")
        plt.xticks(range(10), np.array(feature_names)[indices], rotation=90, ha='right')  # Rotate labels
        plt.xlim([-1, 10])
        plt.tight_layout()  # Adjust layout to fit labels
        plt.savefig('top_10_feature_importances.png')  # Save the plot as a PNG file
        plt.close()


def plot_gradient_boosting_results(results):
    """Plot predictions vs actual values for Gradient Boosting and save as PNG."""
    preds, actuals, _ = zip(*results)

    plt.figure(figsize=(7, 7))

    # Plot predictions vs actual values
    plt.plot(actuals, preds, 'o', label='Predicted vs Actual')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', label='Ideal')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig('gradient_boosting_predictions_vs_actuals.png')
    plt.close()


def plot_model_performance(results):
    """Plot the R2 performance of different models and save as PNG."""
    # Extract model names and R2 values
    model_names = [result[0] for result in results]
    r2_values = [result[3] for result in results]

    # Find the model with the highest R2 value
    best_model_index = np.argmax(r2_values)
    best_model_name = model_names[best_model_index]
    best_r2_value = r2_values[best_model_index]

    # Set up the bar width and positions
    bar_width = 0.4
    index = np.arange(len(model_names))

    # Create the bar plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(index, r2_values, bar_width, label='R2')

    # Highlight the best model
    bars[best_model_index].set_color('r')

    # Add labels and title
    plt.xlabel('Models')
    plt.ylabel('R2 Score')
    plt.title('Model Performance Comparison (R2)')
    plt.xticks(index, model_names, rotation=45)
    plt.legend()

    # Annotate the best model
    plt.text(best_model_index, best_r2_value, f'{best_model_name}\nR2: {best_r2_value:.2f}',
             ha='center', va='bottom', color='red', fontweight='bold')

    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    plt.show()


def train_and_evaluate(models, X_train, y_train, X_test, y_test, feature_names):
    """Train and evaluate each model, then save the trained model."""
    results = []
    gradient_boosting_results = []
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        explained_var = explained_variance_score(y_test, preds)

        # Save the model
        joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.pkl')

        logger.info(f'{name} - MSE: {mse}, MAE: {mae}, R2: {r2}, Explained Variance: {explained_var}')
        results.append((name, mse, mae, r2))

        if name == 'Gradient Boosting':
            gradient_boosting_results = list(zip(preds, y_test, y_test.index))

        # Plot feature importances if applicable
        plot_feature_importances(model, feature_names)

    # Plot Gradient Boosting results
    if gradient_boosting_results:
        plot_gradient_boosting_results(gradient_boosting_results)

    return results


def sanitize_filename(filename):
    """Sanitize the filename by replacing invalid characters."""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)


def convert_percentage_strings(df):
    """Convert percentage strings to numeric values."""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if the column contains percentage strings
            if df[col].str.contains('%').any():
                df[col] = df[col].str.replace('%', '').astype(float) / 100
    return df


def plot_top_5_features(df, target_col):
    """Plot graphs for the top 5 features based on correlation with the target variable."""
    # Convert percentage strings to numeric values
    df = convert_percentage_strings(df)

    # Drop non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    # Compute the correlation matrix
    corr_matrix = df_numeric.corr()

    # Get the top 5 features correlated with the target variable
    top_5_features = corr_matrix[target_col].abs().sort_values(ascending=False).index[1:6]

    # Plot distributions and relationships with the target variable for the top 5 features
    for feature in top_5_features:
        plt.figure(figsize=(12, 6))

        # Plot distribution of the feature
        plt.subplot(1, 2, 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')

        # Plot relationship with the target variable
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=df[feature], y=df[target_col])
        plt.title(f'{feature} vs {target_col}')

        plt.tight_layout()
        plt.show()  # Display the plots instead of saving them

    logger.info("Top 5 feature analysis plots displayed.")


def eda(df, target_col='user_of_ibbwifi'):
    """Perform Exploratory Data Analysis (EDA) on the dataset."""
    # Convert percentage strings to numeric values
    df = convert_percentage_strings(df)

    # Summarize the dataset
    logger.info("Dataset Summary:")
    logger.info(df.describe())
    logger.info(df.info())

    # Plot distributions and relationships for the top 5 features
    plot_top_5_features(df, target_col)

    logger.info("EDA completed and plots displayed.")


def main():
    # Define file path
    file_path = 'your_file_path'

    # Load and preprocess data
    df = load_data(file_path)
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

    # Perform EDA
    eda(df, target_col='user_of_ibbwifi')

    # Define models
    models = define_models()

    # Train and evaluate models
    results = train_and_evaluate(models, X_train, y_train, X_test, y_test, feature_names)

    # Print final results
    for name, mse, mae, r2 in results:
        print(f'{name} - MSE: {mse}, MAE: {mae}, R2: {r2}')

    # Plot model performance
    plot_model_performance(results)


if __name__ == "__main__":
    main()
