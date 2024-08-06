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

    # Normalize features
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info("Data preprocessing completed.")
    return X_train, X_test, y_train, y_test


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
    """Plot feature importances."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure()
        plt.title("Feature Importances")
        plt.bar(range(len(feature_names)), importances[indices], align="center")
        plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], rotation=90)
        plt.xlim([-1, len(feature_names)])
        plt.show()


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


def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """Train and evaluate each model, then save the trained model."""
    results = []
    gradient_boosting_results = []
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append((name, mse, mae, r2))
        logger.info(f'{name} - MSE: {mse}, MAE: {mae}, R2: {r2}')
        joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.pkl')

        # Store predictions, actual values, and percentage differences for Gradient Boosting
        if name == 'Gradient Boosting':
            for pred, actual in zip(y_pred, y_test):
                percentage_diff = ((pred - actual) / actual) * 100 if actual != 0 else float('inf')
                gradient_boosting_results.append((pred, actual, percentage_diff))

    # Print predictions, actual values, and percentage differences for Gradient Boosting
    if gradient_boosting_results:
        print("\nGradient Boosting Predictions vs Actual Values:")
        for pred, actual, percentage_diff in gradient_boosting_results:
            print(f"Predicted: {pred}, Actual: {actual}, Percentage Difference: {percentage_diff:.2f}%")

        # Plot the results
        plot_gradient_boosting_results(gradient_boosting_results)

    return results


def main():
    # Define file path
    file_path = 'your_file_path'

    # Load and preprocess data
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Define models
    models = define_models()

    # Train and evaluate models
    results = train_and_evaluate(models, X_train, y_train, X_test, y_test)

    # Print final results
    for name, mse, mae, r2 in results:
        print(f'{name} - MSE: {mse}, MAE: {mae}, R2: {r2}')

    # Plot model performance
    plot_model_performance(results)


if __name__ == "__main__":
    main()
