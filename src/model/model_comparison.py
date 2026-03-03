import pandas as pd
import numpy as np
import os
import json
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml


def load_params(params_path: str) -> dict:
    """Load parameters from params.yaml"""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        raise Exception(f"Error loading parameters from {params_path}: {e}")


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file"""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")


def prepare_data(data: pd.DataFrame) -> tuple:
    """Split data into features(X) and target(y)"""
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a model and return metrics dictionary"""
    y_pred = model.predict(X_test)
    return {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall': round(recall_score(y_test, y_pred), 4),
        'f1_score': round(f1_score(y_test, y_pred), 4)
    }


def plot_model_comparison(results: dict, save_path: str) -> None:
    """Create a bar chart comparing all models across all metrics"""
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2196F3', '#FF9800', '#4CAF50']  # Blue, Orange, Green

    for i, model_name in enumerate(models):
        values = [results[model_name][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model_name, color=colors[i])
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison: Random Forest vs SVM vs Logistic Regression', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Model comparison chart saved to: {save_path}")


def main():
    try:
        # Paths
        params_path = "params.yaml"
        train_data_path = "./data/processed/train_processed_mean.csv"
        test_data_path = "./data/processed/test_processed_mean.csv"
        comparison_metrics_path = "reports/model_comparison.json"
        comparison_chart_path = "reports/figures/model_comparison.png"

        # Load params
        params = load_params(params_path)
        n_estimators = params["model_building"]["n_estimators"]

        # Load and prepare data
        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)

        X_train, y_train = prepare_data(train_data)
        X_test, y_test = prepare_data(test_data)

        # ============================================================
        # StandardScaler: Required for SVM and Logistic Regression
        # because they are distance-based algorithms and need features
        # on similar scales. Random Forest is tree-based and doesn't
        # need scaling, but we scale for fair comparison.
        # ============================================================
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ============================================================
        # MODEL 1: Random Forest Classifier
        # - Ensemble of decision trees
        # - Handles non-linear relationships well
        # - Robust to outliers and missing values
        # - Does NOT require feature scaling
        # ============================================================
        print("=" * 60)
        print("Training Model 1: Random Forest Classifier")
        print("=" * 60)
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_metrics = evaluate_model(rf_model, X_test, y_test)
        print(f"  Accuracy:  {rf_metrics['accuracy']}")
        print(f"  Precision: {rf_metrics['precision']}")
        print(f"  Recall:    {rf_metrics['recall']}")
        print(f"  F1-Score:  {rf_metrics['f1_score']}")

        # ============================================================
        # MODEL 2: Support Vector Machine (SVM)
        # - Finds optimal hyperplane to separate classes
        # - Uses RBF kernel for non-linear decision boundary
        # - REQUIRES feature scaling (distance-based)
        # - Good for smaller datasets
        # ============================================================
        print("\n" + "=" * 60)
        print("Training Model 2: Support Vector Machine (SVM)")
        print("=" * 60)
        svm_model = SVC(kernel='rbf', random_state=42)
        svm_model.fit(X_train_scaled, y_train)
        svm_metrics = evaluate_model(svm_model, X_test_scaled, y_test)
        print(f"  Accuracy:  {svm_metrics['accuracy']}")
        print(f"  Precision: {svm_metrics['precision']}")
        print(f"  Recall:    {svm_metrics['recall']}")
        print(f"  F1-Score:  {svm_metrics['f1_score']}")

        # ============================================================
        # MODEL 3: Logistic Regression
        # - Linear classifier using sigmoid function
        # - Fast to train, easy to interpret
        # - REQUIRES feature scaling (gradient-based)
        # - Good baseline model
        # ============================================================
        print("\n" + "=" * 60)
        print("Training Model 3: Logistic Regression")
        print("=" * 60)
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        lr_metrics = evaluate_model(lr_model, X_test_scaled, y_test)
        print(f"  Accuracy:  {lr_metrics['accuracy']}")
        print(f"  Precision: {lr_metrics['precision']}")
        print(f"  Recall:    {lr_metrics['recall']}")
        print(f"  F1-Score:  {lr_metrics['f1_score']}")

        # ============================================================
        # COMPARISON RESULTS
        # ============================================================
        results = {
            'Random Forest': rf_metrics,
            'SVM': svm_metrics,
            'Logistic Regression': lr_metrics
        }

        # Find the best model based on accuracy
        best_model_name = max(results, key=lambda k: results[k]['accuracy'])

        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 73)
        for model_name, metrics in results.items():
            marker = " ⭐ BEST" if model_name == best_model_name else ""
            print(f"{model_name:<25} {metrics['accuracy']:<12} {metrics['precision']:<12} {metrics['recall']:<12} {metrics['f1_score']:<12}{marker}")

        print(f"\n✅ CONCLUSION: {best_model_name} shows the BEST overall performance!")
        print(f"   Therefore, we use {best_model_name} as our final model.\n")

        # Save comparison metrics
        os.makedirs(os.path.dirname(comparison_metrics_path), exist_ok=True)
        with open(comparison_metrics_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Comparison metrics saved to: {comparison_metrics_path}")

        # Save comparison chart
        os.makedirs(os.path.dirname(comparison_chart_path), exist_ok=True)
        plot_model_comparison(results, comparison_chart_path)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
