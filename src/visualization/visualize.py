import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file"""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")


def plot_correlation_heatmap(data: pd.DataFrame, save_path: str) -> None:
    """
    Plot correlation heatmap to find relationships between all columns.
    This helps us understand which features are positively or negatively
    correlated with each other and with the target (Potability).
    """
    plt.figure(figsize=(12, 10))
    corr_matrix = data.corr()

    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation Coefficient'})

    plt.title('Correlation Heatmap: Relationships Between Water Quality Features',
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Correlation heatmap saved to: {save_path}")

    # Print key correlations with Potability
    print("\n--- Correlation with Potability (Target) ---")
    potability_corr = corr_matrix['Potability'].drop('Potability').sort_values(ascending=False)
    for feature, corr in potability_corr.items():
        direction = "Positive" if corr > 0 else "Negative"
        strength = "Strong" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.1 else "Weak"
        print(f"  {feature:<20} : {corr:+.4f}  ({strength} {direction})")


def plot_feature_distributions(data: pd.DataFrame, save_path: str) -> None:
    """
    Plot distribution of each feature, colored by Potability class.
    This helps us see if potable and non-potable water have different
    distributions for each feature.
    """
    features = data.columns.drop('Potability')
    n_features = len(features)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('Feature Distributions by Potability Class',
                 fontsize=16, fontweight='bold', y=1.02)

    colors = ['#E53935', '#43A047']  # Red for Not Potable, Green for Potable
    labels = ['Not Potable (0)', 'Potable (1)']

    for idx, feature in enumerate(features):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        for potability_class in [0, 1]:
            subset = data[data['Potability'] == potability_class][feature].dropna()
            ax.hist(subset, bins=30, alpha=0.6,
                    color=colors[potability_class],
                    label=labels[potability_class],
                    edgecolor='white', linewidth=0.5)

        ax.set_title(feature, fontsize=12, fontweight='bold')
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature distribution plots saved to: {save_path}")


def plot_boxplots(data: pd.DataFrame, save_path: str) -> None:
    """
    Plot box plots for each feature grouped by Potability.
    This helps detect outliers and compare feature ranges between classes.
    """
    features = data.columns.drop('Potability')
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('Box Plots: Feature Distribution by Potability Class',
                 fontsize=16, fontweight='bold', y=1.02)

    plot_data = data.copy()
    plot_data['Potability'] = plot_data['Potability'].astype(str)
    palette = {'0': '#E53935', '1': '#43A047'}

    for idx, feature in enumerate(features):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        sns.boxplot(x='Potability', y=feature, data=plot_data.dropna(subset=[feature]),
                    palette=palette, ax=ax)

        ax.set_title(feature, fontsize=12, fontweight='bold')
        ax.set_xticklabels(['Not Potable (0)', 'Potable (1)'], fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Box plots saved to: {save_path}")


def plot_missing_values(data: pd.DataFrame, save_path: str) -> None:
    """
    Visualize missing values in the dataset.
    Shows which columns have missing data and how much.
    """
    null_counts = data.isnull().sum()
    null_percentages = (null_counts / len(data)) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#E53935' if p > 0 else '#43A047' for p in null_percentages]
    bars = ax.bar(null_counts.index, null_counts.values, color=colors, edgecolor='white')

    # Add value labels on bars
    for bar, count, pct in zip(bars, null_counts.values, null_percentages.values):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 5,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Missing Value Count', fontsize=12)
    ax.set_title('Missing Values per Feature', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Missing values chart saved to: {save_path}")

    # Print summary
    print("\n--- Missing Values Summary ---")
    for col in data.columns:
        null_count = data[col].isnull().sum()
        if null_count > 0:
            print(f"  {col:<20} : {null_count} missing ({null_count/len(data)*100:.1f}%)")
        else:
            print(f"  {col:<20} : No missing values ✓")


def plot_pairplot(data: pd.DataFrame, save_path: str) -> None:
    """
    Create pairplot for top features to visualize relationships.
    We select a subset of features to keep the plot readable.
    """
    # Select top features with highest absolute correlation to Potability
    corr = data.corr()['Potability'].drop('Potability').abs().sort_values(ascending=False)
    top_features = list(corr.head(4).index) + ['Potability']

    print(f"\nPairplot using top 4 correlated features: {top_features[:-1]}")

    plot_data = data[top_features].dropna().copy()
    plot_data['Potability'] = plot_data['Potability'].astype(str)

    pairplot = sns.pairplot(plot_data,
                            hue='Potability',
                            palette={'0': '#E53935', '1': '#43A047'},
                            plot_kws={'alpha': 0.5, 's': 20},
                            diag_kind='hist',
                            height=2.5)

    pairplot.figure.suptitle('Pairplot: Top Correlated Features with Potability',
                             fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pairplot saved to: {save_path}")


def plot_target_distribution(data: pd.DataFrame, save_path: str) -> None:
    """
    Plot the distribution of the target variable (Potability).
    Shows class imbalance — more non-potable than potable samples.
    """
    counts = data['Potability'].value_counts()
    labels = ['Not Potable (0)', 'Potable (1)']
    colors = ['#E53935', '#43A047']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars = ax1.bar(labels, counts.values, color=colors, edgecolor='white')
    for bar, count in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 10,
                 f'{count}', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Target Distribution (Count)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Pie chart
    ax2.pie(counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
    ax2.set_title('Target Distribution (%)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Target distribution plot saved to: {save_path}")


def main():
    try:
        # Path to raw dataset (before any processing)
        data_path = "src/data/water_potability.csv"
        figures_path = "reports/figures"

        # Create output directory
        os.makedirs(figures_path, exist_ok=True)

        # Load data
        data = load_data(data_path)

        print("=" * 60)
        print("WATER POTABILITY - DATA VISUALIZATION")
        print("=" * 60)
        print(f"\nDataset Shape: {data.shape}")
        print(f"Features: {list(data.columns)}")
        print(f"Total Samples: {len(data)}")

        # 1. Missing Values Visualization
        print("\n" + "=" * 60)
        print("1. Missing Values Analysis")
        print("=" * 60)
        plot_missing_values(data, os.path.join(figures_path, "missing_values.png"))

        # 2. Target Distribution
        print("\n" + "=" * 60)
        print("2. Target Variable Distribution")
        print("=" * 60)
        plot_target_distribution(data, os.path.join(figures_path, "target_distribution.png"))

        # 3. Correlation Heatmap
        print("\n" + "=" * 60)
        print("3. Correlation Heatmap")
        print("=" * 60)
        plot_correlation_heatmap(data, os.path.join(figures_path, "correlation_heatmap.png"))

        # 4. Feature Distributions by Class
        print("\n" + "=" * 60)
        print("4. Feature Distributions by Potability Class")
        print("=" * 60)
        plot_feature_distributions(data, os.path.join(figures_path, "feature_distributions.png"))

        # 5. Box Plots
        print("\n" + "=" * 60)
        print("5. Box Plots for Outlier Detection")
        print("=" * 60)
        plot_boxplots(data, os.path.join(figures_path, "boxplots.png"))

        # 6. Pairplot of top features
        print("\n" + "=" * 60)
        print("6. Pairplot of Top Correlated Features")
        print("=" * 60)
        plot_pairplot(data, os.path.join(figures_path, "pairplot.png"))

        print("\n" + "=" * 60)
        print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print(f"Check the '{figures_path}' folder for all plots.")
        print("=" * 60)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
