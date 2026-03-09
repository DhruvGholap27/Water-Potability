import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file"""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")


def plot_correlation_heatmap(data: pd.DataFrame, save_path: str) -> None:
    """Plot correlation heatmap of all features with the target"""
    plt.figure(figsize=(12, 10))
    corr_matrix = data.corr()

    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                vmin=-1,
                vmax=1,
                cbar_kws={'label': 'Correlation Coefficient'})

    plt.title(
        'Correlation Heatmap: Relationships Between Water Quality Features',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Correlation heatmap saved to: {save_path}")


def plot_feature_distributions(data: pd.DataFrame, save_path: str) -> None:
    """Plot distribution of features by Potability class"""
    features = data.columns.drop('Potability')
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(
        'Feature Distributions by Potability Class',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    colors = ['#E53935', '#43A047']  # 0=Not Potable, 1=Potable
    labels = ['Not Potable (0)', 'Potable (1)']

    for idx, feature in enumerate(features):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        for cls in [0, 1]:
            subset = data[data['Potability'] == cls][feature].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=colors[cls],
                    label=labels[cls], edgecolor='white', linewidth=0.5)

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
    """Plot boxplots for features grouped by Potability"""
    features = data.columns.drop('Potability')
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(
        'Box Plots: Feature Distribution by Potability Class',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    plot_data = data.copy()
    plot_data['Potability'] = plot_data['Potability'].astype(str)
    palette = {'0': '#E53935', '1': '#43A047'}

    for idx, feature in enumerate(features):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        sns.boxplot(
            x='Potability', y=feature,
            data=plot_data.dropna(subset=[feature]),
            palette=palette,
            ax=ax
        )
        ax.set_title(feature, fontsize=12, fontweight='bold')
        ax.set_xticklabels(['Not Potable (0)', 'Potable (1)'], fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Box plots saved to: {save_path}")


def plot_missing_values(data: pd.DataFrame, save_path: str) -> None:
    """Visualize missing values in dataset"""
    null_counts = data.isnull().sum()
    null_percentages = (null_counts / len(data)) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#E53935' if p > 0 else '#43A047' for p in null_percentages]
    bars = ax.bar(null_counts.index, null_counts.values, color=colors, edgecolor='white')

    for bar, count, pct in zip(bars, null_counts.values, null_percentages.values):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 5,
                f'{count}\n({pct:.1f}%)',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Missing Value Count', fontsize=12)
    ax.set_title('Missing Values per Feature', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Missing values chart saved to: {save_path}")


def main():
    try:
        data_path = "src/data/water_potability.csv"
        figures_path = "reports/figures"
        os.makedirs(figures_path, exist_ok=True)

        data = load_data(data_path)
        print(f"Dataset shape: {data.shape}, features: {list(data.columns)}")

        plot_missing_values(data, os.path.join(figures_path, "missing_values.png"))
        plot_feature_distributions(data, os.path.join(figures_path, "feature_distributions.png"))
        plot_boxplots(data, os.path.join(figures_path, "boxplots.png"))
        plot_correlation_heatmap(data, os.path.join(figures_path, "correlation_heatmap.png"))

        print("All visualizations generated successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()