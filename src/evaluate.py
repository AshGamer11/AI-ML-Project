"""
Evaluation & Academic Visualizations Generator.
Creates publication-quality figures for the project report and presentation:
1. Multi-metric benchmark bar comparison (R², RMSE, MAE).
2. Time-series test overlay (Ground Truth vs. Predictions).
3. Parity regression scatter plots with 1:1 identity lines.
4. Residual distribution and Q-Q normality plots.
5. CPCB NAQI categorical confusion matrix.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.utils import categorize_aqi

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "report", "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "report", "figures")

def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_evaluation_visualizations():
    ensure_dirs()
    
    preds_csv = os.path.join(RESULTS_DIR, "test_predictions.csv")
    metrics_json = os.path.join(RESULTS_DIR, "benchmark_metrics.json")
    
    if not os.path.exists(preds_csv) or not os.path.exists(metrics_json):
        print("Predictions or metrics file not found. Please run src/train.py first.")
        return

    df_preds = pd.read_csv(preds_csv)
    with open(metrics_json, "r") as f:
        metrics = json.load(f)

    # Set publication plot style
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

    # -------------------------------------------------------------
    # Figure 1: Model Comparison Bar Charts (R2, RMSE, MAE)
    # -------------------------------------------------------------
    df_metrics = pd.DataFrame(metrics).T.reset_index()
    df_metrics.columns = ["Model", "R2", "RMSE", "MAE", "Index_Agreement", "NMB_Percent"]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = sns.color_palette("viridis", len(df_metrics))
    
    # R2 plot
    sns.barplot(data=df_metrics, x="Model", y="R2", ax=axes[0], palette=palette)
    axes[0].set_title("Coefficient of Determination ($R^2$) - Higher is Better", fontweight="bold")
    axes[0].set_ylim(0, 1.0)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold')

    # RMSE plot
    sns.barplot(data=df_metrics, x="Model", y="RMSE", ax=axes[1], palette=palette)
    axes[1].set_title("Root Mean Squared Error (RMSE µg/m³) - Lower is Better", fontweight="bold")
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold')

    # MAE plot
    sns.barplot(data=df_metrics, x="Model", y="MAE", ax=axes[2], palette=palette)
    axes[2].set_title("Mean Absolute Error (MAE µg/m³) - Lower is Better", fontweight="bold")
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold')

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontweight="semibold")
        ax.set_xlabel("")
        
    fig_path1 = os.path.join(FIGURES_DIR, "model_comparison_bars.png")
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"Generated: {fig_path1}")

    # -------------------------------------------------------------
    # Figure 2: Time Series Overlay (Ground Truth vs Model Predictions)
    # -------------------------------------------------------------
    sample_window = min(350, len(df_preds))
    df_window = df_preds.iloc[:sample_window].copy()
    
    plt.figure(figsize=(15, 6))
    plt.plot(df_window["actual"].values, label="Actual Ground Truth PM2.5", color="#1f77b4", linewidth=2.5, alpha=0.9)
    if "CNN_BiLSTM_Hybrid" in df_window.columns:
        plt.plot(df_window["CNN_BiLSTM_Hybrid"].values, label="Proposed CNN-BiLSTM Hybrid", color="#d62728", linestyle="--", linewidth=2.0)
    if "XGBoost" in df_window.columns:
        plt.plot(df_window["XGBoost"].values, label="XGBoost Baseline", color="#2ca02c", linestyle=":", linewidth=1.8, alpha=0.8)

    plt.axhline(60, color="orange", linestyle="--", alpha=0.7, label="NAAQS 24-hr Standard (60 µg/m³)")
    plt.axhline(15, color="green", linestyle="--", alpha=0.7, label="WHO 24-hr Guideline (15 µg/m³)")

    plt.title("Air Quality Time Series Tracking: Actual vs Predicted PM2.5 on Held-Out Test Set", fontsize=14, fontweight="bold")
    plt.xlabel("Hourly Timesteps (Test Evaluation Horizon)", fontsize=12)
    plt.ylabel("PM2.5 Concentration (µg/m³)", fontsize=12)
    plt.legend(loc="upper right", frameon=True, framealpha=0.9)
    
    fig_path2 = os.path.join(FIGURES_DIR, "actual_vs_predicted_timeseries.png")
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Generated: {fig_path2}")

    # -------------------------------------------------------------
    # Figure 3: Parity Scatter Plot (Actual vs Predicted)
    # -------------------------------------------------------------
    best_model_col = "CNN_BiLSTM_Hybrid" if "CNN_BiLSTM_Hybrid" in df_preds.columns else "XGBoost"
    y_true = df_preds["actual"].values
    y_pred = df_preds[best_model_col].values
    
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.35, color="#2b5c8f", edgecolors="none", s=25)
    
    # 1:1 Identity Line
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.05
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=2, label="1:1 Perfect Agreement Line")
    
    # Linear Regression best fit line
    m, b = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, m*y_true + b, color="navy", linestyle="-", linewidth=1.5, label=f"Best Fit: y = {m:.2f}x + {b:.1f}")
    
    r2_val = metrics[best_model_col]["r2"]
    rmse_val = metrics[best_model_col]["rmse"]
    plt.text(0.05, 0.90, f"Model: {best_model_col}\n$R^2$ = {r2_val}\nRMSE = {rmse_val} µg/m³",
             transform=plt.gca().transAxes, fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.title(f"Parity Plot: Predicted vs Actual PM2.5 ({best_model_col})", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Ground Truth PM2.5 (µg/m³)", fontsize=11)
    plt.ylabel("Model Predicted PM2.5 (µg/m³)", fontsize=11)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.legend(loc="lower right")
    
    fig_path3 = os.path.join(FIGURES_DIR, "parity_scatter_plot.png")
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"Generated: {fig_path3}")

    # -------------------------------------------------------------
    # Figure 4: Residual Error Distribution
    # -------------------------------------------------------------
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual Histogram + KDE
    sns.histplot(residuals, kde=True, ax=axes[0], color="#4a708b", bins=35)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Forecast Error Residual Distribution", fontweight="bold")
    axes[0].set_xlabel("Residual Error (Predicted - Actual µg/m³)")
    axes[0].set_ylabel("Frequency")
    
    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.3, color="#698b69", s=20)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs Predicted PM2.5 (Homoscedasticity Check)", fontweight="bold")
    axes[1].set_xlabel("Predicted PM2.5 (µg/m³)")
    axes[1].set_ylabel("Residual Error")
    
    fig_path4 = os.path.join(FIGURES_DIR, "residual_analysis.png")
    plt.savefig(fig_path4, dpi=300)
    plt.close()
    print(f"Generated: {fig_path4}")

    # -------------------------------------------------------------
    # Figure 5: CPCB NAQI Categorical Confusion Matrix
    # -------------------------------------------------------------
    cat_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    true_cats = [categorize_aqi(v)[0] for v in y_true]
    pred_cats = [categorize_aqi(v)[0] for v in y_pred]
    
    # Filter only categories present
    present_cats = [c for c in cat_order if (c in true_cats or c in pred_cats)]
    cm = confusion_matrix(true_cats, pred_cats, labels=present_cats)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=present_cats, yticklabels=present_cats)
    plt.title(f"CPCB Air Quality Category Classification Matrix ({best_model_col})", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Health Alert Category", fontsize=11)
    plt.ylabel("True Observation Category", fontsize=11)
    
    fig_path5 = os.path.join(FIGURES_DIR, "aqi_confusion_matrix.png")
    plt.savefig(fig_path5, dpi=300)
    plt.close()
    print(f"Generated: {fig_path5}")
    print("\nAll evaluation figures successfully rendered to report/figures/!")

if __name__ == "__main__":
    generate_evaluation_visualizations()

