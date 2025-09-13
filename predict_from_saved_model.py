# predict_from_saved_model.py

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def predict_from_saved(X_raw, model_path='ANN_BinaryVLE/ann_vle_model.h5', pipeline_path='ANN_BinaryVLE/preprocessing_pipeline.pkl'):
    model = keras.models.load_model(model_path, compile=False)
    preprocessing_pipeline = joblib.load(pipeline_path)
    scaler = preprocessing_pipeline['scaler']

    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(1, -1)

    X_scaled = scaler.transform(X_raw)
    y_pred = model.predict(X_scaled, verbose=0).flatten()
    return np.clip(y_pred, 0, 1)

def recreate_plots(model_path='ANN_BinaryVLE/ann_vle_model.h5', pipeline_path='ANN_BinaryVLE/preprocessing_pipeline.pkl', data_file='Data/vle_data.csv'):
    df = pd.read_csv(data_file)
    X = df[['x1', 'T', 'P']].values
    y = np.clip(np.asarray(df['y1']), 0, 1)  # pyright: ignore[reportCallIssue]

    model = keras.models.load_model(model_path, compile=False)
    preprocessing_pipeline = joblib.load(pipeline_path)
    scaler = preprocessing_pipeline['scaler']
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled, verbose=0).flatten()
    y_pred = np.clip(y_pred, 0, 1)

    # Parity plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y, y=y_pred, alpha=0.7, s=60)
    min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Experimental y1'); plt.ylabel('Predicted y1')
    plt.title('Parity Plot: Experimental vs Predicted y1 (Recreated)')
    plt.legend(['Perfect prediction']); plt.axis('equal')
    plt.tight_layout()
    plt.savefig('ANN_BinaryVLE/parity_plot_recreated.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Residuals plot
    residuals = y_pred - y
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=residuals, alpha=0.7, s=60)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Experimental y1'); plt.ylabel('Residuals')
    plt.title('Residuals Plot (Recreated)')
    plt.legend(['Zero residual'])
    plt.tight_layout()
    plt.savefig('ANN_BinaryVLE/residuals_plot_recreated.png', dpi=300, bbox_inches='tight')
    plt.close()

    # y1 vs x1 curve
    x1_grid = np.linspace(0, 1, 100)
    T_mean_original = np.mean(X[:, 1])
    P_original = X[0, 2]
    X_grid_original = np.column_stack([
        x1_grid,
        T_mean_original * np.ones_like(x1_grid),
        P_original * np.ones_like(x1_grid)
    ])
    X_grid_scaled = scaler.transform(X_grid_original)
    y_pred_grid = model.predict(X_grid_scaled, verbose=0).flatten()
    y_pred_grid = np.clip(y_pred_grid, 0, 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x1_grid, y_pred_grid, 'b-', linewidth=2, label='ANN Prediction')
    plt.plot(x1_grid, x1_grid, 'r--', linewidth=2, label='y1 = x1')
    plt.scatter(X[:, 0], y, color='red', s=60, alpha=0.7, label='Experimental')
    plt.xlabel('x1'); plt.ylabel('y1')
    plt.title(f'VLE Curve at T≈{T_mean_original:.1f}K, P≈{P_original:.3f}bar (Recreated)')
    plt.legend(); plt.grid(True, alpha=0.3); plt.xlim(0,1); plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig('ANN_BinaryVLE/y_vs_x_plot_recreated.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if not os.path.exists('ANN_BinaryVLE/ann_vle_model.h5') or not os.path.exists('ANN_BinaryVLE/preprocessing_pipeline.pkl'):
        raise FileNotFoundError("Model and/or pipeline not found. Run preprocessing_training_saving.py first.")
    recreate_plots()