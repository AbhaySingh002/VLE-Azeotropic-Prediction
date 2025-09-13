# preprocessing_training_saving.py

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import zipfile

# Import from ANN.py
from ANN import build_model, train_model

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_data(csv_filepath='vle_data.csv'):
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"Data file '{csv_filepath}' not found. Please provide a valid CSV file.")
    df = pd.read_csv(csv_filepath)
    return df

def preprocess_data(df, scaler_type='standard'):
    X = df[['x1', 'T', 'P']].values
    y = df['y1'].values
    y = np.clip(y, 0, 1)

    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    preprocessing_pipeline = {
        'scaler': scaler,
        'scaler_type': scaler_type,
        'feature_names': ['x1', 'T', 'P'],
        'target_name': 'y1'
    }

    return {
        'X_scaled': X_scaled,
        'y': y,
        'preprocessing_pipeline': preprocessing_pipeline
    }

def split_data(X_scaled, y, test_size=0.15, val_size=0.15, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, shuffle=True
    )

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test
    }

def create_evaluation_plots(model, data_splits, preprocessing_pipeline, y_test_true, y_test_pred):
    # Parity plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test_true, y=y_test_pred, alpha=0.7, s=60)
    min_val, max_val = min(y_test_true.min(), y_test_pred.min()), max(y_test_true.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    plt.xlabel('Experimental y1')
    plt.ylabel('Predicted y1')
    plt.title('Parity Plot: Experimental vs Predicted y1')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('parity_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Residuals plot
    residuals = y_test_pred - y_test_true
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_true, y=residuals, alpha=0.7, s=60)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero residual')
    plt.xlabel('Experimental y1')
    plt.ylabel('Residuals (Predicted - Experimental)')
    plt.title('Residuals Plot')
    plt.legend()
    plt.tight_layout()
    plt.savefig('residuals_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # y1 vs x1 curve
    scaler = preprocessing_pipeline['scaler']
    x1_grid = np.linspace(0, 1, 100)
    X_train_original = scaler.inverse_transform(data_splits['X_train'])
    T_mean_original = np.mean(X_train_original[:, 1])
    P_original = X_train_original[0, 2]

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
    plt.plot(x1_grid, x1_grid, 'r--', linewidth=2, label='y1 = x1 (Azeotrope line)')
    X_test_original = scaler.inverse_transform(data_splits['X_test'])
    plt.scatter(X_test_original[:, 0], data_splits['y_test'],
                color='red', s=60, alpha=0.7, label='Experimental data', zorder=5)
    plt.xlabel('x1 (Liquid mole fraction)')
    plt.ylabel('y1 (Vapor mole fraction)')
    plt.title(f'VLE Curve at T≈{T_mean_original:.1f}K, P≈{P_original:.3f}bar')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('y_vs_x_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE Loss'); ax1.set_title('Training and Validation Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Mean Absolute Error'); ax2.set_title('Training and Validation MAE')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, data_splits, preprocessing_pipeline):
    y_test_pred = model.predict(data_splits['X_test'], verbose=0).flatten()
    y_test_true = data_splits['y_test']
    y_test_pred = np.clip(y_test_pred, 0, 1)

    mae = mean_absolute_error(y_test_true, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))

    create_evaluation_plots(model, data_splits, preprocessing_pipeline, y_test_true, y_test_pred)

    return {'mae': mae, 'rmse': rmse}

def implement_raoult_baseline(data_splits, preprocessing_pipeline):
    scaler = preprocessing_pipeline['scaler']
    X_test_original = scaler.inverse_transform(data_splits['X_test'])

    def estimate_psat(T, component='light'):
        if component == 'light':
            return 0.5 * np.exp(10 - 3000/T)
        else:
            return 0.3 * np.exp(8 - 2500/T)

    x1_test = X_test_original[:, 0]
    T_test = X_test_original[:, 1]
    P_test = X_test_original[:, 2]

    P_sat1 = estimate_psat(T_test, 'light')
    y1_baseline = (x1_test * P_sat1) / P_test
    y1_baseline = np.clip(y1_baseline, 0, 1)

    y_test_true = data_splits['y_test']
    mae_baseline = mean_absolute_error(y_test_true, y1_baseline)
    rmse_baseline = np.sqrt(mean_squared_error(y_test_true, y1_baseline))

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test_true, y=y1_baseline, alpha=0.7, s=60, color='orange')
    min_val, max_val = 0, 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    plt.xlabel('Experimental y1'); plt.ylabel('Baseline Predicted y1 (Raoult)')
    plt.title('Baseline Parity Plot: Raoult\'s Law vs Experimental')
    plt.legend(); plt.axis('equal'); plt.xlim(0, 1); plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('baseline_parity_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {'mae_baseline': mae_baseline, 'rmse_baseline': rmse_baseline}

def save_artifacts(model, preprocessing_pipeline):
    model.save('ann_vle_model.h5')
    joblib.dump(preprocessing_pipeline, 'preprocessing_pipeline.pkl')

def create_results_zip():
    files_to_zip = [
        'ann_vle_model.h5',
        'preprocessing_pipeline.pkl',
        'best_model.weights.h5',
        'parity_plot.png',
        'residuals_plot.png',
        'y_vs_x_plot.png',
        'training_history.png',
        'baseline_parity_plot.png'
    ]
    zip_filename = 'ANN_BinaryVLE.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in files_to_zip:
            if os.path.exists(file):
                zipf.write(file)
    return zip_filename

def main():
    EPOCHS = 200
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    SCALER_TYPE = 'standard'

    df = load_data('vle_data.csv')
    data_final = preprocess_data(df, SCALER_TYPE)
    splits_final = split_data(data_final['X_scaled'], data_final['y'])

    input_shape = (3,)
    model = build_model(input_shape, LEARNING_RATE)
    model, history = train_model(
        model,
        splits_final['X_train'], splits_final['y_train'],
        splits_final['X_val'], splits_final['y_val'],
        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=20
    )

    plot_training_history(history)
    save_artifacts(model, data_final['preprocessing_pipeline'])
    evaluation_results = evaluate_model(model, splits_final, data_final['preprocessing_pipeline'])
    baseline_results = implement_raoult_baseline(splits_final, data_final['preprocessing_pipeline'])

    zip_filename = create_results_zip()

if __name__ == "__main__":
    main()