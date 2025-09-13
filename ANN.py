# ANN.py
# Artificial Neural Network (ANN) for Binary VLE Surrogate Modeling
# ---------------------------------------------------------------
# This module defines functions to:
# 1. Build the ANN model with three hidden layers (64, 128, 64 neurons).
# 2. Compile the model with MSE loss and metrics (MAE, RMSE).
# 3. Train the model with early stopping and model checkpointing.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, learning_rate=0.001):
    """
    Build and compile an ANN model for VLE prediction.

    Args:
        input_shape (tuple): Shape of input features (e.g., (3,) for [x1, T, P]).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        keras.Model: Compiled ANN model.
    """
    # Define a Sequential ANN with 3 hidden layers
    model = keras.Sequential([
        layers.Input(shape=input_shape),                # Input layer (expects 3 features)
        layers.Dense(64, activation='relu', name='hidden_1'),   # Hidden layer 1: 64 neurons, ReLU activation
        layers.Dense(128, activation='relu', name='hidden_2'),  # Hidden layer 2: 128 neurons, ReLU activation
        layers.Dense(64, activation='relu', name='hidden_3'),   # Hidden layer 3: 64 neurons, ReLU activation
        layers.Dense(1, activation='sigmoid', name='output')    # Output layer: 1 neuron, sigmoid ensures [0,1]
    ])

    # Custom metric: Root Mean Squared Error (RMSE)
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    # Compile the model with Adam optimizer, MSE loss, and metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',              # Mean Squared Error (MSE) for regression
        metrics=['mae', rmse]    # Track Mean Absolute Error (MAE) and RMSE during training
    )

    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=32, patience=20):
    """
    Train the ANN model with early stopping and checkpointing.

    Args:
        model (keras.Model): Compiled ANN model.
        X_train (array): Training features.
        y_train (array): Training targets.
        X_val (array): Validation features.
        y_val (array): Validation targets.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per batch.
        patience (int): Number of epochs to wait for improvement before stopping.

    Returns:
        model (keras.Model): Trained model (with best weights restored).
        history (keras.callbacks.History): Training history (loss/metrics curves).
    """
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    # Define callbacks to control training
    callbacks = [
        # EarlyStopping: Stop training if validation loss does not improve
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,  # Revert to best weights when stopped
            verbose=0
        ),
        # ModelCheckpoint: Save the best model weights to disk
        ModelCheckpoint(
            'best_model.weights.h5',     # Filepath to save the weights
            monitor='val_loss',
            save_best_only=True,         # Save only the best model (lowest val_loss)
            save_weights_only=True,      # Save weights (not the whole model structure)
            verbose=0
        )
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,                # Training data
        validation_data=(X_val, y_val),  # Validation data
        epochs=epochs,                   # Maximum number of epochs
        batch_size=batch_size,           # Mini-batch size
        callbacks=callbacks,             # Apply callbacks
        verbose=0                        # Suppress detailed per-epoch output
    )

    return model, history