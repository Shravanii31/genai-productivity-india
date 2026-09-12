"""
Step 6: Train the deep-learning models -- a plain feedforward NN (for comparison)
and the hybrid Wide & Deep model (Cheng et al., 2016).

Wide & Deep, built with the Keras functional API:
  - Wide component: a single Dense(1, linear) applied directly to the scaled
    inputs -- mathematically a linear regression.
  - Deep component: 3 hidden layers (128 -> 64 -> 32, ReLU), each followed by
    BatchNormalization and Dropout (0.3 / 0.2 / 0.1), feeding a Dense(1) output.
  - The wide output and deep output are summed into the final prediction.

Both models train on the imputed+scaled training features, with a validation split
carved out of the training data (never the test set), EarlyStopping (patience=25,
restore_best_weights=True) and ReduceLROnPlateau.

Output:
  models/plain_nn.keras, models/hybrid_wide_deep.keras
  reports/metrics_deep.csv
  figures/06_plain_nn_loss.png, figures/07_hybrid_loss.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIG_DIR = Path("figures")
RANDOM_STATE = 42

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_splits():
    X_train = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]
    return X_train, X_test, y_train, y_test


def build_plain_nn(n_features: int) -> keras.Model:
    inputs = keras.Input(shape=(n_features,), name="features")
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    output = layers.Dense(1, name="output")(x)
    model = keras.Model(inputs, output, name="plain_nn")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def build_wide_and_deep(n_features: int) -> keras.Model:
    inputs = keras.Input(shape=(n_features,), name="features")

    # Wide component: a single linear path == linear regression
    wide_output = layers.Dense(1, activation="linear", name="wide_output")(inputs)

    # Deep component: 3 hidden layers with BN + Dropout
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    deep_output = layers.Dense(1, name="deep_output")(x)

    combined = layers.Add(name="wide_deep_sum")([wide_output, deep_output])
    model = keras.Model(inputs, combined, name="wide_and_deep")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def fit_with_callbacks(model, X_tr, y_tr, X_val, y_val):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=25, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
        ),
    ]
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )
    return history


def plot_loss(history, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train", color="#F0A83A")
    axes[0].plot(history.history["val_loss"], label="val", color="#33C7B0")
    axes[0].set_title(f"{title} -- Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["mae"], label="train", color="#F0A83A")
    axes[1].plot(history.history["val_mae"], label="val", color="#33C7B0")
    axes[1].set_title(f"{title} -- MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate(name, model, X_test, y_test, results):
    pred = model.predict(X_test, verbose=0).flatten()
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    results.append({"model": name, "r2": r2, "rmse": rmse, "mae": mae, "fit_seconds": None})
    print(f"{name:28s}  R2={r2:7.4f}  RMSE={rmse:7.4f}  MAE={mae:7.4f}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_splits()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    n_features = X_train.shape[1]
    results = []

    print("Training plain feedforward NN ...")
    plain = build_plain_nn(n_features)
    hist_plain = fit_with_callbacks(plain, X_tr, y_tr, X_val, y_val)
    plot_loss(hist_plain, "Plain NN", FIG_DIR / "06_plain_nn_loss.png")
    evaluate("Plain NN", plain, X_test, y_test, results)
    plain.save(MODELS_DIR / "plain_nn.keras")

    print("Training hybrid Wide & Deep model ...")
    hybrid = build_wide_and_deep(n_features)
    hist_hybrid = fit_with_callbacks(hybrid, X_tr, y_tr, X_val, y_val)
    plot_loss(hist_hybrid, "Hybrid Wide & Deep", FIG_DIR / "07_hybrid_loss.png")
    evaluate("Hybrid Wide & Deep", hybrid, X_test, y_test, results)
    hybrid.save(MODELS_DIR / "hybrid_wide_deep.keras")

    results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
    results_df.to_csv(REPORTS_DIR / "metrics_deep.csv", index=False)
    print(f"\nSaved -> {REPORTS_DIR / 'metrics_deep.csv'}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
