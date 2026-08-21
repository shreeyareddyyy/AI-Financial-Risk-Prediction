import argparse
import json
import pathlib
import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

import yfinance as yf
import matplotlib.pyplot as plt

from preprocessing import prepare, FEATURES
from model import PriceVolatilityLSTM
from config import *


TICKERS = {
    'gold': 'GC=F',
    'bitcoin': 'BTC-USD'
}


# =============================================================
# REPRODUCIBILITY
# =============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# =============================================================
# DOWNLOAD DATA
# =============================================================

def download(asset):

    ticker = TICKERS[asset]

    df = yf.download(
        ticker,
        period=DATA_PERIOD,
        interval='1d',
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)

    if df.empty:

        raise RuntimeError(
            f'No data returned for {ticker}. '
            'Check internet/yfinance and try again.'
        )

    return df


# =============================================================
# SAFE MAPE
# =============================================================

def calculate_mape(actual, predicted):

    actual = np.asarray(actual)

    predicted = np.asarray(predicted)

    denominator = np.maximum(
        np.abs(actual),
        1e-8
    )

    return float(
        np.mean(
            np.abs(
                (actual - predicted)
                / denominator
            )
        ) * 100
    )


# =============================================================
# EVALUATION
# =============================================================

def evaluate(
    model,
    X,
    y_return_scaled,
    y_volatility_scaled,
    return_scaler,
    volatility_scaler,
    base_close
):

    model.eval()

    with torch.no_grad():

        pred_return_scaled, pred_volatility_scaled = model(
            torch.tensor(X)
        )

        pred_return_scaled = (
            pred_return_scaled
            .numpy()
            .ravel()
        )

        pred_volatility_scaled = (
            pred_volatility_scaled
            .numpy()
            .ravel()
        )

    # ---------------------------------------------------------
    # RETURN
    # ---------------------------------------------------------

    pred_log_return = (
        return_scaler
        .inverse_transform(
            pred_return_scaled.reshape(-1, 1)
        )
        .ravel()
    )

    actual_log_return = (
        return_scaler
        .inverse_transform(
            y_return_scaled.reshape(-1, 1)
        )
        .ravel()
    )

    # ---------------------------------------------------------
    # PRICE
    # ---------------------------------------------------------

    pred_price = (
        base_close
        * np.exp(pred_log_return)
    )

    actual_price = (
        base_close
        * np.exp(actual_log_return)
    )

    # ---------------------------------------------------------
    # PRICE METRICS
    # ---------------------------------------------------------

    rmse = float(
        np.sqrt(
            mean_squared_error(
                actual_price,
                pred_price
            )
        )
    )

    mae = float(
        mean_absolute_error(
            actual_price,
            pred_price
        )
    )

    mape = calculate_mape(
        actual_price,
        pred_price
    )

    direction_actual = (
        actual_log_return > 0
    )

    direction_pred = (
        pred_log_return > 0
    )

    directional_accuracy = float(
        (
            direction_actual
            == direction_pred
        ).mean()
    )

    # ---------------------------------------------------------
    # NAIVE BASELINE
    # ---------------------------------------------------------

    naive_pred = base_close

    naive_rmse = float(
        np.sqrt(
            mean_squared_error(
                actual_price,
                naive_pred
            )
        )
    )

    naive_mae = float(
        mean_absolute_error(
            actual_price,
            naive_pred
        )
    )

    # ---------------------------------------------------------
    # IMPROVEMENT
    # ---------------------------------------------------------

    rmse_improvement = (
        (naive_rmse - rmse)
        / naive_rmse
        * 100
    )

    mae_improvement = (
        (naive_mae - mae)
        / naive_mae
        * 100
    )

    # ---------------------------------------------------------
    # VOLATILITY
    # ---------------------------------------------------------

    pred_log_volatility = (
        volatility_scaler
        .inverse_transform(
            pred_volatility_scaled.reshape(-1, 1)
        )
        .ravel()
    )

    actual_log_volatility = (
        volatility_scaler
        .inverse_transform(
            y_volatility_scaled.reshape(-1, 1)
        )
        .ravel()
    )

    # Convert log-volatility back to actual volatility
    pred_volatility = np.exp(
        pred_log_volatility
    )

    actual_volatility = np.exp(
        actual_log_volatility
    )

    # Numerical safety
    pred_volatility = np.maximum(
        pred_volatility,
        0
    )

    actual_volatility = np.maximum(
        actual_volatility,
        0
    )

    volatility_rmse = float(
        np.sqrt(
            mean_squared_error(
                actual_volatility,
                pred_volatility
            )
        )
    )

    volatility_mae = float(
        mean_absolute_error(
            actual_volatility,
            pred_volatility
        )
    )

    volatility_mape = calculate_mape(
        actual_volatility,
        pred_volatility
    )
    return (
        actual_price,
        pred_price,

        actual_log_return,
        pred_log_return,

        actual_volatility,
        pred_volatility,

        rmse,
        mae,
        mape,

        directional_accuracy,

        naive_rmse,
        naive_mae,

        rmse_improvement,
        mae_improvement,

        volatility_rmse,
        volatility_mae,
        volatility_mape
    )


# =============================================================
# MAIN
# =============================================================

def main(asset):

    set_seed(42)

    root = pathlib.Path(
        __file__
    ).resolve().parents[1]

    for directory in [
        'data',
        'models',
        'results',
        'results/plots'
    ]:

        (
            root / directory
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------

    raw = download(asset)

    raw.to_csv(
        root / f'data/{asset}_raw.csv'
    )

    (
        df,

        feature_scaler,
        return_scaler,
        volatility_scaler,

        Xtr,
        ytr_return,
        ytr_volatility,

        Xv,
        yv_return,
        yv_volatility,

        Xte,
        yte_return,
        yte_volatility,

        test_dates,
        test_base_close,
        test_actual_close,
        test_actual_volatility
    ) = prepare(
        raw,
        WINDOW_SIZE,
        TRAIN_RATIO,
        VAL_RATIO
    )

    # ---------------------------------------------------------
    # DATASET
    # ---------------------------------------------------------

    train_dataset = TensorDataset(
        torch.tensor(Xtr),
        torch.tensor(ytr_return).unsqueeze(1),
        torch.tensor(ytr_volatility).unsqueeze(1)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    model = PriceVolatilityLSTM(
        input_size=Xtr.shape[2],
        hidden1=HIDDEN_1,
        hidden2=HIDDEN_2,
        dropout=DROPOUT
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # ---------------------------------------------------------
    # LOSSES
    # ---------------------------------------------------------

    return_loss_fn = torch.nn.HuberLoss(
        delta=1.0
    )

    volatility_loss_fn = torch.nn.HuberLoss(
        delta=1.0
    )

    # Volatility receives a slightly higher weight because
    # it is the new prediction target we are improving.

    VOLATILITY_WEIGHT = 1.5

    best_val = float('inf')

    best_state = None

    patience_counter = 0

    train_hist = []

    val_hist = []

    # ---------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for (
            xb,
            yb_return,
            yb_volatility
        ) in train_loader:

            optimizer.zero_grad()

            pred_return, pred_volatility = model(xb)

            return_loss = return_loss_fn(
                pred_return,
                yb_return
            )

            volatility_loss = volatility_loss_fn(
                pred_volatility,
                yb_volatility
            )

            loss = (
                return_loss
                + VOLATILITY_WEIGHT
                * volatility_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += (
                loss.item()
                * len(xb)
            )

        train_loss = (
            total_loss
            / len(train_loader.dataset)
        )

        # -----------------------------------------------------
        # VALIDATION
        # -----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_return_pred, val_vol_pred = model(
                torch.tensor(Xv)
            )

            val_return_loss = return_loss_fn(
                val_return_pred,
                torch.tensor(
                    yv_return
                ).unsqueeze(1)
            )

            val_volatility_loss = volatility_loss_fn(
                val_vol_pred,
                torch.tensor(
                    yv_volatility
                ).unsqueeze(1)
            )

            val_loss = (
                val_return_loss
                + VOLATILITY_WEIGHT
                * val_volatility_loss
            ).item()

        scheduler.step(val_loss)

        train_hist.append(
            train_loss
        )

        val_hist.append(
            val_loss
        )

        print(
            f'Epoch {epoch+1:03d}/{EPOCHS} '
            f'train={train_loss:.6f} '
            f'val={val_loss:.6f} '
            f'lr={optimizer.param_groups[0]["lr"]:.6g}'
        )

        # -----------------------------------------------------
        # CHECKPOINT
        # -----------------------------------------------------

        if val_loss < best_val - 1e-6:

            best_val = val_loss

            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    'Early stopping.'
                )

                break

    if best_state is None:

        raise RuntimeError(
            'Training did not produce a valid checkpoint.'
        )

    model.load_state_dict(
        best_state
    )

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------

    (
        actual_price,
        pred_price,

        actual_ret,
        pred_ret,

        actual_volatility,
        pred_volatility,

        rmse,
        mae,
        mape,

        dir_acc,

        naive_rmse,
        naive_mae,

        rmse_improvement,
        mae_improvement,

        volatility_rmse,
        volatility_mae,
        volatility_mape

    ) = evaluate(
        model,
        Xte,

        yte_return,
        yte_volatility,

        return_scaler,
        volatility_scaler,

        test_base_close
    )

    # ---------------------------------------------------------
    # LATEST
    # ---------------------------------------------------------

    latest_actual = float(
        actual_price[-1]
    )

    latest_pred = float(
        pred_price[-1]
    )

    latest_pred_return = float(
        pred_ret[-1]
    )

    latest_pred_volatility = float(
        pred_volatility[-1]
    )

    latest_actual_volatility = float(
        actual_volatility[-1]
    )

    trend = (
        'UP'
        if latest_pred_return > 0
        else
        'DOWN'
        if latest_pred_return < 0
        else
        'NEUTRAL'
    )

    latest_volatility_30 = float(
        df['Volatility_30'].iloc[-1]
    )

    # ---------------------------------------------------------
    # RESULT CSV
    # ---------------------------------------------------------

    result = pd.DataFrame({

        'date': test_dates,

        'asset': asset,

        'actual_price': actual_price,

        'predicted_price': pred_price,

        'actual_return': actual_ret,

        'predicted_return': pred_ret,

        'actual_volatility': actual_volatility,

        'predicted_volatility': pred_volatility
    })

    result['predicted_return_percent'] = (
        result['predicted_return']
        * 100
    )

    result['trend'] = np.where(
        result['predicted_return'] > 0,
        'UP',
        np.where(
            result['predicted_return'] < 0,
            'DOWN',
            'NEUTRAL'
        )
    )

    result['volatility_30'] = (
        df['Volatility_30']
        .iloc[-len(result):]
        .to_numpy()
    )

    result['rmse'] = rmse

    result['mae'] = mae

    result['mape'] = mape

    result['volatility_rmse'] = (
        volatility_rmse
    )

    result['volatility_mae'] = (
        volatility_mae
    )

    result['volatility_mape'] = (
        volatility_mape
    )

    result.to_csv(
        root
        / f'results/{asset}_prediction_output.csv',
        index=False
    )

    # ---------------------------------------------------------
    # METRICS JSON
    # ---------------------------------------------------------

    metrics = {

        'asset': asset,

        'ticker': TICKERS[asset],

        'model': 'Multivariate LSTM - price return + future volatility',

        'data_period': DATA_PERIOD,

        'window_size': WINDOW_SIZE,

        'volatility_horizon_days': 5,

        'features': list(FEATURES),

        # Price
        'rmse': rmse,

        'mae': mae,

        'mape': mape,

        'naive_rmse': naive_rmse,

        'naive_mae': naive_mae,

        'rmse_improvement_vs_naive':
            rmse_improvement,

        'mae_improvement_vs_naive':
            mae_improvement,

        'directional_accuracy':
            dir_acc,

        # Volatility
        'volatility_rmse':
            volatility_rmse,

        'volatility_mae':
            volatility_mae,

        'volatility_mape':
            volatility_mape,

        # Latest
        'latest_actual_price':
            latest_actual,

        'latest_predicted_price':
            latest_pred,

        'latest_predicted_return':
            latest_pred_return,

        'latest_predicted_return_percent':
            latest_pred_return * 100,

        'latest_trend':
            trend,

        'latest_historical_volatility_30':
            latest_volatility_30,

        'latest_actual_future_volatility':
            latest_actual_volatility,

        'latest_predicted_future_volatility':
            latest_pred_volatility,

        'train_rows':
            len(Xtr),

        'validation_rows':
            len(Xv),

        'test_rows':
            len(Xte)
    }

    (
        root
        / f'results/{asset}_metrics.json'
    ).write_text(
        json.dumps(
            metrics,
            indent=2
        )
    )

    # ---------------------------------------------------------
    # MODEL CHECKPOINT
    # ---------------------------------------------------------

    torch.save({

        'model_state_dict':
            model.state_dict(),

        'feature_scaler':
        feature_scaler,

        'return_scaler':
            return_scaler,

        'volatility_scaler':
            volatility_scaler,

        'window_size':
            WINDOW_SIZE,

        'features':
            list(FEATURES),

        'ticker':
            TICKERS[asset],

        'target':
            'next_day_log_return',

        'volatility_target':
            'next_5_day_future_volatility',

        'volatility_target_transform':
            'none',

        'rmse':
            rmse,

        'mae':
            mae,

        'mape':
            mape,

        'volatility_rmse':
            volatility_rmse,

        'volatility_mae':
            volatility_mae,

        'volatility_mape':
            volatility_mape,

        'hidden1':
            HIDDEN_1,

        'hidden2':
            HIDDEN_2,

        'dropout':
            DROPOUT

    }, root / f'models/{asset}_lstm.pt')

    # ---------------------------------------------------------
    # PLOTS
    # ---------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        actual_price,
        label='Actual'
    )

    plt.plot(
        pred_price,
        label='Predicted'
    )

    plt.title(
        f'{asset.title()} - Actual vs Predicted Price'
    )

    plt.xlabel(
        'Test observations'
    )

    plt.ylabel(
        'Price'
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        root
        / f'results/plots/{asset}_actual_vs_predicted.png',
        dpi=160
    )

    plt.close()

    # ---------------------------------------------------------
    # VOLATILITY PLOT
    # ---------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        actual_volatility,
        label='Actual Future Volatility'
    )

    plt.plot(
        pred_volatility,
        label='Predicted Future Volatility'
    )

    plt.title(
        f'{asset.title()} - Future Volatility'
    )

    plt.xlabel(
        'Test observations'
    )

    plt.ylabel(
        '5-day volatility'
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        root
        / f'results/plots/{asset}_future_volatility.png',
        dpi=160
    )

    plt.close()

    # ---------------------------------------------------------
    # TRAINING LOSS
    # ---------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        train_hist,
        label='Train loss'
    )

    plt.plot(
        val_hist,
        label='Validation loss'
    )

    plt.title(
        f'{asset.title()} - Training/Validation Loss'
    )

    plt.xlabel(
        'Epoch'
    )

    plt.ylabel(
        'Combined Huber loss'
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        root
        / f'results/plots/{asset}_training_loss.png',
        dpi=160
    )

    plt.close()

    # ---------------------------------------------------------
    # HISTORICAL VOLATILITY PLOT
    # ---------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        df.index,
        df['Volatility_30']
    )

    plt.title(
        f'{asset.title()} - 30-day Historical Volatility'
    )

    plt.xlabel(
        'Date'
    )

    plt.ylabel(
        'Volatility'
    )

    plt.tight_layout()

    plt.savefig(
        root
        / f'results/plots/{asset}_volatility_30.png',
        dpi=160
    )

    plt.close()

    # =========================================================
    # FINAL OUTPUT
    # =========================================================

    print()
    print('=' * 60)
    print('FINAL METRICS')
    print('=' * 60)

    print(
        f'Asset: {asset} ({TICKERS[asset]})'
    )

    print()

    print('--- PRICE / TREND ---')

    print(
        f'RMSE: {rmse:.4f}'
    )

    print(
        f'MAE : {mae:.4f}'
    )

    print(
        f'MAPE: {mape:.2f}%'
    )

    print(
        f'Naive RMSE (previous close): '
        f'{naive_rmse:.4f}'
    )

    print(
        f'Naive MAE  (previous close): '
        f'{naive_mae:.4f}'
    )

    print(
        f'RMSE improvement vs naive: '
        f'{rmse_improvement:.2f}%'
    )

    print(
        f'MAE improvement vs naive: '
        f'{mae_improvement:.2f}%'
    )

    print(
        f'Directional accuracy: '
        f'{dir_acc:.2%}'
    )

    print()

    print('--- VOLATILITY ---')

    print(
        f'Volatility RMSE: '
        f'{volatility_rmse:.6f}'
    )

    print(
        f'Volatility MAE : '
        f'{volatility_mae:.6f}'
    )

    print(
        f'Volatility MAPE: '
        f'{volatility_mape:.2f}%'
    )

    print()

    print('--- LATEST PREDICTION ---')

    print(
        f'Latest actual price: '
        f'{latest_actual:.4f}'
    )

    print(
        f'Latest predicted price: '
        f'{latest_pred:.4f}'
    )

    print(
        f'Predicted return: '
        f'{latest_pred_return * 100:.4f}%'
    )

    print(
        f'Trend: {trend}'
    )

    print(
        f'Current 30-day historical volatility: '
        f'{latest_volatility_30:.6f}'
    )

    print(
        f'Actual future volatility: '
        f'{latest_actual_volatility:.6f}'
    )

    print(
        f'Predicted future volatility: '
        f'{latest_pred_volatility:.6f}'
    )

    print('=' * 60)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--asset',
        choices=TICKERS,
        required=True
    )

    args = parser.parse_args()

    main(args.asset)