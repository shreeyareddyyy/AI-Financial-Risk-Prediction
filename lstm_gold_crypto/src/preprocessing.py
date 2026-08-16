import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


FEATURES = [
    'Open',
    'High',
    'Low',
    'Close',
    'Volume',
    'Return',
    'SMA_7',
    'SMA_30',
    'EMA_14',
    'RSI_14',
    'MACD',
    'MACD_Signal',
    'Volatility_7',
    'Volatility_30',
    'BB_Upper',
    'BB_Lower'
]


def clean_data(df):

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]

    cols = [
        c for c in ['Open', 'High', 'Low', 'Close', 'Volume']
        if c in df.columns
    ]

    df = df[cols].apply(pd.to_numeric, errors='coerce')

    df = df.dropna(subset=['Close'])

    if 'Volume' not in df.columns:
        df['Volume'] = 0.0

    return df


def add_features(df):

    out = clean_data(df)

    close = out['Close']

    # ---------------------------------------------------------
    # DAILY RETURN
    # ---------------------------------------------------------

    out['Return'] = close.pct_change()

    # ---------------------------------------------------------
    # MOVING AVERAGES
    # ---------------------------------------------------------

    out['SMA_7'] = close.rolling(7).mean()
    out['SMA_30'] = close.rolling(30).mean()

    out['EMA_14'] = close.ewm(
        span=14,
        adjust=False
    ).mean()

    # ---------------------------------------------------------
    # RSI
    # ---------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)

    out['RSI_14'] = 100 - (
        100 / (1 + rs)
    )

    # ---------------------------------------------------------
    # MACD
    # ---------------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    out['MACD'] = ema12 - ema26

    out['MACD_Signal'] = out['MACD'].ewm(
        span=9,
        adjust=False
    ).mean()

    # ---------------------------------------------------------
    # HISTORICAL VOLATILITY
    # ---------------------------------------------------------

    out['Volatility_7'] = (
        out['Return']
        .rolling(7)
        .std()
    )

    out['Volatility_30'] = (
        out['Return']
        .rolling(30)
        .std()
    )

    # ---------------------------------------------------------
    # BOLLINGER BANDS
    # ---------------------------------------------------------

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()

    out['BB_Upper'] = mid + 2 * std
    out['BB_Lower'] = mid - 2 * std

    # ---------------------------------------------------------
    # LOG RETURN
    # ---------------------------------------------------------

    out['Log_Return'] = np.log(
        close / close.shift(1)
    )

    # ---------------------------------------------------------
    # FUTURE VOLATILITY TARGET
    # ---------------------------------------------------------
    #
    # At day t:
    #
    # target = standard deviation of the NEXT 5
    # daily log returns.
    #
    # Therefore today's return is NOT included.
    #
    # Example:
    #
    # t+1, t+2, t+3, t+4, t+5
    #
    # are used to calculate the target for t.
    # ---------------------------------------------------------

    future_returns = out['Log_Return'].shift(-1)

    out['Future_Volatility'] = (
        future_returns
        .rolling(5)
        .std()
        .shift(-4)
    )

    # ---------------------------------------------------------
    # LOG FUTURE VOLATILITY
    # ---------------------------------------------------------
    #
    # Volatility is always positive and often highly skewed,
    # especially for Bitcoin.
    #
    # Taking log makes the target distribution much more stable.
    # ---------------------------------------------------------

    out['Log_Future_Volatility'] = np.log(
        out['Future_Volatility'].clip(lower=1e-8)
    )

    out = out.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    return out


def make_sequences(
    features,
    return_target,
    volatility_target,
    window
):

    X = []
    y_return = []
    y_volatility = []

    for i in range(window, len(features)):

        X.append(
            features[i-window:i]
        )

        y_return.append(
            return_target[i]
        )

        y_volatility.append(
            volatility_target[i]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y_return, dtype=np.float32),
        np.asarray(y_volatility, dtype=np.float32)
    )


def prepare(
    df,
    window=60,
    train_ratio=0.8,
    val_ratio=0.1
):

    df = add_features(df)

    n = len(df)

    train_end = int(
        n * train_ratio
    )

    val_end = int(
        n * (train_ratio + val_ratio)
    )

    # ---------------------------------------------------------
    # FEATURE SCALER
    # ---------------------------------------------------------

    feature_scaler = MinMaxScaler()

    feature_scaler.fit(
        df.iloc[:train_end][FEATURES]
    )

    scaled_features = feature_scaler.transform(
        df[FEATURES]
    )

    # ---------------------------------------------------------
    # RETURN SCALER
    # ---------------------------------------------------------

    return_scaler = StandardScaler()

    return_scaler.fit(
        df.iloc[:train_end][['Log_Return']]
    )

    scaled_return = return_scaler.transform(
        df[['Log_Return']]
    ).ravel()

    # ---------------------------------------------------------
    # LOG VOLATILITY SCALER
    # ---------------------------------------------------------

    volatility_scaler = StandardScaler()

    volatility_scaler.fit(
        df.iloc[:train_end][[
            'Log_Future_Volatility'
        ]]
    )

    scaled_volatility = (
        volatility_scaler
        .transform(
            df[['Log_Future_Volatility']]
        )
        .ravel()
    )

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    (
        X_train,
        y_train_return,
        y_train_volatility
    ) = make_sequences(
        scaled_features[:train_end],
        scaled_return[:train_end],
        scaled_volatility[:train_end],
        window
    )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    (
        X_val,
        y_val_return,
        y_val_volatility
    ) = make_sequences(
        scaled_features[
            train_end-window:val_end
        ],
        scaled_return[
            train_end-window:val_end
        ],
        scaled_volatility[
            train_end-window:val_end
        ],
        window
    )

    # ---------------------------------------------------------
    # TEST
    # ---------------------------------------------------------

    (
        X_test,
        y_test_return,
        y_test_volatility
    ) = make_sequences(
        scaled_features[
            val_end-window:
        ],
        scaled_return[
            val_end-window:
        ],
        scaled_volatility[
            val_end-window:
        ],
        window
    )

    # ---------------------------------------------------------
    # PRICE RECONSTRUCTION
    # ---------------------------------------------------------

    test_base_close = (
        df['Close']
        .iloc[val_end-1:-1]
        .to_numpy(dtype=np.float64)
        [:len(X_test)]
    )

    test_actual_close = (
        df['Close']
        .iloc[val_end:]
        .to_numpy(dtype=np.float64)
        [:len(X_test)]
    )

    test_dates = df.index[
        val_end:
        val_end + len(X_test)
    ]
    
    test_actual_volatility = (
    df['Future_Volatility']
    .iloc[val_end:]
    .to_numpy(dtype=np.float64)
    [:len(X_test)]
    )

    return (
    df,

    feature_scaler,
    return_scaler,
    volatility_scaler,

    X_train,
    y_train_return,
    y_train_volatility,

    X_val,
    y_val_return,
    y_val_volatility,

    X_test,
    y_test_return,
    y_test_volatility,

    test_dates,
    test_base_close,
    test_actual_close,
    test_actual_volatility
)