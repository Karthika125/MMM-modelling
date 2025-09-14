import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ========================
# 1. Load & Preprocess Data
# ========================

def load_and_prepare(filepath):
    df = pd.read_csv(filepath, parse_dates=['week'])
    df.sort_values('week', inplace=True)

    # Fill missing values with 0 for spends, promotions, comms
    cols_to_fill = [
        'facebook_spend','google_spend','tiktok_spend',
        'instagram_spend','snapchat_spend',
        'emails_send','sms_send','promotions'
    ]
    for col in cols_to_fill:
        df[col] = df[col].fillna(0)

    # Lag Google spend (mediator assumption)
    df['google_lag1'] = df['google_spend'].shift(1)

    # Weekly seasonality
    df['week_of_year'] = df['week'].dt.isocalendar().week

    # Log-transform revenue
    df['revenue_log'] = np.log1p(df['revenue'])

    df = df.dropna()  # remove first NA row due to lag
    return df

# ========================
# 2. Mediation Analysis
# ========================

def mediation_analysis(df):
    treatment_cols = ['facebook_spend','tiktok_spend','instagram_spend','snapchat_spend']
    mediator_col = 'google_spend'
    target_col = 'revenue_log'

    # Stage 1: Predict Google spend from other media
    X_mediator = sm.add_constant(df[treatment_cols])
    y_mediator = df[mediator_col]
    mediator_model = sm.OLS(y_mediator, X_mediator).fit()
    df['pred_google'] = mediator_model.predict(X_mediator)

    # Stage 2: Predict revenue using predicted Google + treatments
    X_rev = sm.add_constant(df[treatment_cols + ['pred_google']])
    y_rev = df[target_col]
    revenue_model = sm.OLS(y_rev, X_rev).fit()

    print("\n--- Mediator Model (Google Spend) ---")
    print(mediator_model.summary())
    print("\n--- Revenue Model (with predicted Google) ---")
    print(revenue_model.summary())
    return df, mediator_model, revenue_model

# ========================
# 3. Machine Learning Model
# ========================

from sklearn.metrics import mean_squared_error

def train_xgb(df):
    feature_cols = [
        'facebook_spend','tiktok_spend','instagram_spend','snapchat_spend',
        'google_lag1','emails_send','sms_send','promotions',
        'average_price','social_followers'
    ]
    target_col = 'revenue_log'

    # Scale numeric features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X = df[feature_cols]
    y = df[target_col]

    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # ✅ FIXED
        rmses.append(rmse)

    # Train final model on full dataset
    final_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    final_model.fit(X, y)

    print("\n--- XGBoost CV RMSEs ---")
    print(rmses)
    print("Average RMSE:", np.mean(rmses))
    return final_model, scaler, feature_cols


# ========================
# 4. Diagnostics & Save
# ========================

def plot_importance(model, feature_cols, save_path="feature_importance.png"):
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    feat, imp = zip(*feat_imp)

    plt.figure(figsize=(10,6))
    sns.barplot(x=imp, y=feat)
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def save_outputs(model, scaler, folder="outputs"):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(model, os.path.join(folder, "xgb_model.pkl"))
    joblib.dump(scaler, os.path.join(folder, "scaler.pkl"))
    print(f"Models saved to {folder}/")

# ========================
# Main Execution
# ========================

if __name__ == "__main__":
    # 1. Load & preprocess
    df = load_and_prepare("data/Assessment 2 - MMM Weekly.csv")

    # 2. Mediation analysis
    df, mediator_model, revenue_model = mediation_analysis(df)

    # 3. Train ML model
    xgb_model, scaler, feature_cols = train_xgb(df)

    # 4. Plot feature importance
    plot_importance(xgb_model, feature_cols, save_path="outputs/feature_importance.png")

    # 5. Save model + scaler
    save_outputs(xgb_model, scaler, folder="outputs")
