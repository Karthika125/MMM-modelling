Marketing Mix Modeling (MMM) with Mediation Assumption
Overview

This project develops a Marketing Mix Model (MMM) to explain and forecast Revenue using a 2-year weekly dataset. The dataset includes:

Paid media spends (Facebook, Google, TikTok, Instagram, Snapchat)

Direct response levers (Email, SMS)

Average price, promotions, and social followers

A key aspect of the modeling is the mediation assumption:
Google spend is treated as a mediator between social/display channels (Facebook, TikTok, Snapchat, Instagram) and Revenue.

This reflects the idea that social media advertising stimulates search intent, which increases Google spend, and in turn contributes to revenue.

Folder Structure
ASSESSMENT2/
│
├── data/
│   └── Assesment 2-MMM weekly_data.csv         # Input dataset (2 years weekly)
│
├── outputs/                    # Auto-generated when running
│   ├── feature_importance.png  # Visualization of key drivers
│   ├── xgb_model.pkl           # Trained XGBoost model
│   └── scaler.pkl              # Saved preprocessing scaler
│
├── mmm_modeling.py             # Main end-to-end script
├── requirements.txt            # Dependencies
└── README.md                   # Documentation

Setup and Installation

Clone the repository and move into the project folder:

git clone <your_repo_link>
cd MMM_Modeling


Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt

How to Run

Place the dataset in:

data/weekly_data.csv


Run the script:

python mmm_modeling.py


Outputs will be generated in the outputs/ folder:

Trained model and scaler (xgb_model.pkl, scaler.pkl)

Feature importance plot (feature_importance.png)

Mediation results, validation metrics, and diagnostics will also be printed in the console.

Methodology
1. Data Preparation

Weekly index feature added for trend/seasonality.

Zero-spend periods handled with robust scaling.

Revenue log-transformed for better distribution.

StandardScaler applied before modeling.

2. Modeling Approach

Stage 1 (Mediator model): Google spend regressed on Facebook, TikTok, Instagram, and Snapchat.

Stage 2 (Revenue model): Log(Revenue) regressed on predicted Google spend and other features.

Chosen model: XGBoost Regressor, for handling nonlinearities and interactions.

Validation: Rolling time-series cross-validation to avoid lookahead bias.

3. Causal Framing

Google modeled as a mediator to avoid back-door bias.

Interpretation considers both direct and indirect effects of social media.

4. Diagnostics

Out-of-sample RMSE reported from cross-validation.

Residual analysis performed for bias and autocorrelation.

Sensitivity tests included for Average Price and Promotions.

5. Insights and Recommendations

Price increases show negative effects on revenue (price elasticity).

Promotions lift revenue but with diminishing returns.

Email/SMS consistently add incremental value.

Social media influences revenue primarily through its effect on Google spend.

Risks include multicollinearity and attribution complexity due to mediated effects.

Deliverables

End-to-end modeling pipeline in mmm_modeling.py

Saved model, scaler, and diagnostic plots in outputs/

This README as a short write-up documenting the methodology, assumptions, and findings