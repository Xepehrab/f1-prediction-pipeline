"""LightGBM training and prediction for F1 race outcomes.

Uses a temporal split (train on seasons before split_year,
evaluate on split_year onwards) to avoid look-ahead leakage.
"""

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from .config import MODEL_DIR, PROCESSED_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def train_baseline_model(target_col='podium', split_year=2025):
    """
    Train a LightGBM model using a temporal split.
    Data prior to split_year is used for training, and split_year onwards for evaluation.
    """
    features_path = PROCESSED_DIR / "features.csv"
    dict_path = PROCESSED_DIR / "feature_dictionary.csv"
    
    if not features_path.exists() or not dict_path.exists():
        logger.error("Data files not found. Run 'python -m f1_prediction build' first.")
        return False

    df = pd.read_csv(features_path)
    feat_dict = pd.read_csv(dict_path)

    # Extract features based on the data dictionary
    features = feat_dict[feat_dict['role'] == 'feature']['column'].tolist()
    
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in the dataset.")
        return False

    # Temporal split (leakage-safe)
    train_df = df[df['season'] < split_year].copy()
    test_df = df[df['season'] >= split_year].copy()

    X_train, y_train = train_df[features], train_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]

    logger.info(f"Training on {len(X_train)} rows (pre-{split_year})")
    logger.info(f"Evaluating on {len(X_test)} rows ({split_year} onwards)")

    # Initialize LightGBM baseline model
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        class_weight='balanced', # Useful for imbalanced targets like podiums
        random_state=42,
        # device='gpu' # Uncomment this line if LightGBM GPU build is installed
    )
    
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    # Predict and evaluate
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    logger.info("\n--- Model Evaluation Results ---")
    logger.info(f"\n{classification_report(y_test, preds)}")
    logger.info(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")

    # Calculate and display feature importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': clf.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(10)
    
    logger.info("\n--- Top 10 Features ---")
    logger.info(f"\n{importance_df.to_string(index=False)}")

    # Save the trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_file = MODEL_DIR / f"lgbm_{target_col}_model.pkl"
    joblib.dump(clf, model_file)
    logger.info(f"Model successfully saved to {model_file}")
    
    return True

def predict_specific_race(season, round_num, target_col='won'):
    """
    Predict outcomes for a specific race and rank drivers by probability.
    """
    features_path = PROCESSED_DIR / "features.csv"
    dict_path = PROCESSED_DIR / "feature_dictionary.csv"
    model_file = MODEL_DIR / f"lgbm_{target_col}_model.pkl"
    
    if not model_file.exists():
        logger.error(f"Model for target '{target_col}' not found. Train the model first.")
        return

    # Load model and data
    clf = joblib.load(model_file)
    df = pd.read_csv(features_path)
    feat_dict = pd.read_csv(dict_path)
    features = feat_dict[feat_dict['role'] == 'feature']['column'].tolist()

    # Filter for the specific race
    race_df = df[(df['season'] == season) & (df['round'] == round_num)].copy()
    
    if race_df.empty:
        logger.error(f"No data found for season {season}, round {round_num}.")
        return

    # Predict probabilities
    probs = clf.predict_proba(race_df[features])[:, 1]
    race_df['probability'] = probs * 100  # Convert to percentage

    # Sort drivers by highest probability
    results = race_df[['driver_id', 'probability']].sort_values(by='probability', ascending=False)

    logger.info(f"\n--- Prediction for target '{target_col}' | Season {season} Round {round_num} ---")
    for index, row in results.iterrows():
        print(f"Driver: {row['driver_id']:<20} | Probability: {row['probability']:.2f}%")

if __name__ == "__main__":
    train_baseline_model()