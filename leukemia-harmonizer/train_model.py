import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


def train_leukemia_model(df: pd.DataFrame, target_col: str):
    """
    Simple XGBoost training wrapper.
    Assumes df has numeric CBC features and a target label column.
    """

    # X = all numeric columns except target
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(y.unique()),
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "n_classes": int(len(y.unique())),
        "n_samples": int(len(df)),
    }

    return model, metrics
