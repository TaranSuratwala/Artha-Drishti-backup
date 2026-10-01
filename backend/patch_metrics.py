import re
import sys

def patch_metrics():
    with open('MLPredictor.py', 'r', encoding='utf-8') as f:
        code = f.read()

    # Guard compute_regression_metrics
    code = re.sub(
        r'(def compute_regression_metrics\(y_true: np\.ndarray, y_pred: np\.ndarray\) -> Dict:\n\s*\"\"\".*?\"\"\")',
        r'\1\n        if len(y_true) == 0:\n            return {"mse": 0, "rmse": 0, "mae": 0, "r2_score": 0, "mape": "N/A", "smape": 0, "max_error": 0, "explained_variance": 0}',
        code, count=1
    )

    # Guard compute_classification_metrics
    code = re.sub(
        r'(def compute_classification_metrics\(y_true: np\.ndarray, y_pred: np\.ndarray\) -> Dict:\n\s*\"\"\".*?\"\"\")',
        r'\1\n        if len(y_true) == 0:\n            return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "balanced_accuracy": 0, "mcc": 0, "true_positives": 0, "true_negatives": 0, "false_positives": 0, "false_negatives": 0}',
        code, count=1
    )

    # Guard compute_business_metrics
    code = re.sub(
        r'(def compute_business_metrics\(predictions: List\[Dict\]\) -> Dict:\n\s*\"\"\".*?\"\"\")',
        r'\1\n        if not predictions:\n            return {"win_rate": 0, "avg_profit": 0, "avg_loss": 0, "profit_factor": 0, "avg_risk_reward": 0, "sharpe_ratio": 0, "max_drawdown": 0, "total_pnl": 0}',
        code, count=1
    )

    with open('MLPredictor.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print("Metrics guards applied.")

if __name__ == '__main__':
    patch_metrics()
