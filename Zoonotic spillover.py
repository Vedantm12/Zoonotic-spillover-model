import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    make_scorer
)

# Core Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# 1. Pipeline Setup & Model Definitions

def build_model_zoo(scale_pos_weight_val=1.0, random_state=42):
    """
    Constructs an ensemble of ML classifiers configured for 
    ecological non-linearities and severe class imbalance.
    """
    models = {
        # 1. Random Forest (Subsample Balanced)
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state
        ),

        # 2. XGBoost (Gradient Boosted Trees)
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight_val,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        ),

        # 3. LightGBM (Leaf-wise Tree Growth)
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight_val,
            verbose=-1,
            random_state=random_state,
            n_jobs=-1
        ),

        # 4. Support Vector Machine (RBF Kernel + Scaling)
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,  # Necessary for AUC and risk probability output
                class_weight="balanced",
                random_state=random_state
            ))
        ]),

        # 5. Artificial Neural Network / MLP (+ Scaling)
        "ANN (MLP)": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-3,          # L2 regularization
                learning_rate_init=1e-3,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=random_state
            ))
        ])
    }
    return models


# 2. Evaluation & Benchmarking Engine

def evaluate_models(X, y, n_splits=5, random_state=42):
    """
    Runs Stratified K-Fold CV evaluating ROC-AUC, PR-AUC, and Brier Score.
    """
    # Calculate positive-class weight for boosted models: (negatives / positives)
    num_neg = np.sum(y == 0)
    num_pos = np.sum(y == 1)
    scale_pos_weight = num_neg / max(num_pos, 1)

    models = build_model_zoo(scale_pos_weight_val=scale_pos_weight, random_state=random_state)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    scoring = {
        "ROC_AUC": "roc_auc",
        "PR_AUC": "average_precision",
        "Brier_Score": make_scorer(brier_score_loss, response_method="predict_proba", greater_is_better=False)
    }

    results = []

    print(f"Dataset Shape: {X.shape} | Positive Spillover Events: {num_pos} ({num_pos / len(y):.2%})\n")
    print(f"{'Model':<16} | {'ROC-AUC':<15} | {'PR-AUC (Avg Prec)':<18} | {'Brier Score (lower=better)':<20}")
    print("-" * 80)

    for name, model in models.items():
        cv_res = cross_validate(
            model, X, y, 
            cv=cv, 
            scoring=scoring, 
            n_jobs=-1, 
            return_train_score=False
        )
        
        mean_roc = np.mean(cv_res["test_ROC_AUC"])
        std_roc = np.std(cv_res["test_ROC_AUC"])
        
        mean_pr = np.mean(cv_res["test_PR_AUC"])
        std_pr = np.std(cv_res["test_PR_AUC"])
        
        # Brier score is returned negative by scikit-learn convention
        mean_brier = -np.mean(cv_res["test_Brier_Score"])
        std_brier = np.std(cv_res["test_Brier_Score"])
        
        print(f"{name:<16} | {mean_roc:.3f} ± {std_roc:.3f}     | {mean_pr:.3f} ± {std_pr:.3f}        | {mean_brier:.4f} ± {std_brier:.4f}")
        
        results.append({
            "Model": name,
            "ROC_AUC_mean": mean_roc,
            "ROC_AUC_std": std_roc,
            "PR_AUC_mean": mean_pr,
            "PR_AUC_std": std_pr,
            "Brier_mean": mean_brier,
            "Brier_std": std_brier
        })

    return pd.DataFrame(results)




# 3. Example Execution (Synthetic Spillover Data)
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Simulating an imbalanced zoonotic dataset:
    # 1,500 locations, 15 bioclimatic/ecological features, 5% positive spillover rate
    X_sim, y_sim = make_classification(
        n_samples=1500,
        n_features=15,
        n_informative=8,
        n_redundant=4,
        weights=[0.95, 0.05],
        random_state=42
    )

    # Convert to DataFrame (mimicking your actual features: bio1, bio12, host_density, etc.)
    feature_names = [f"bio_{i+1}" for i in range(15)]
    X_df = pd.DataFrame(X_sim, columns=feature_names)
    y_series = pd.Series(y_sim)

    # Run Benchmark
    benchmark_df = evaluate_models(X_df, y_series)

    # Plot Comparison
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # PR-AUC Plot (Crucial for rare spillover events)
    ax[0].barh(benchmark_df["Model"], benchmark_df["PR_AUC_mean"], xerr=benchmark_df["PR_AUC_std"], color="#2b5c8f", capsize=4)
    ax[0].set_title("PR-AUC (Precision-Recall) by Model")
    ax[0].set_xlabel("Average Precision Score")
    
    # ROC-AUC Plot
    ax[1].barh(benchmark_df["Model"], benchmark_df["ROC_AUC_mean"], xerr=benchmark_df["ROC_AUC_std"], color="#3b8f62", capsize=4)
    ax[1].set_title("ROC-AUC by Model")
    ax[1].set_xlabel("Area Under ROC Curve")
    
    plt.tight_layout()
    plt.show()
