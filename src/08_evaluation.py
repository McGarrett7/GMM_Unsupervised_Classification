import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# =====================================================
# Tải dữ liệu
# =====================================================

print("=" * 70)
print("LOAD DATA")
print("=" * 70)

df = pd.read_csv("../results/tables/processed_data.csv")

X = df.drop(columns=["diagnosis"])

y_true = df["diagnosis"].values

print("Dataset Shape :", df.shape)
print("Feature Shape :", X.shape)

# =====================================================
# Tải mô hình GMM
# =====================================================

print()
print("=" * 70)
print("LOAD MODEL")
print("=" * 70)

gmm = joblib.load("../results/models/gmm_model.pkl")

print("Model Loaded Successfully")

# =====================================================
# Dự đoán cụm
# =====================================================

cluster = gmm.predict(X)
posterior = gmm.predict_proba(X)
confidence = posterior.max(axis=1)

# =====================================================
# Thuật toán Hungarian
# =====================================================

print()
print("=" * 70)
print("HUNGARIAN MATCHING")
print("=" * 70)

cm = confusion_matrix(y_true, cluster)
row_ind, col_ind = linear_sum_assignment(-cm)
mapping = {}

for r, c in zip(row_ind, col_ind):
    mapping[c] = r

print("Cluster Mapping")
print(mapping)

# =====================================================
# Chuyển đổi Cụm -> Nhãn
# =====================================================

y_pred = np.array(
    [
        mapping[c]
        for c in cluster
    ]
)

# =====================================================
# Độ chính xác
# =====================================================

accuracy = accuracy_score(y_true, y_pred)

print("Accuracy = {:.4f}".format(accuracy))

# =====================================================
# Ma trận nhầm lẫn
# =====================================================

cm_final = confusion_matrix(y_true, y_pred)

cm_df = pd.DataFrame(
    cm_final,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

cm_df.to_csv("../results/tables/confusion_matrix.csv")

print(cm_df)

# =====================================================
# Vẽ ma trận nhầm lẫn
# =====================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_final,
    display_labels=[
        "Class 0",
        "Class 1"
    ]
)

fig, ax = plt.subplots(
    figsize=(6, 6)
)

disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("../results/figures/confusion_matrix.png", dpi=300)
plt.show()

# =====================================================
# Lưu kết quả dự đoán
# =====================================================

prediction_df = pd.DataFrame({
    "Actual": y_true,
    "Cluster": cluster,
    "Prediction": y_pred,
    "Confidence": confidence
})

prediction_df.to_csv("../results/tables/prediction_result.csv", index=False)

# =====================================================
# Tổng hợp
# =====================================================

summary = pd.DataFrame({
    "Metric": [
        "Number of Samples",
        "Accuracy"
    ],

    "Value": [
        len(y_true),
        accuracy
    ]

})

summary.to_csv("../results/tables/evaluation_summary.csv", index=False)

print(summary)
print("=" * 70)
print("CLUSTERING METRICS")
print("=" * 70)

ari = adjusted_rand_score(y_true,cluster)
print()
print("ARI =", ari)

nmi = normalized_mutual_info_score(y_true,cluster)
print("NMI =", nmi)

silhouette = silhouette_score(X,cluster)
print("Silhouette =", silhouette)

ch = calinski_harabasz_score(X,cluster)
print("CH =", ch)

db = davies_bouldin_score(X,cluster)
print("DB =", db)
log_likelihood = gmm.score(X)

print()
print("Average Log-Likelihood =",log_likelihood)

aic = gmm.aic(X)
print("AIC =",aic)

bic = gmm.bic(X)
print("BIC =",bic)

mean_confidence = confidence.mean()
print()
print("Mean Posterior =",mean_confidence)

posterior_df = pd.read_csv("../results/tables/posterior_analysis.csv")
mean_entropy = posterior_df["Entropy"].mean()

print("Mean Entropy =",mean_entropy)
mean_uncertainty = posterior_df["Uncertainty"].mean()

print("Mean Uncertainty =",mean_uncertainty)

cluster_metric = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "ARI",
        "NMI",
        "Silhouette",
        "Calinski-Harabasz",
        "Davies-Bouldin",
        "Average LogLikelihood",
        "AIC",
        "BIC",
        "Mean Posterior",
        "Mean Entropy",
        "Mean Uncertainty"
    ],

    "Value": [
        accuracy,
        ari,
        nmi,
        silhouette,
        ch,
        db,
        log_likelihood,
        aic,
        bic,
        mean_confidence,
        mean_entropy,
        mean_uncertainty
    ]
})

cluster_metric.to_csv("../results/tables/cluster_metrics.csv",index=False)

print()

print(cluster_metric)

plt.figure(figsize=(9, 5))

metric_name = [
    "Accuracy",
    "ARI",
    "NMI",
    "Silhouette"
]

metric_value = [
    accuracy,
    ari,
    nmi,
    silhouette
]

plt.bar(
    metric_name,
    metric_value
)

plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Evaluation Metrics")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("../results/figures/metric_comparison.png",dpi=300)
plt.show()
print()
print("=" * 70)


# =====================================================
# Các chỉ số phân loại
# =====================================================

print()

print("="*70)

print("CLASSIFICATION METRICS")

print("="*70)

precision = precision_score(y_true, y_pred)

print("Precision =", precision)

recall = recall_score(y_true, y_pred)

print("Recall =",recall)

f1 = f1_score(y_true, y_pred)

print("F1 =", f1)

report = classification_report(y_true,y_pred,output_dict=True)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv("../results/tables/classification_report.csv")

print()

print(report_df)

positive_label = 1

positive_cluster = {v: k for k, v in mapping.items()}[positive_label]

y_score = posterior[:, positive_cluster]

fpr, tpr, thresholds = roc_curve(y_true,y_score)

roc_auc = auc(fpr,tpr)

roc_df = pd.DataFrame({"FPR": fpr,"TPR": tpr,"Threshold": thresholds})

roc_df.to_csv("../results/tables/roc_curve.csv",index=False)

plt.figure(figsize=(7,7))

plt.plot(fpr,tpr,label=f"AUC = {roc_auc:.4f}")

plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("../results/figures/roc_curve.png",dpi=300)

plt.show()


# =====================================================
# Đường cong Precision-Recall
# =====================================================

precision_curve, recall_curve, thresholds = (
    precision_recall_curve(y_true, y_score)
)

pr_df = pd.DataFrame({
    "Precision": precision_curve[:-1],
    "Recall": recall_curve[:-1],
    "Threshold": thresholds
})

pr_df.to_csv("../results/tables/pr_curve.csv",index=False)

plt.figure(figsize=(7,7))

plt.plot(recall_curve,precision_curve)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.grid(True)

plt.tight_layout()

plt.savefig("../results/figures/precision_recall_curve.png",dpi=300)

plt.show()


# =====================================================
# Tổng hợp
# =====================================================

evaluation = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC AUC"
    ],

    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]

})

evaluation.to_csv("../results/tables/classification_metrics.csv", index=False)
print()
print(evaluation)
print()
print("="*70)


print()
print("=" * 70)
print("MODEL QUALITY ASSESSMENT")
print("=" * 70)

def evaluate_quality(score):
    if score >= 0.95:
        return "Excellent"

    elif score >= 0.90:
        return "Very Good"

    elif score >= 0.80:
        return "Good"

    elif score >= 0.70:
        return "Acceptable"

    else:
        return "Poor"


quality_table = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "ARI",
        "NMI",
        "Silhouette"
    ],

    "Score": [
        accuracy,
        ari,
        nmi,
        silhouette
    ]
})

quality_table["Quality"] = quality_table["Score"].apply(evaluate_quality)

quality_table.to_csv("../results/tables/model_quality.csv", index=False)

print()
print(quality_table)

probability_metrics = pd.DataFrame({

    "Metric": [
        "Average Log-Likelihood",
        "AIC",
        "BIC",
        "Mean Posterior",
        "Mean Entropy",
        "Mean Uncertainty"
    ],

    "Value": [
        log_likelihood,
        aic,
        bic,
        mean_confidence,
        mean_entropy,
        mean_uncertainty
    ]

})

probability_metrics.to_csv("../results/tables/probability_metrics.csv", index=False)

print(probability_metrics)

final_report = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC AUC",
        "ARI",
        "NMI",
        "Silhouette",
        "Calinski-Harabasz",
        "Davies-Bouldin",
        "Average Log-Likelihood",
        "AIC",
        "BIC",
        "Mean Posterior",
        "Mean Entropy",
        "Mean Uncertainty"
    ],

    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        ari,
        nmi,
        silhouette,
        ch,
        db,
        log_likelihood,
        aic,
        bic,
        mean_confidence,
        mean_entropy,
        mean_uncertainty
    ]
})

final_report.to_csv("../results/tables/final_evaluation_report.csv", index=False)

print()
print(final_report)

overall_score = np.mean([
    accuracy,
    ari,
    nmi,
    silhouette,
    mean_confidence
])

print()
print("Overall Score = {:.4f}".format(overall_score))

overall_level = evaluate_quality(overall_score)

print("Overall Rating :", overall_level)

summary = pd.DataFrame({

    "Item": [
        "Dataset",
        "Samples",
        "Clusters",
        "Overall Score",
        "Overall Rating"
    ],

    "Value": [
        "Breast Cancer Wisconsin",
        len(df),
        gmm.n_components,
        overall_score,
        overall_level
    ]

})

summary.to_csv("../results/tables/summary_report.csv", index=False)

print()
print(summary)

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Accuracy            : {accuracy:.4f}")
print(f"ARI                 : {ari:.4f}")
print(f"NMI                 : {nmi:.4f}")
print(f"Silhouette          : {silhouette:.4f}")
print(f"Average Posterior   : {mean_confidence:.4f}")
print(f"Average Entropy     : {mean_entropy:.4f}")
print(f"Average Uncertainty : {mean_uncertainty:.4f}")
print(f"AIC                 : {aic:.2f}")
print(f"BIC                 : {bic:.2f}")
print(f"Overall Score       : {overall_score:.4f}")
print(f"Overall Rating      : {overall_level}")

print()
print("=" * 70)
