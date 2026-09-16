import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Tải dữ liệu
# =====================================================

print("=" * 70)

df = pd.read_csv("../results/tables/processed_data.csv")
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

print("Dataset Shape :", df.shape)
print("Feature Shape :", X.shape)

# =====================================================
# Tải mô hình
# =====================================================

print()
print("=" * 70)
print("LOAD GMM MODEL")
print("=" * 70)

gmm = joblib.load("../results/models/gmm_model.pkl")

print("Model Loaded Successfully")

# =====================================================
# Dự đoán cụm
# =====================================================

cluster = gmm.predict(X)
posterior = gmm.predict_proba(X)
confidence = posterior.max(axis=1)
uncertainty = 1.0 - confidence

# =====================================================
# Tạo bảng kết quả
# =====================================================

posterior_df = pd.DataFrame({
    "Actual_Label": y,
    "Predicted_Cluster": cluster,
    "Posterior_Cluster0": posterior[:, 0],
    "Posterior_Cluster1": posterior[:, 1],
    "Confidence": confidence,
    "Uncertainty": uncertainty

})

posterior_df.to_csv("../results/tables/posterior_analysis.csv",index=False)

print()
print("Posterior table saved.")

# =====================================================
# Thống kê tổng hợp
# =====================================================

summary = pd.DataFrame({

    "Statistic": [
        "Minimum Confidence",
        "Maximum Confidence",
        "Mean Confidence",
        "Median Confidence",
        "Std Confidence",
        "Minimum Uncertainty",
        "Maximum Uncertainty",
        "Mean Uncertainty"
    ],

    "Value": [
        confidence.min(),
        confidence.max(),
        confidence.mean(),
        np.median(confidence),
        confidence.std(),
        uncertainty.min(),
        uncertainty.max(),
        uncertainty.mean()
    ]

})

summary.to_csv(
    "../results/tables/posterior_summary.csv",
    index=False
)

print()
print(summary)

# =====================================================
# Thống kê theo cụm
# =====================================================

cluster_summary = posterior_df.groupby(
    "Predicted_Cluster"

).agg({
    "Confidence": [
        "count",
        "mean",
        "std",
        "min",
        "max"
    ],

    "Uncertainty": [
        "mean",
        "std",
        "min",
        "max"
    ]

})

cluster_summary.to_csv(
    "../results/tables/cluster_probability_summary.csv"
)

print()
print(cluster_summary)

# =====================================================
# Top 10 độ tin cậy cao nhất
# =====================================================

top_confidence = posterior_df.sort_values(
    by="Confidence",
    ascending=False
).head(10)

top_confidence.to_csv(
    "../results/tables/top10_confidence.csv",
    index=False
)

# =====================================================
# Top 10 độ bất định cao nhất
# =====================================================

top_uncertain = posterior_df.sort_values(
    by="Uncertainty",
    ascending=False
).head(10)

top_uncertain.to_csv(
    "../results/tables/top10_uncertainty.csv",
    index=False
)

print("=" * 70)
print("PART 1 COMPLETED")
print("=" * 70)

# =====================================================
# Biểu đồ phân phối xác suất hậu nghiệm
# =====================================================

print("="*70)

print("DRAW POSTERIOR HISTOGRAM")

print("="*70)


plt.figure(
    figsize=(8,6)
)

plt.hist(
    posterior[:,0],
    bins=30,
    alpha=0.7,
    label="Cluster 0"

)

plt.hist(
    posterior[:,1],
    bins=30,
    alpha=0.7,
    label="Cluster 1"
)

plt.xlabel(
    "Posterior Probability"
)
plt.ylabel(
    "Frequency"
)
plt.title(
    "Posterior Probability Distribution"
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "../results/figures/posterior_probability_histogram.png",
    dpi=300
)

plt.show()

# =====================================================
# Biểu đồ phân phối độ tin cậy
# =====================================================

plt.figure(
    figsize=(8,6)
)

plt.hist(
    confidence,
    bins=30
)

plt.xlabel(
    "Confidence"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Confidence Distribution"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "../results/figures/confidence_histogram.png",
    dpi=300
)

plt.show()

# =====================================================
# Biểu đồ phân phối độ bất định
# =====================================================

plt.figure(
    figsize=(8,6)
)

plt.hist(
    uncertainty,
    bins=30
)

plt.xlabel(
    "Uncertainty"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Uncertainty Distribution"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "../results/figures/uncertainty_histogram.png",
    dpi=300
)

plt.show()


# =====================================================
# Biểu đồ hộp độ tin cậy
# =====================================================

plt.figure(figsize=(5,6))

plt.boxplot(
    confidence,
    vert=True
)

plt.ylabel(
    "Confidence"
)

plt.title(
    "Confidence Boxplot"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "../results/figures/confidence_boxplot.png",
    dpi=300
)

plt.show()

# =====================================================
# Biểu đồ hộp độ bất định
# =====================================================

plt.figure(figsize=(5,6))

plt.boxplot(uncertainty,vert=True)

plt.ylabel("Uncertainty")

plt.title("Uncertainty Boxplot")

plt.grid(True)

plt.tight_layout()

plt.savefig("../results/figures/uncertainty_boxplot.png", dpi=300)

plt.show()


# =====================================================
# Entropy
# =====================================================

epsilon = 1e-12

entropy = -np.sum( posterior * np.log(posterior + epsilon), axis=1)

posterior_df["Entropy"] = entropy

posterior_df.to_csv("../results/tables/posterior_analysis.csv", index=False)

entropy_summary = pd.DataFrame({

    "Statistic":[
        "Minimum",
        "Maximum",
        "Mean",
        "Median",
        "Std"
    ],

    "Value":[
        entropy.min(),
        entropy.max(),
        entropy.mean(),
        np.median(entropy),
        entropy.std()
    ]
})

entropy_summary.to_csv("../results/tables/entropy_summary.csv", index=False)

print(entropy_summary)

plt.figure(figsize=(8,6))

plt.hist( entropy,bins=30)

plt.xlabel("Entropy")

plt.ylabel("Frequency")

plt.title("Posterior Entropy")

plt.grid(True)

plt.tight_layout()

plt.savefig("../results/figures/entropy_histogram.png", dpi=300)

plt.show()


# =====================================================
# Ngưỡng bất định
# =====================================================

print()

print("="*70)

print("UNCERTAINTY ANALYSIS")

print("="*70)

CONFIDENCE_THRESHOLD = 0.60

uncertain_samples = posterior_df[
    posterior_df["Confidence"] < CONFIDENCE_THRESHOLD
].copy()

print()

print("Number of uncertain samples:",
      len(uncertain_samples))


print("Percentage: {:.2f}%".format(

    len(uncertain_samples) /
    len(posterior_df) * 100

))
uncertain_samples.to_csv(

    "../results/tables/uncertain_samples.csv",

    index=False

)

# =====================================================
# Top 20 mẫu bất định nhất
# =====================================================

top20 = posterior_df.sort_values(

    by="Confidence",

    ascending=True

).head(20)

top20.to_csv(

    "../results/tables/top20_uncertain_samples.csv",

    index=False

)

print(top20)

# =====================================================
# Phân tích đa ngưỡng
# =====================================================

thresholds = [

    0.55,

    0.60,

    0.70,

    0.80,

    0.90

]

result = []

for t in thresholds:

    n = np.sum(

        confidence < t

    )

    result.append([

        t,

        n,

        n/len(confidence)*100

    ])

threshold_table = pd.DataFrame(

    result,

    columns=[

        "Threshold",

        "Samples",

        "Percentage"

    ]

)

threshold_table.to_csv(

    "../results/tables/uncertainty_threshold_analysis.csv",

    index=False

)

print()

print(threshold_table)


# =====================================================
# Độ tin cậy vs Entropy
# =====================================================


plt.figure(

    figsize=(8,6)

)

plt.scatter(

    confidence,

    entropy,

    alpha=0.7,

    s=35

)

plt.xlabel(

    "Confidence"

)

plt.ylabel(

    "Entropy"

)

plt.title(

    "Confidence vs Entropy"

)

plt.grid(True)

plt.tight_layout()

plt.savefig(

    "../results/figures/confidence_entropy.png",

    dpi=300

)

plt.show()

# =====================================================
# Làm nổi bật các mẫu bất định bằng PCA
# =====================================================

pca = pd.read_csv(

    "../results/tables/pca_dataset.csv"

)
mask = confidence < CONFIDENCE_THRESHOLD

plt.figure(

    figsize=(9,7)

)

plt.scatter(

    pca.loc[~mask,"PC1"],

    pca.loc[~mask,"PC2"],

    s=25,

    alpha=0.5,

    label="Confident"

)

plt.scatter(
    pca.loc[mask,"PC1"],
    pca.loc[mask,"PC2"],
    s=80,
    marker="x",
    label="Uncertain"
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(
    "Uncertain Samples"
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/figures/uncertain_samples_pca.png", dpi=300)
plt.show()

# =====================================================
# Xếp hạng Entropy
# =====================================================

entropy_rank = posterior_df.sort_values(by="Entropy", ascending=False)
entropy_rank.to_csv("../results/tables/entropy_ranking.csv", index=False)

# =====================================================
# Tương quan
# =====================================================

corr = posterior_df[["Confidence", "Entropy", "Uncertainty"]].corr()
corr.to_csv("../results/tables/correlation_matrix.csv")
print(corr)

print()
print("=" * 70)
print("GROUND TRUTH ANALYSIS")
print("=" * 70)

# Gắn lại nhãn thật
posterior_df["Diagnosis"] = y.values

diagnosis_summary = (
    posterior_df
    .groupby("Diagnosis")
    .agg(
        Sample_Count=("Diagnosis", "count"),
        Mean_Confidence=("Confidence", "mean"),
        Std_Confidence=("Confidence", "std"),
        Mean_Entropy=("Entropy", "mean"),
        Mean_Uncertainty=("Uncertainty", "mean")
    )
)

diagnosis_summary.to_csv(
    "../results/tables/diagnosis_probability_summary.csv"
)

print(diagnosis_summary)

# =====================================================
# Thống kê mẫu bất định theo lớp
# =====================================================

uncertain_by_class = (
    uncertain_samples.groupby("Actual_Label").size().reset_index(name="Count")
)

uncertain_by_class.to_csv("../results/tables/uncertain_by_class.csv", index=False)

print()

print(uncertain_by_class)


# =====================================================
# Biểu đồ Confidence theo lớp
# =====================================================

plt.figure(figsize=(8,6))

plt.boxplot(

    [

        posterior_df[
            posterior_df["Diagnosis"]==0
        ]["Confidence"],

        posterior_df[
            posterior_df["Diagnosis"]==1
        ]["Confidence"]

    ],

    labels=["Malignant","Benign"]

)

plt.ylabel("Confidence")
plt.title("Confidence by Diagnosis")
plt.grid(True)
plt.tight_layout()

plt.savefig("../results/figures/confidence_by_class.png",dpi=300)

plt.show()


# =====================================================
# Entropy theo lớp
# =====================================================

plt.figure(figsize=(8,6))

plt.boxplot(
    [
        posterior_df[
            posterior_df["Diagnosis"]==0
        ]["Entropy"],

        posterior_df[
            posterior_df["Diagnosis"]==1
        ]["Entropy"]
    ],

    labels=["Malignant","Benign"]
)

plt.ylabel("Entropy")
plt.title("Entropy by Diagnosis")
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/figures/entropy_by_class.png", dpi=300)
plt.show()

# =====================================================
# Phân loại mức độ tin cậy
# =====================================================

def confidence_level(c):
    if c >= 0.95:
        return "Very High"

    elif c >= 0.80:
        return "High"

    elif c >= 0.60:
        return "Medium"

    else:
        return "Low"

posterior_df["Confidence_Level"] = (posterior_df["Confidence"].apply(confidence_level))


level_summary = (posterior_df.groupby("Confidence_Level").size().reset_index(name="Count"))

level_summary.to_csv(
    "../results/tables/confidence_level_summary.csv",
    index=False
)

print(level_summary)
plt.figure(figsize=(7,5))
plt.bar(
    level_summary["Confidence_Level"],
    level_summary["Count"]
)

plt.ylabel("Number of Samples")
plt.title("Confidence Level Distribution")

plt.savefig(
    "../results/figures/confidence_level_distribution.png",
    dpi=300
)

plt.show()

# =====================================================
# Sinh báo cáo tổng hợp
# =====================================================

report = pd.DataFrame({

    "Metric":[
        "Total Samples",
        "Average Confidence",
        "Average Uncertainty",
        "Average Entropy",
        "Maximum Confidence",
        "Minimum Confidence",
        "High Uncertainty Samples"

    ],

    "Value":[
        len(posterior_df),
        posterior_df["Confidence"].mean(),
        posterior_df["Uncertainty"].mean(),
        posterior_df["Entropy"].mean(),
        posterior_df["Confidence"].max(),
        posterior_df["Confidence"].min(),
        len(uncertain_samples)
    ]
})

report.to_csv(
    "../results/tables/posterior_final_report.csv",
    index=False

)

print(report)
print("="*70)