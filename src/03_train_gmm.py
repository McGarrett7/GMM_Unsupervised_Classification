import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

# =====================================================
# Đọc dữ liệu
# =====================================================

DATA_PATH = "../results/tables/processed_data.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("LOAD PREPROCESSED DATA")
print("=" * 60)

print(df.shape)

# =====================================================
# Tách X và y
# =====================================================

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

print("Number of samples :", X.shape[0])
print("Number of features:", X.shape[1])

# =====================================================
# Khởi tạo mô hình
# =====================================================

gmm = GaussianMixture(
    n_components=2,
    covariance_type="full",
    max_iter=500,
    tol=1e-3,
    n_init=20,
    random_state=42
)

print("Training Gaussian Mixture Model ...")

# =====================================================
# Huấn luyện
# =====================================================

gmm.fit(X)
print()
print("Training completed.")

# =====================================================
# Kiểm tra hội tụ
# =====================================================

print()
print("Converged :", gmm.converged_)
print("Iterations:", gmm.n_iter_)
print("Lower Bound:", gmm.lower_bound_)

# =====================================================
# Log Likelihood
# =====================================================

average_loglikelihood = gmm.score(X)
total_loglikelihood = average_loglikelihood * len(X)
print("Average LogLikelihood")
print(average_loglikelihood)
print("Total LogLikelihood")
print(total_loglikelihood)

# =====================================================
# Mixing Coefficient
# =====================================================

weights = pd.DataFrame({
    "Cluster": [1,2],
    "Weight": gmm.weights_
})

weights.to_csv("../results/tables/weights.csv", index=False)

print(weights)

# =====================================================
# Mean Vector
# =====================================================

means = pd.DataFrame(gmm.means_, columns=X.columns)
means.index = ["Cluster_1", "Cluster_2"]
means.to_csv("../results/tables/means.csv")
print(means.iloc[:,0:4])

# =====================================================
# Covariance Matrix
# =====================================================

cov1 = pd.DataFrame(gmm.covariances_[0], columns=X.columns, index=X.columns)
cov2 = pd.DataFrame(gmm.covariances_[1], columns=X.columns, index=X.columns)
cov1.to_csv("../results/tables/covariance_cluster_1.csv")
cov2.to_csv("../results/tables/covariance_cluster_2.csv")
print("Covariance matrices saved.")

# =====================================================
# Predict Cluster
# =====================================================

cluster = gmm.predict(X)

cluster_result = pd.DataFrame({
    "Actual": y,
    "Cluster": cluster
})

cluster_result.to_csv("../results/tables/cluster_result.csv", index=False)
print()
print(cluster_result.head())

# =====================================================
# Posterior Probability
# =====================================================

# posterior = gmm.predict_proba(X)
#
# posterior = pd.DataFrame(
#     posterior,
#     columns=["Cluster1_Probability", "Cluster2_Probability"]
# )
#
# posterior.to_csv("../results/tables/posterior_probability.csv", index=False)
#
# print(posterior.head())

# ==========================
# Dự đoán cụm
# ==========================
cluster = gmm.predict(X)

# ==========================
# Thống kê số lượng mẫu mỗi cụm
# ==========================
cluster_count = (
    pd.Series(cluster)
    .value_counts()
    .sort_index()
)

cluster_table = pd.DataFrame({
    "Cluster": cluster_count.index,
    "Number of Samples": cluster_count.values
})

print("\nCluster Distribution")
print(cluster_table)

cluster_table.to_csv(
    "../results/tables/cluster_distribution.csv",
    index=False
)

# ==========================
# Vẽ biểu đồ số lượng mẫu
# ==========================
plt.figure(figsize=(6,4))

bars = plt.bar(
    cluster_table["Cluster"].astype(str),
    cluster_table["Number of Samples"]
)

# Hiển thị số trên đầu cột
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 3,
        f"{int(height)}",
        ha="center"
    )

plt.xlabel("Cluster")
plt.ylabel("Number of Samples")
plt.title("Number of Samples in Each Cluster")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "../results/figures/cluster_distribution.png",
    dpi=300
)

plt.show()

# ==========================
# Posterior Probability
# ==========================
posterior = gmm.predict_proba(X)

posterior = pd.DataFrame(
    posterior,
    columns=[
        "Cluster1_Probability",
        "Cluster2_Probability"
    ]
)

posterior.to_csv(
    "../results/tables/posterior_probability.csv",
    index=False
)

print("\nPosterior Probability")
print(posterior.head())

# =====================================================
# Lưu Model
# =====================================================

joblib.dump(gmm,"../results/models/gmm_model.pkl")
print()
print("Model saved.")

# =====================================================
# Training Summary
# =====================================================

summary = pd.DataFrame({

    "Metric":[
        "Number of Samples",
        "Number of Features",
        "Number of Components",
        "Covariance Type",
        "Converged",
        "Iterations",
        "Average LogLikelihood",
        "Total LogLikelihood"
    ],

    "Value":[
        len(X),
        X.shape[1],
        2,
        "full",
        gmm.converged_,
        gmm.n_iter_,
        average_loglikelihood,
        total_loglikelihood
    ]
})

summary.to_csv("../results/tables/training_summary.csv", index=False)
print(summary)
print("="*60)