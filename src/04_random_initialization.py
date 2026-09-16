import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

# =====================================================
# Đọc dữ liệu
# =====================================================

df = pd.read_csv("../results/tables/processed_data.csv")
X = df.drop(columns=["diagnosis"])

# =====================================================
# Danh sách lưu kết quả
# =====================================================

results = []

# =====================================================
# Thực hiện 20 lần khởi tạo
# =====================================================

print("=" * 60)

for seed in range(20):

    print(f"Training Seed = {seed}")

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=1,
        random_state=seed,
        max_iter=500,
        tol=1e-3
    )

    gmm.fit(X)
    avg_ll = gmm.score(X)
    total_ll = avg_ll * len(X)

    results.append({
        "Seed": seed,
        "Average_LogLikelihood": avg_ll,
        "Total_LogLikelihood": total_ll,
        "Iterations": gmm.n_iter_,
        "Converged": gmm.converged_
    })

# =====================================================
# DataFrame
# =====================================================

result_df = pd.DataFrame(results)
result_df.to_csv("../results/tables/random_initialization.csv",index=False)
print()
print(result_df)

# =====================================================
# Thống kê
# =====================================================

summary = pd.DataFrame({
    "Metric":[
        "Maximum LogLikelihood",
        "Minimum LogLikelihood",
        "Mean LogLikelihood",
        "Std LogLikelihood"
    ],

    "Value":[
        result_df["Average_LogLikelihood"].max(),
        result_df["Average_LogLikelihood"].min(),
        result_df["Average_LogLikelihood"].mean(),
        result_df["Average_LogLikelihood"].std()
    ]
})

summary.to_csv("../results/tables/random_initialization_summary.csv",index=False)
print()
print(summary)

# =====================================================
# Vẽ đồ thị LogLikelihood
# =====================================================

plt.figure(figsize=(10,6))

plt.plot(
    result_df["Seed"],
    result_df["Average_LogLikelihood"],
    marker="o",
    linewidth=2
)

plt.title("Log-Likelihood under Different Random Initializations")
plt.xlabel("Random Seed")
plt.ylabel("Average Log-Likelihood")
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/figures/random_initialization.png", dpi=300)
plt.show()

# =====================================================
# Boxplot
# =====================================================

plt.figure(figsize=(5,6))
plt.boxplot(result_df["Average_LogLikelihood"])
plt.ylabel("Average Log-Likelihood")
plt.title("Distribution of Log-Likelihood")
plt.tight_layout()
plt.savefig("../results/figures/random_initialization_boxplot.png",dpi=300)
plt.show()

# =====================================================
# Histogram
# =====================================================

plt.figure(figsize=(8,5))

plt.hist(
    result_df["Average_LogLikelihood"],
    bins=8
)

plt.xlabel("Average Log-Likelihood")
plt.ylabel("Frequency")
plt.title("Histogram of Log-Likelihood")

plt.tight_layout()

plt.savefig(
    "../results/figures/random_initialization_histogram.png",
    dpi=300
)

plt.show()
print("=" * 60)
print(result_df["Average_LogLikelihood"])