import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

# =====================================================
# Đọc dữ liệu
# =====================================================

DATA_PATH = "../results/tables/processed_data.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["diagnosis"])

print("=" * 60)
print("MODEL SELECTION")
print("=" * 60)

print("Dataset Shape:", X.shape)

# =====================================================
# Danh sách lưu kết quả
# =====================================================

results = []

# =====================================================
# Thử các giá trị K
# =====================================================

for k in range(1, 7):

    print(f"\nTraining GMM with {k} component(s)...")

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=42,
        n_init=20,
        max_iter=500,
        tol=1e-3
    )

    gmm.fit(X)
    avg_ll = gmm.score(X)
    total_ll = avg_ll * len(X)
    aic = gmm.aic(X)
    bic = gmm.bic(X)
    results.append({
        "Components": k,
        "Average_LogLikelihood": avg_ll,
        "Total_LogLikelihood": total_ll,
        "AIC": aic,
        "BIC": bic,
        "Iterations": gmm.n_iter_,
        "Converged": gmm.converged_
    })

# =====================================================
# DataFrame
# =====================================================

result_df = pd.DataFrame(results)
result_df.to_csv("../results/tables/model_selection.csv", index=False)
print(result_df)

# =====================================================
# Giá trị tối ưu
# =====================================================

best_bic = result_df.loc[result_df["BIC"].idxmin()]
best_aic = result_df.loc[result_df["AIC"].idxmin()]
summary = pd.DataFrame({

    "Criterion": [
        "Best BIC",
        "Best AIC"
    ],

    "Components": [
        int(best_bic["Components"]),
        int(best_aic["Components"])
    ],

    "Value": [
        best_bic["BIC"],
        best_aic["AIC"]
    ]

})

summary.to_csv("../results/tables/model_selection_summary.csv", index=False)

print("\nBest Model Summary:")
print(summary)

# =====================================================
# Plot AIC
# =====================================================

plt.figure(figsize=(8,5))
plt.plot(result_df["Components"], result_df["AIC"], marker="o", linewidth=2)
plt.xticks(result_df["Components"])
plt.xlabel("Number of Components")
plt.ylabel("AIC")
plt.title("AIC vs Number of Components")
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/figures/aic_curve.png", dpi=300)
plt.show()

# =====================================================
# Plot BIC
# =====================================================

plt.figure(figsize=(8,5))
plt.plot(result_df["Components"], result_df["BIC"], marker="o", linewidth=2)
plt.xticks(result_df["Components"])
plt.xlabel("Number of Components")
plt.ylabel("BIC")
plt.title("BIC vs Number of Components")
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/figures/bic_curve.png", dpi=300)
plt.show()

# =====================================================
# Plot Log-Likelihood
# =====================================================

plt.figure(figsize=(8,5))
plt.plot(result_df["Components"], result_df["Average_LogLikelihood"], marker="o", linewidth=2)
plt.xticks(result_df["Components"])
plt.xlabel("Number of Components")
plt.ylabel("Average Log-Likelihood")
plt.title("Log-Likelihood vs Number of Components")
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/figures/loglikelihood_curve.png", dpi=300)
plt.show()
print("\nModel Selection Completed.")