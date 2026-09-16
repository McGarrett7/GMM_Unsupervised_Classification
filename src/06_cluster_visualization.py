import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

# ==========================================================
# Tải dữ liệu
# ==========================================================

print("=" * 70)

df = pd.read_csv("../results/tables/processed_data.csv")

print(df.head())
print()
print("Dataset Shape :", df.shape)

# ==========================================================
# Đặc trưng / Nhãn
# ==========================================================

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]
print()
print("Feature Shape :", X.shape)
print("Label Shape :", y.shape)

# ==========================================================
# Tải mô hình GMM đã huấn luyện
# ==========================================================

print()
print("=" * 70)
print("LOAD GMM MODEL")
print("=" * 70)
gmm = joblib.load("../results/models/gmm_model.pkl")
print("Model Loaded Successfully")

# ==========================================================
# Dự đoán cụm
# ==========================================================

cluster = gmm.predict(X)
posterior = gmm.predict_proba(X)
confidence = posterior.max(axis=1)

print()
print("Cluster Distribution")
print(pd.Series(cluster).value_counts())

# ==========================================================
# PCA
# ==========================================================

print()
print("=" * 70)
print("RUN PCA")
print("=" * 70)

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X)
print()
print("Explained Variance Ratio")
print(pca.explained_variance_ratio_)
print()
print(
    "Total Explained Variance :",
    np.sum(
        pca.explained_variance_ratio_
    )

)

# ==========================================================
# Lưu tập dữ liệu PCA
# ==========================================================

pca_df = pd.DataFrame({
    "PC1": X_pca[:,0],
    "PC2": X_pca[:,1],
    "Diagnosis": y,
    "Cluster": cluster,
    "Confidence": confidence
})

pca_df.to_csv(
    "../results/tables/pca_dataset.csv",
    index=False
)

print()
print("PCA Dataset Saved")

# ==========================================================
# Biểu đồ 1
# Nhãn thực tế (Ground Truth)
# ==========================================================

print()
print("=" * 70)
print("DRAW GROUND TRUTH")
print("=" * 70)
plt.figure(
    figsize=(8,6)
)

plt.scatter(
    X_pca[y==0,0],
    X_pca[y==0,1],
    s=45,
    alpha=0.7,
    label="Malignant"
)

plt.scatter(
    X_pca[y==1,0],
    X_pca[y==1,1],
    s=45,
    alpha=0.7,
    label="Benign"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Ground Truth Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "../results/figures/pca_ground_truth.png",
    dpi=300
)

plt.show()

# ==========================================================
# Biểu đồ 2
# Kết quả gom cụm
# ==========================================================

print()
print("=" * 70)
print("DRAW CLUSTER")
print("=" * 70)
plt.figure(
    figsize=(8,6)
)

plt.scatter(
    X_pca[cluster==0,0],
    X_pca[cluster==0,1],
    s=45,
    alpha=0.7,
    label="Cluster 0"
)

plt.scatter(
    X_pca[cluster==1,0],
    X_pca[cluster==1,1],
    s=45,
    alpha=0.7,
    label="Cluster 1"
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("Gaussian Mixture Model Clustering"
          )
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/figures/pca_cluster.png",dpi=300)
plt.show()

# ==========================================================
# Biểu đồ 3
# Kích thước cụm
# ==========================================================

cluster_size = (pd.Series(cluster).value_counts().sort_index())
cluster_size.to_csv("../results/tables/cluster_size.csv")
plt.figure(figsize=(6,5))

plt.bar(
    ["Cluster 0","Cluster 1"],
    cluster_size.values
)

for i,value in enumerate(cluster_size.values):

    plt.text(
        i,
        value+2,
        str(value),
        ha="center",
        fontsize=11

    )

plt.xlabel("Cluster")

plt.ylabel("Number of Samples")

plt.title("Cluster Size")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "../results/figures/cluster_size.png",
    dpi=300
)

plt.show()

# ==========================================================
# Tổng hợp
# ==========================================================

summary = pd.DataFrame({
    "Metric":[
        "Samples",
        "Features",
        "PC1 Variance",
        "PC2 Variance",
        "Total Variance"
    ],

    "Value":[
        len(df),
        X.shape[1],
        pca.explained_variance_ratio_[0],
        pca.explained_variance_ratio_[1],
        np.sum(
            pca.explained_variance_ratio_
        )

    ]

})

summary.to_csv(
    "../results/tables/pca_summary.csv",
    index=False

)

print(summary)

# Mean trong không gian gốc (30 chiều)
means = gmm.means_

# Chiếu xuống PCA
means_pca = pca.transform(means)
covariances_pca = []

for cov in gmm.covariances_:
    cov2 = pca.components_ @ cov @ pca.components_.T
    covariances_pca.append(cov2)

def draw_ellipse(position,
                 covariance,
                 ax,
                 color):

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(

        np.arctan2(
            eigenvectors[1,0],
            eigenvectors[0,0]
        )

    )

    width = 2*np.sqrt(eigenvalues[0])
    height = 2*np.sqrt(eigenvalues[1])

    ellipse = Ellipse(
        xy=position,width=width,height=height,angle=angle,edgecolor=color,facecolor="none",linewidth=2
    )

    ax.add_patch(ellipse)

fig, ax = plt.subplots(

    figsize=(8,6)

)

ax.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=cluster,
    s=35,
    alpha=0.6
)

colors = ["red", "blue"]

for i in range(2):
    draw_ellipse(
        means_pca[i],
        covariances_pca[i],
        ax,
        colors[i]
    )

    ax.scatter(
        means_pca[i,0],
        means_pca[i,1],
        marker="X",
        s=220,
        color=colors[i]
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Gaussian Ellipses")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "../results/figures/gaussian_ellipse.png",
    dpi=300
)

plt.show()

X_pca = pca.fit_transform(X)

means_pca = pca.transform(gmm.means_)

x_min = X_pca[:,0].min()-1
x_max = X_pca[:,0].max()+1

y_min = X_pca[:,1].min()-1
y_max = X_pca[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid_pca = np.c_[xx.ravel(), yy.ravel()]

grid_original = pca.inverse_transform(grid_pca)

Z = gmm.predict(grid_original)
Z = Z.reshape(xx.shape)

prob = gmm.predict_proba(grid_original)
posterior = prob[:,0]
posterior = posterior.reshape(xx.shape)
plt.figure(figsize=(9,7))

plt.contourf(xx, yy, Z, alpha=0.25, levels=2, cmap="coolwarm")

plt.scatter(X_pca[:,0], X_pca[:,1], c=cluster, s=30, cmap="coolwarm")

plt.contour(
    xx,
    yy,
    posterior,
    levels=[0.5],
    colors="black",
    linewidths=2

)

plt.scatter(
    means_pca[:,0],
    means_pca[:,1],
    marker="X",
    s=250,
    color="yellow",
    edgecolors="black"
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Decision Boundary of Gaussian Mixture Model")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "../results/figures/gmm_decision_boundary.png",
    dpi=300
)

plt.show()

log_density = gmm.score_samples(grid_original)
density = np.exp(log_density)
density = density.reshape(xx.shape)

plt.figure(figsize=(9,7))

plt.contourf(
    xx,
    yy,
    density,
    levels=30,
    cmap="viridis"
)

plt.colorbar(label="Probability Density")

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=cluster,
    s=20,
    alpha=0.55
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Gaussian Mixture Density")

plt.tight_layout()

plt.savefig("../results/figures/gmm_density.png", dpi=300)

plt.show()