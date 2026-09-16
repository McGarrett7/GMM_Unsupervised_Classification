import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


# ======================================================
# Tạo thư mục lưu kết quả
# ======================================================

os.makedirs("../data", exist_ok=True)
os.makedirs("../results/tables", exist_ok=True)
os.makedirs("../results/figures", exist_ok=True)


# ======================================================
# Đọc dữ liệu từ sklearn
# ======================================================

dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

feature_names = dataset.feature_names
target_names = dataset.target_names


# ======================================================
# Chuyển sang DataFrame
# ======================================================

df = pd.DataFrame(X, columns=feature_names)

df["target"] = y

label_map = {
    0: "Malignant",
    1: "Benign"
}

df["Diagnosis"] = df["target"].map(label_map)


# ======================================================
# Lưu bộ dữ liệu
# ======================================================

df.to_csv("../data/breast_cancer.csv", index=False)

print("Đã lưu dữ liệu:")
print("../data/breast_cancer.csv")


# ======================================================
# Thông tin tổng quan
# ======================================================

print("=" * 60)

print("KÍCH THƯỚC DỮ LIỆU")
print(df.shape)
print("=" * 60)


print("THÔNG TIN CỘT")
print(df.info())
print("=" * 60)


print("5 DÒNG ĐẦU")
print(df.head())
print("=" * 60)


# ======================================================
# Kiểm tra dữ liệu thiếu
# ======================================================

missing = df.isnull().sum()
print("GIÁ TRỊ THIẾU")
print(missing)
missing.to_csv("../results/tables/missing_values.csv")
print("=" * 60)


# ======================================================
# Thống kê mô tả
# ======================================================

description = df.describe()
description.to_csv("../results/tables/descriptive_statistics.csv")
print(description)
print("=" * 60)


# ======================================================
# Thống kê số lượng từng lớp
# ======================================================

class_distribution = df["Diagnosis"].value_counts()
print(class_distribution)
class_distribution.to_csv("../results/tables/class_distribution.csv")
print("=" * 60)


# ======================================================
# Vẽ biểu đồ phân bố lớp
# ======================================================

plt.figure(figsize=(6,5))

class_distribution.plot(
    kind="bar"
)

plt.title("Distribution of Breast Cancer Classes")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(
    "../results/figures/class_distribution.png",
    dpi=300
)

plt.show()


# ======================================================
# Lưu thông tin dataset
# ======================================================

summary = pd.DataFrame({

    "Metric": [
        "Number of Samples",
        "Number of Features",
        "Benign",
        "Malignant"
    ],

    "Value": [
        len(df),
        len(feature_names),
        class_distribution["Benign"],
        class_distribution["Malignant"]
    ]

})

summary.to_csv("../results/tables/dataset_summary.csv", index=False)

print(summary)
print("=" * 60)
