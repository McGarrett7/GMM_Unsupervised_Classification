import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =====================================================
# Tạo thư mục
# =====================================================

os.makedirs("../results/tables", exist_ok=True)
os.makedirs("../results/models", exist_ok=True)

# =====================================================
# Đọc dữ liệu
# =====================================================

DATA_PATH = "../data/breast_cancer.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Đọc dữ liệu thành công")
print("=" * 60)

# =====================================================
# Kiểm tra dữ liệu thiếu
# =====================================================

missing = df.isnull().sum().sum()
print(f"Tổng số giá trị thiếu: {missing}")

if missing > 0:
    print("Có dữ liệu thiếu!")
    df = df.dropna()
    print("Đã loại bỏ các dòng chứa NA")
else:
    print("Không có dữ liệu thiếu")

# =====================================================
# Loại bỏ ID
# =====================================================

if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

# =====================================================
# Chuyển nhãn
# =====================================================

label_mapping = {"M":0, "B":1}
df["diagnosis"] = df["diagnosis"].map(label_mapping)
print(df["diagnosis"].value_counts())

# =====================================================
# Tách X và y
# =====================================================

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

print("Số đặc trưng:", X.shape[1])
print("Số mẫu:", X.shape[0])

# =====================================================
# Chuẩn hóa
# =====================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# Đưa về DataFrame
# =====================================================

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# =====================================================
# Ghép nhãn
# =====================================================

processed = X_scaled.copy()
processed["diagnosis"] = y.values

# =====================================================
# Lưu dữ liệu
# =====================================================

processed.to_csv("../results/tables/processed_data.csv", index=False)
print("Đã lưu processed_data.csv")

# =====================================================
# Lưu scaler
# =====================================================

joblib.dump(scaler, "../results/models/scaler.pkl")
print("Đã lưu scaler.pkl")

# =====================================================
# Thống kê sau chuẩn hóa
# =====================================================

statistics = processed.describe()
statistics.to_csv("../results/tables/processed_statistics.csv")
print(statistics)

# =====================================================
# Kiểm tra Mean
# =====================================================

print("Mean sau chuẩn hóa")
print(X_scaled.mean().head())

# =====================================================
# Kiểm tra Std
# =====================================================

print("Std sau chuẩn hóa")
print( X_scaled.std().head())

print("="*60)