# Phân loại không giám sát với Gaussian Mixture Models (GMM)

Dự án này áp dụng mô hình **Gaussian Mixture Models (GMM)** để thực hiện phân loại không giám sát trên tập dữ liệu **Breast Cancer Wisconsin**. Mục tiêu của dự án là phân nhóm các khối u thành hai loại: "Lành tính" (Benign) và "Ác tính" (Malignant) dựa trên các đặc trưng sinh học mà không sử dụng nhãn phân lớp trong quá trình huấn luyện mô hình.

## Cấu trúc thư mục

<!--Dự án được tổ chức thành các thư mục sau: -->

- `data/`: Chứa dữ liệu gốc (được tải về tự động qua script).
- `results/`: Nơi lưu trữ các kết quả đầu ra của quá trình chạy.
  - `tables/`: Các bảng thống kê (giá trị thiếu, thống kê mô tả, phân bố lớp) và dữ liệu đã qua tiền xử lý.
  - `figures/`: Các biểu đồ trực quan hóa (phân bố dữ liệu, kết quả cụm).
  - `models/`: Lưu các mô hình đã huấn luyện (ví dụ: mô hình GMM, StandardScaler).
- `src/`: Thư mục mã nguồn chính, chứa các script Python được đánh số thứ tự thực hiện:
  - `01_load_data.py`: Tải dữ liệu Breast Cancer từ thư viện `sklearn`, phân tích khám phá cơ bản (EDA) và lưu thành file CSV.
  - `02_preprocessing.py`: Tiền xử lý dữ liệu bao gồm kiểm tra giá trị thiếu và chuẩn hóa dữ liệu bằng `StandardScaler`.
  - `03_train_gmm.py`: Huấn luyện mô hình Gaussian Mixture Model cơ bản.
  - `04_random_initialization.py`: Thử nghiệm mô hình GMM với các điểm khởi tạo ngẫu nhiên khác nhau.
  - `05_model_selection.py`: Đánh giá và lựa chọn số lượng cụm / cấu hình mô hình tối ưu sử dụng các tiêu chí thông tin như BIC/AIC.
  - `06_cluster_visualization.py`: Trực quan hóa các cụm (clusters) được phân loại bởi GMM trong không gian đặc trưng.
  - `07_posterior_analysis.py`: Phân tích xác suất hậu nghiệm (posterior probabilities) của các mẫu thuộc về từng cụm.
  - `08_evaluation.py`: Đánh giá hiệu suất của mô hình GMM bằng cách đối chiếu kết quả cụm (clusters) với nhãn thực tế.
- `breast-cancer-wisconsin.ipynb`: Notebook Jupyter dùng để chạy thử nghiệm và phân tích tương tác.

## Yêu cầu môi trường

Để chạy dự án, bạn cần cài đặt Python (phiên bản 3.7+ được khuyến nghị) cùng với các thư viện sau:

```bash
pip install pandas scikit-learn matplotlib joblib
```

## Hướng dẫn sử dụng

Chạy tuần tự các script trong thư mục `src/`. Ví dụ (nếu bạn đang ở thư mục `src/`):

```bash
python 01_load_data.py
python 02_preprocessing.py
python 03_train_gmm.py
...
```

Sau khi chạy xong, toàn bộ kết quả phân tích, mô hình và hình ảnh trực quan sẽ được lưu tại thư mục `results/`.
