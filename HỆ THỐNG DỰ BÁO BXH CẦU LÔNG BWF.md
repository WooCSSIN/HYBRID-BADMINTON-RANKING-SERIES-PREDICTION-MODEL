# 📊 HỆ THỐNG DỰ BÁO BẢNG XẾP HẠNG CẦU LÔNG BWF

> **Dự án Machine Learning dự báo bảng xếp hạng cầu lông BWF đến năm 2035**
>
> Phân tích dữ liệu lịch sử, huấn luyện mô hình ML (LightGBM) và tạo dự báo cho toàn bộ 5 nội dung thi đấu trên 3 khu vực châu lục.

---

## 📑 Mục Lục

1. [Tổng Quan Dự Án](#-tổng-quan-dự-án)
2. [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
3. [Luồng Hoạt Động](#-luồng-hoạt-động)
4. [Chi Tiết Các Module](#-chi-tiết-các-module)
5. [Dữ Liệu](#-dữ-liệu)
6. [Mô Hình Machine Learning](#-mô-hình-machine-learning)
7. [Kết Quả Đầu Ra](#-kết-quả-đầu-ra)
8. [Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
9. [Visualization với Power BI](#-visualization-với-power-bi)

---

## 🎯 Tổng Quan Dự Án

### Mục Tiêu
Xây dựng hệ thống dự báo **bảng xếp hạng cầu lông thế giới (BWF)** đến năm **2035** cho:

- **5 Nội Dung Thi Đấu**: 
  - `MS` - Men's Singles (Đơn nam)
  - `WS` - Women's Singles (Đơn nữ)
  - `MD` - Men's Doubles (Đôi nam)
  - `WD` - Women's Doubles (Đôi nữ)
  - `XD` - Mixed Doubles (Đôi nam nữ)

- **3 Khu Vực Châu Lục**:
  - `Asia` - Châu Á
  - `Europe` - Châu Âu
  - `Global` - Toàn cầu

### Công Nghệ Sử Dụng
- **Ngôn ngữ**: Python 3.x
- **Thư viện ML chính**: 
  - `LightGBM` - Gradient Boosting cho dự báo chính
  - `scikit-learn` (GradientBoostingRegressor) - Dự báo thử nghiệm
  - `PyTorch` - Deep Learning LSTM (thử nghiệm)
- **Xử lý dữ liệu**: Pandas, NumPy
- **Visualization**: Power BI Desktop
- **Lưu trữ**: CSV files

---

## 📁 Cấu Trúc Thư Mục

```
data kaggle badmintont - Copy/
│
├── 📂 file py/                          # Mã nguồn Python
│   ├── bwf_official.py                  # [Module 1] Chuẩn hóa dữ liệu cho SQL
│   ├── prepare_ml_dataset.py            # [Module 2] Chuẩn bị dữ liệu cho ML
│   └── forecast_to_2035.py              # [Module 3] Dự báo đến 2035
│
├── 📂 data/                             # Thư mục dữ liệu (có thể có subfolder)
│   └── dl_test/                         # Dữ liệu test cho Deep Learning
│
├── 📂 MACHINE LEARNING/                 # Dữ liệu đầu vào & đầu ra
│   ├── bwf_official.csv                 # Dữ liệu gốc từ BWF Kaggle
│   ├── bwf_cleaned_full.csv             # Dữ liệu đã làm sạch
│   ├── bwf_for_sql_simple.csv           # Dữ liệu đơn giản hóa cho SQL
│   ├── bwf_rank_history.csv             # Lịch sử xếp hạng
│   ├── bwf_players.csv                  # Danh sách cầu thủ
│   ├── dim_player_clean.csv             # Dimension table cầu thủ
│   ├── bwf_countries.csv / dim_country.csv  # Thông tin quốc gia
│   │
│   ├── Top10_Global_MS_2035.csv         # 🎯 Kết quả: Top 10 Đơn nam Toàn cầu 2035
│   ├── Top10_Global_WS_2035.csv         # 🎯 Kết quả: Top 10 Đơn nữ Toàn cầu 2035
│   ├── Top10_Global_MD_2035.csv         # 🎯 Kết quả: Top 10 Đôi nam Toàn cầu 2035
│   ├── Top10_Global_WD_2035.csv         # 🎯 Kết quả: Top 10 Đôi nữ Toàn cầu 2035
│   ├── Top10_Global_XD_2035.csv         # 🎯 Kết quả: Top 10 Đôi nam nữ Toàn cầu 2035
│   ├── Top10_Asia_{MS|WS|MD|WD|XD}_2035.csv    # 🎯 Kết quả: Top 10 Châu Á
│   ├── Top10_Europe_{MS|WS|MD|WD|XD}_2035.csv  # 🎯 Kết quả: Top 10 Châu Âu
│   │
│   ├── PWBI.pbix                        # File Power BI cho visualization
│   └── POWER BI - DỰ ĐOÁN BXH CẦU LÔNG.pbix  # Dashboard chính
│
└── 📂 models/                           # Mô hình ML đã train
    ├── lightgbm_MS_Global.pkl           # Model Đơn nam Toàn cầu
    ├── lightgbm_WS_Global.pkl           # Model Đơn nữ Toàn cầu
    ├── lightgbm_MD_Global.pkl           # Model Đôi nam Toàn cầu
    ├── lightgbm_WD_Global.pkl           # Model Đôi nữ Toàn cầu
    ├── lightgbm_XD_Global.pkl           # Model Đôi nam nữ Toàn cầu
    ├── lightgbm_{MS|WS|MD|WD|XD}_Asia.pkl      # Models Châu Á
    ├── lightgbm_{MS|WS|MD|WD|XD}_Europe.pkl    # Models Châu Âu
    └── dl_lstm_cpu.pt                   # Model LSTM (thử nghiệm)
```

---

## 🔄 Luồng Hoạt Động

```mermaid
graph TD
    A[Dữ liệu BWF Kaggle CSV] --> B[Module 1: bwf_official.py]
    B --> C[bwf_for_sql_pairs.csv]
    
    A --> D[Module 2: prepare_ml_dataset.py]
    D --> E[bwf_cleaned_full_ready.csv]
    
    C --> F[Module 3: forecast_to_2035.py]
    F --> G{Train Models LightGBM}
    
    G --> H[15 Models theo Draw x Continent]
    H --> I[Dự báo 120 tháng tương lai]
    I --> J[15 files Top10_{Continent}_{Draw}_2035.csv]
    
    J --> K[Power BI Dashboard]
    K --> L[Visualization & Analysis]
    
    style A fill:#e1f5ff
    style G fill:#fff4e1
    style J fill:#e1ffe1
    style K fill:#ffe1f5
```

### Các Bước Chính

1. **Thu thập dữ liệu**: Dữ liệu lịch sử BWF từ Kaggle
2. **Tiền xử lý**: Làm sạch, chuẩn hóa, tạo features
3. **Training**: Huấn luyện 15+ mô hình LightGBM riêng biệt
4. **Dự báo**: Recursive forecasting 120 tháng (đến 2035)
5. **Xuất kết quả**: 15 file CSV Top 10 theo từng phân khúc
6. **Visualization**: Dashboard Power BI

---

## 🔧 Chi Tiết Các Module

### Module 1: `bwf_official.py` - Chuẩn Hóa Dữ Liệu

**Mục đích**: Chuyển đổi dữ liệu thô BWF thành định dạng chuẩn cho SQL/Database

#### Input
- `bwf_cleaned_full.csv` - Dữ liệu đã làm sạch sơ bộ

#### Output
- `bwf_for_sql_pairs.csv` - Dữ liệu chuẩn hóa cho SQL

#### Xử lý chính

```python
Bước 1: Xác định player_id chính
  - Ưu tiên: id_main → id_main_int → parse từ uid
  - Chuyển sang kiểu Int64

Bước 2: Xác định player2_id (cho đôi)
  - Lấy từ: id_aux hoặc parse từ uid
  - Cho phép giá trị NA với đơn

Bước 3: Xử lý các trường
  - date: Convert về datetime → format YYYY-MM-DD
  - draw: Nội dung thi đấu (MS/WS/MD/WD/XD)
  - country_code: Mã quốc gia
  - rank, points, tournaments_played: Numeric Int64

Bước 4: Lọc và xuất
  - Chỉ giữ các cột cần thiết
  - Xuất với encoding UTF-8-sig
```

#### Các cột đầu ra
- `uid` - Unique ID
- `date` - Ngày ghi nhận
- `draw` - Nội dung thi đấu
- `country_code` - Quốc gia
- `player_id`, `player_name` - Cầu thủ chính
- `player2_id`, `player2_name` - Cầu thủ đôi (nếu có)
- `rank` - Xếp hạng
- `points` - Điểm BWF
- `tournaments_played` - Số giải đã thi đấu

---

### Module 2: `prepare_ml_dataset.py` - Chuẩn Bị Dataset ML

**Mục đích**: Ép kiểu dữ liệu và tạo features phù hợp cho Machine Learning

#### Input
- `bwf_cleaned_full_casted.csv`

#### Output
- `bwf_cleaned_full_ready.csv`

#### Xử lý chính

```python
Bước 1: Ép kiểu datetime
  - date → pd.datetime (để extract year, month, day)

Bước 2: Ép kiểu category (tiết kiệm memory)
  - draw, country_code, gender
  - draw_type, event_name, category
  - draw_full_name

Bước 3: Giữ kiểu string cho ID
  - uid, id, name

Bước 4: Thống kê và kiểm tra
  - df.describe(include='all')
  - Kiểm tra dtypes
```

#### Lợi ích
- Giảm memory usage với categorical
- Sẵn sàng cho feature engineering
- Đảm bảo tính nhất quán kiểu dữ liệu

---

### Module 3: `forecast_to_2035.py` - Dự Báo ML Đến 2035

**Mục đích**: Huấn luyện model và dự báo Top 10 cho từng nội dung và châu lục

#### Input
- `bwf_official.csv`

#### Output
- 15 files: `Top10_{Global|Asia|Europe}_{MS|WS|MD|WD|XD}_2035.csv`

#### Pipeline chi tiết

##### 1️⃣ Load & Prepare Data
```python
- Load bwf_official.csv
- Convert date sang datetime
- Filter NaN trong date, draw
- Chuẩn hóa draw thành uppercase
- Map country_code → continent (Asia/Europe/Global)
- Convert rank, points, tournaments_played sang numeric
- Sort theo: player_id, draw, date
```

##### 2️⃣ Feature Engineering
Tạo **Lag Features** và **Rolling Statistics** cho mỗi cầu thủ theo từng draw:

```python
Lag Features (xem dữ liệu quá khứ):
  - points_lag_1: Điểm của tháng trước
  - points_lag_3: Điểm của 3 tháng trước
  - points_lag_6: Điểm của 6 tháng trước
  - rank_lag_1: Xếp hạng tháng trước

Rolling Features (xu hướng):
  - avg_points_3m: Điểm trung bình 3 tháng
  - avg_points_6m: Điểm trung bình 6 tháng
  - std_points_6m: Độ lệch chuẩn 6 tháng (đo biến động)

Time Features:
  - month: Tháng trong năm (1-12)
```

**Xử lý missing values**:
- Fill với median của từng nhóm (player_id, draw)
- Nếu vẫn NA → fill = 0

##### 3️⃣ Train Models
```python
Algorithm: GradientBoostingRegressor (sklearn)
Hyperparameters:
  - n_estimators = 200
  - learning_rate = 0.05
  - max_depth = 5
  - random_state = 42

Split Strategy:
  - Train: 70% dữ liệu sớm nhất (theo date)
  - Test: 30% còn lại (không dùng trong code này, chỉ train)

Tạo mô hình riêng:
  - 1 model cho mỗi draw (MS, WS, MD, WD, XD)
  - Tổng: 5 models
```

##### 4️⃣ Recursive Forecasting
```python
Forecast Horizon: 120 tháng (10 năm từ dữ liệu mới nhất → 2035)

Quy trình cho mỗi cầu thủ:
  1. Lấy trạng thái mới nhất (latest record)
  2. For month_step in 1..120:
       a. Chuẩn bị features từ trạng thái hiện tại
       b. Dự báo điểm: pred_points = model.predict(features)
       c. Lưu kết quả dự báo
       d. Cập nhật trạng thái (state update):
          - points_lag_1 = pred_points
          - points_lag_3 = weighted average
          - points_lag_6 = weighted average
          - avg_points_3m, avg_points_6m update tương ứng
          - month = (month % 12) + 1

Output cho mỗi forecast:
  - player_id, player_name, player2_id, player2_name
  - country_code, continent, draw
  - date (forecast date)
  - predicted_points (điểm dự báo)
  - horizon (month_step)
```

##### 5️⃣ Generate Top 10 Files
```python
Cho mỗi (draw, continent):
  1. Lọc forecasts theo draw và continent
  2. Lấy record mới nhất của mỗi player (tháng 120)
  3. Sort theo predicted_points giảm dần
  4. Top 10 players
  5. Assign predicted_rank từ 1-10
  6. Format columns:
     - Singles (MS/WS): player_id, player_name, country_code, ...
     - Doubles (MD/WD/XD): + player2_id, player2_name
  7. Save to CSV: Top10_{continent}_{draw}_2035.csv
```

#### Ví dụ Output (Top10_Asia_MS_2035.csv)
| predicted_rank | player_id | player_name | country_code | continent | draw | predicted_points | date |
|----------------|-----------|-------------|--------------|-----------|------|------------------|-----------|
| 1 | 12345 | Player A | CHN | Asia | MS | 95432.5 | 2035-01-15 |
| 2 | 23456 | Player B | JPN | Asia | MS | 92100.3 | 2035-01-15 |
| ... | ... | ... | ... | ... | ... | ... | ... |

---

## 📊 Dữ Liệu

### Nguồn Dữ Liệu
- **Kaggle BWF Dataset**: Dữ liệu lịch sử bảng xếp hạng BWF
- **Thời gian**: Từ quá khứ đến hiện tại (cần check actual date range)
- **Độ phủ**: Toàn bộ 5 nội dung, nhiều quốc gia

### Các File Dữ Liệu Chính

#### 1. `bwf_official.csv`
Dữ liệu gốc từ BWF với đầy đủ thông tin cầu thủ, xếp hạng, điểm số

#### 2. `bwf_cleaned_full.csv`
Đã làm sạch: loại bỏ duplicates, xử lý missing values, chuẩn hóa format

#### 3. `bwf_rank_history.csv`
Lịch sử xếp hạng theo thời gian - dùng cho time series analysis

#### 4. `bwf_players.csv` & `dim_player_clean.csv`
Thông tin cầu thủ (dimension table cho data warehouse)

#### 5. `bwf_countries.csv` & `dim_country.csv`
Mapping country_code → continent, region

### Đặc Điểm Dữ Liệu
- **Time Series**: Dữ liệu theo tháng
- **Multi-variate**: rank, points, tournaments_played
- **Hierarchical**: Global → Continent → Country → Player
- **Mixed Types**: Singles vs Doubles structures

---

## 🤖 Mô Hình Machine Learning

### Thuật Toán Chính: LightGBM

**16 mô hình được lưu trong thư mục `models/`:**

#### Phân loại Models
```
Global Models (5):
├── lightgbm_MS_Global.pkl  - Đơn nam toàn cầu
├── lightgbm_WS_Global.pkl  - Đơn nữ toàn cầu
├── lightgbm_MD_Global.pkl  - Đôi nam toàn cầu
├── lightgbm_WD_Global.pkl  - Đôi nữ toàn cầu
└── lightgbm_XD_Global.pkl  - Đôi nam nữ toàn cầu

Asia Models (5):
├── lightgbm_MS_Asia.pkl
├── lightgbm_WS_Asia.pkl
├── lightgbm_MD_Asia.pkl
├── lightgbm_WD_Asia.pkl
└── lightgbm_XD_Asia.pkl

Europe Models (5):
├── lightgbm_MS_Europe.pkl
├── lightgbm_WS_Europe.pkl
├── lightgbm_MD_Europe.pkl
├── lightgbm_WD_Europe.pkl
└── lightgbm_XD_Europe.pkl

Experimental:
└── dl_lstm_cpu.pt  - LSTM Deep Learning (PyTorch)
```

### Tại Sao Tách Riêng 15 Models?

1. **Đặc thù nội dung khác nhau**:
   - Singles vs Doubles có dynamics khác nhau
   - Điểm số và ranking có phân phối khác

2. **Đặc điểm châu lục**:
   - Châu Á: Competitive cao, nhiều tournaments
   - Châu Âu: Pattern khác, ít player hơn
   - Global: Tổng hợp mọi khu vực

3. **Tăng độ chính xác**:
   - Specialized model > General model
   - Học được patterns riêng của từng segment

### LightGBM vs GradientBoostingRegressor

| Aspect | LightGBM (saved models) | GradientBoostingRegressor (code) |
|--------|------------------------|-----------------------------------|
| Speed | Rất nhanh | Chậm hơn |
| Memory | Hiệu quả | Tốn nhiều memory hơn |
| Accuracy | Cao hơn | Tốt |
| Status | Models đã train sẵn | Dùng trong forecast_to_2035.py |

> **Lưu ý**: Code hiện tại (`forecast_to_2035.py`) dùng GradientBoostingRegressor để train mới mỗi lần chạy. Các file `.pkl` LightGBM trong `models/` là các model đã được train trước đó và có thể được load để dùng lại thay vì train mới.

### Deep Learning: LSTM

File `dl_lstm_cpu.pt` - Model thử nghiệm với:
- Architecture: LSTM (Long Short-Term Memory)
- Framework: PyTorch
- Device: CPU (không dùng GPU)
- Status: Experimental, không dùng trong production pipeline

---

## 📤 Kết Quả Đầu Ra

### 15 Files Top 10

Mỗi file chứa **Top 10 cầu thủ/cặp đôi** được dự báo xếp hạng cao nhất vào năm **2035**:

#### Format: `Top10_{Continent}_{Draw}_2035.csv`

**Ví dụ cụ thể**:
- `Top10_Global_MS_2035.csv` - Top 10 Đơn nam Thế giới
- `Top10_Asia_WD_2035.csv` - Top 10 Đôi nữ Châu Á
- `Top10_Europe_XD_2035.csv` - Top 10 Đôi nam nữ Châu Âu

### Cấu Trúc File Output

#### Singles (MS, WS)
```csv
predicted_rank,player_id,player_name,country_code,continent,draw,predicted_points,date
1,12345,Nguyễn Văn A,VIE,Asia,MS,95432.5,2035-01-15
2,23456,Trần Thị B,VIE,Asia,MS,92100.3,2035-01-15
...
```

#### Doubles (MD, WD, XD)
```csv
predicted_rank,player_id,player_name,player2_id,player2_name,country_code,continent,draw,predicted_points,date
1,12345,Nguyễn Văn A,12346,Trần Văn B,VIE,Asia,MD,95432.5,2035-01-15
2,23456,Lê Văn C,23457,Phạm Văn D,VIE,Asia,MD,92100.3,2035-01-15
...
```

### Tổng Cộng Outputs
- **15 CSV files** (5 draws × 3 continents)
- Mỗi file: **10 rows** (Top 10)
- **Power BI Dashboard**: Tích hợp tất cả 15 files

---

## 🚀 Hướng Dẫn Sử Dụng

### Yêu Cầu Hệ Thống

```bash
Python: 3.8+
Thư viện:
  - pandas
  - numpy
  - scikit-learn
  - lightgbm (nếu dùng saved models)
  - torch (nếu dùng LSTM)
  - warnings

Power BI Desktop (cho visualization)
```

### Cài Đặt Dependencies

```bash
pip install pandas numpy scikit-learn lightgbm torch
```

### Chạy Pipeline

#### Bước 1: Chuẩn hóa dữ liệu cho SQL
```bash
cd "file py"
python bwf_official.py
```
**Output**: `bwf_for_sql_pairs.csv`

#### Bước 2: Chuẩn bị dataset cho ML (Tùy chọn)
```bash
python prepare_ml_dataset.py
```
**Output**: `bwf_cleaned_full_ready.csv`

#### Bước 3: Chạy dự báo đến 2035
```bash
python forecast_to_2035.py
```
**Output**: 15 files `Top10_*.csv` trong thư mục `MACHINE LEARNING/`

**Thời gian chạy**: 
- Tùy thuộc vào kích thước dataset
- Có thể mất 10-30 phút cho toàn bộ pipeline

#### Bước 4: Mở Power BI Dashboard
```bash
Mở file: POWER BI - DỰ ĐOÁN BXH CẦU LÔNG.pbix
Hoặc: MACHINE LEARNING/PWBI.pbix
```

### Tùy Chỉnh

#### Thay đổi forecast horizon
Trong `forecast_to_2035.py`:
```python
FORECAST_MONTHS = 120  # Đổi thành số tháng mong muốn
TARGET_YEAR = 2035     # Đổi năm target
```

#### Thay đổi hyperparameters
```python
model = GradientBoostingRegressor(
    n_estimators=200,      # Tăng để model phức tạp hơn
    learning_rate=0.05,    # Giảm để train chậm hơn nhưng stable
    max_depth=5,           # Tăng để capture patterns phức tạp hơn
    random_state=42
)
```

#### Thêm features mới
Trong `forecast_to_2035.py`, function `add_features()`:
```python
def add_features(group_df):
    group_df = group_df.sort_values('date')
    # ... existing features ...
    
    # Thêm features mới
    group_df['win_rate'] = ...
    group_df['points_change'] = group_df['points'].diff()
    
    return group_df
```

---

## 📈 Visualization với Power BI

### Dashboard Chính

File: `POWER BI - DỰ ĐOÁN BXH CẦU LÔNG.pbix`

#### Thành Phần Dashboard (Dự kiến)

1. **Overview Tab**
   - Top 10 Global cho từng draw
   - Biểu đồ so sánh điểm số
   - Phân bố theo quốc gia

2. **Continental Analysis**
   - Filter: Asia / Europe / Global
   - Drill-down theo country_code
   - Time series forecast visualization

3. **Draw Comparison**
   - So sánh MS vs WS vs MD vs WD vs XD
   - Heatmap theo continent

4. **Player Details**
   - Search và filter theo tên cầu thủ
   - Lịch sử và dự báo điểm
   - Trajectory visualization

### Data Connections

Power BI kết nối với:
1. Các file CSV trong `MACHINE LEARNING/`
2. Relationships giữa:
   - Players ← → Countries
   - Rankings ← → Forecasts
   - Draws ← → Continents

---

## 🔍 Phân Tích Kỹ Thuật

### Ưu Điểm Của Hệ Thống

✅ **Modular Design**: Tách biệt data prep, training, forecasting  
✅ **Scalable**: Dễ thêm draws, continents mới  
✅ **Recursive Forecasting**: Tự động cập nhật state cho long-term prediction  
✅ **Feature Engineering**: Sử dụng lags, rolling stats hiệu quả  
✅ **Specialized Models**: Tối ưu cho từng segment  

### Hạn Chế và Cải Tiến

⚠️ **Drift Risk**: Model có thể drift sau nhiều năm do recursive forecasting  
⚠️ **External Factors**: Không tính COVID, thay đổi luật, chấn thương  
⚠️ **Cold Start**: Cầu thủ mới không có đủ lags  

#### Cải Tiến Đề Xuất

1. **Thêm Features**:
   - Win/Loss ratio
   - Head-to-head records
   - Tournament categories (Super Series, Grand Prix)
   - Age, career length

2. **Advanced Models**:
   - Ensemble của LightGBM + LSTM
   - Prophet cho time series
   - XGBoost với custom objectives

3. **Validation**:
   - Walk-forward validation
   - Backtesting với historical data
   - Confidence intervals cho predictions

4. **Real-time Updates**:
   - API tích hợp với BWF live data
   - Incremental learning
   - Automated retraining pipeline

---

## 📞 Thông Tin Thêm

### Tài Liệu Tham Khảo

- **BWF Official**: [https://bwfbadminton.com/rankings/](https://bwfbadminton.com/rankings/)
- **Kaggle Dataset**: [Search "BWF Badminton Rankings"]
- **LightGBM Docs**: [https://lightgbm.readthedocs.io/](https://lightgbm.readthedocs.io/)
- **Gradient Boosting**: scikit-learn documentation

### Troubleshooting

**Lỗi: "File not found"**
- Kiểm tra đường dẫn tương đối `BASE = Path(__file__).parent`
- Đảm bảo chạy từ đúng thư mục

**Lỗi: "Insufficient data for draw X"**
- Draw đó có ít hơn 100 records
- Cần bổ sung data hoặc skip draw đó

**Power BI không load được data**
- Refresh data source
- Kiểm tra encoding của CSV (UTF-8-sig)
- Verify file paths trong Power Query

---

## 📝 Ghi Chú

> **Dự án này là proof-of-concept cho dự báo thể thao bằng ML.**  
> Kết quả dự báo chỉ mang tính chất tham khảo, phụ thuộc vào chất lượng dữ liệu lịch sử và giả định rằng xu hướng quá khứ sẽ tiếp tục.

**Version**: 1.0  
**Last Updated**: 2025-01-21  
**Created by**: [Your Name]

---

## 🎓 Kiến Thức Học Được

Dự án này minh họa:

- ✨ **Time Series Forecasting** với Gradient Boosting
- ✨ **Feature Engineering** cho sequential data
- ✨ **Recursive Multi-step Prediction**
- ✨ **Model Specialization** cho segments khác nhau
- ✨ **End-to-end ML Pipeline** từ data đến visualization
- ✨ **Business Intelligence** với Power BI

---

**Happy Forecasting! 🏸🚀**
