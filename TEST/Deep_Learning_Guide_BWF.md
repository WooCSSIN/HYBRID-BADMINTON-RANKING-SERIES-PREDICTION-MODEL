# Hướng Dẫn Deep Learning Cho Dự Báo Ranking Cầu Lông BWF

## 📚 Mục Lục
1. [Deep Learning là gì?](#1-deep-learning-là-gì)
2. [Tại sao sử dụng DL cho BWF Ranking?](#2-tại-sao-sử-dụng-dl-cho-bwf-ranking)
3. [Các kiến trúc DL phù hợp](#3-các-kiến-trúc-dl-phù-hợp)
4. [Dữ liệu BWF và Feature Engineering](#4-dữ-liệu-bwf-và-feature-engineering)
5. [Implementation với PyTorch/TensorFlow](#5-implementation-với-pytorchtensorflow)
6. [So sánh DL vs ML truyền thống](#6-so-sánh-dl-vs-ml-truyền-thống)
7. [Best Practices và Tips](#7-best-practices-và-tips)

---

## 1. Deep Learning là gì?

### 1.1 Định nghĩa
**Deep Learning (DL)** là một nhánh con của Machine Learning, sử dụng mạng neural nhân tạo (Artificial Neural Networks) với nhiều tầng ẩn (hidden layers) để học các đặc trưng phức tạp từ dữ liệu.

### 1.2 Cấu trúc cơ bản của Neural Network

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Output Layer
```

**Ví dụ với dữ liệu BWF:**
```
[Features: points, rank, win_rate, ...] 
    ↓
[Hidden Layer 1: 128 neurons] 
    ↓
[Hidden Layer 2: 64 neurons]
    ↓
[Hidden Layer 3: 32 neurons]
    ↓
[Output: Predicted ranking/points]
```

### 1.3 Các thành phần chính

#### a) Neurons (Nơ-ron)
- Đơn vị xử lý cơ bản
- Nhận input, áp dụng weights và bias
- Kết quả qua activation function

#### b) Weights và Biases
- **Weights (trọng số)**: Độ quan trọng của mỗi kết nối
- **Biases (độ lệch)**: Điều chỉnh ngưỡng kích hoạt
- Được học tự động qua quá trình training

#### c) Activation Functions
- **ReLU**: `f(x) = max(0, x)` - Phổ biến nhất
- **Sigmoid**: `f(x) = 1/(1+e^-x)` - Cho output 0-1
- **Tanh**: `f(x) = (e^x - e^-x)/(e^x + e^-x)` - Output -1 đến 1

---

## 2. Tại sao sử dụng DL cho BWF Ranking?

### 2.1 Ưu điểm của DL

✅ **Automatic Feature Learning**
- DL tự động học các pattern phức tạp
- Không cần thiết kế features thủ công nhiều
- Phát hiện được các mối quan hệ phi tuyến

✅ **Temporal Dependencies**
- Xử lý tốt chuỗi thời gian (time series)
- Học được xu hướng dài hạn và ngắn hạn
- Phù hợp với dữ liệu ranking theo thời gian

✅ **Multiple Draws Handling**
- Có thể học đặc trưng riêng cho từng draw (WS, MS, WD, MD, XD)
- Transfer learning giữa các draws

### 2.2 Khi nào NÊN dùng DL?

✔️ Có nhiều dữ liệu (>10,000 samples)
✔️ Pattern phức tạp, phi tuyến
✔️ Cần dự báo chuỗi thời gian
✔️ Muốn tự động feature engineering

### 2.3 Khi nào KHÔNG NÊN dùng DL?

❌ Dữ liệu ít (<1,000 samples)
❌ Cần interpretability cao (giải thích từng quyết định)
❌ Tài nguyên tính toán hạn chế
❌ ML truyền thống đã cho kết quả tốt

---

## 3. Các kiến trúc DL phù hợp

### 3.1 Feedforward Neural Network (FNN)

**Khi nào dùng:** Dự báo điểm/ranking tại thời điểm hiện tại

```python
class RankingFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32]):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.output = nn.Linear(hidden_sizes[2], 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return self.output(x)
```

**Ưu điểm:**
- Đơn giản, dễ implement
- Training nhanh
- Phù hợp với tabular data

**Nhược điểm:**
- Không xử lý tốt temporal dependencies
- Cần feature engineering kỹ

---

### 3.2 LSTM (Long Short-Term Memory)

**Khi nào dùng:** Dự báo xu hướng ranking theo thời gian

```python
class RankingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        lstm_out, _ = self.lstm(x)
        # Lấy output của timestep cuối cùng
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)
```

**Ưu điểm:**
- Xử lý tốt sequential data
- Nhớ được thông tin dài hạn
- Phù hợp với time series

**Nhược điểm:**
- Training chậm hơn FNN
- Cần nhiều dữ liệu hơn
- Có thể overfit

**Cách sử dụng với dữ liệu BWF:**
```python
# Tạo sequences từ dữ liệu
# VD: Dùng 12 tuần trước để dự báo tuần tiếp theo
sequence_length = 12
features = ['points', 'rank', 'win_rate_estimated', 'momentum_score', ...]

# Input shape: (batch_size, 12, num_features)
# Output: Predicted points/rank cho tuần tiếp theo
```

---

### 3.3 Transformer

**Khi nào dùng:** Bài toán phức tạp, cần attention mechanism

```python
class RankingTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=256
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Global average pooling
        return self.fc(x)
```

**Ưu điểm:**
- State-of-the-art performance
- Attention mechanism học được quan hệ quan trọng
- Parallel processing (nhanh hơn LSTM)

**Nhược điểm:**
- Phức tạp, khó tune
- Cần nhiều dữ liệu
- Tài nguyên tính toán cao

---

## 4. Dữ liệu BWF và Feature Engineering

### 4.1 Phân tích dữ liệu bwf_official_enhanced.csv

Từ dữ liệu của bạn, tôi thấy các features sau:

#### A. Temporal Features (Thời gian)
```python
temporal_features = [
    'career_years',      # Số năm trong sự nghiệp
    'career_months',     # Số tháng
    'in_peak_career',    # Có đang ở đỉnh cao không (0/1)
    'career_stage'       # Giai đoạn: rookie/rising/peak/declining
]
```

#### B. Performance Features (Thành tích)
```python
performance_features = [
    'points',                  # Điểm hiện tại
    'rank',                    # Hạng hiện tại
    'tournaments_played',      # Số giải đấu đã chơi
    'win_rate_estimated',      # Tỷ lệ thắng ước tính
    'loss_rate_estimated',     # Tỷ lệ thua
    'win_streak_estimated'     # Chuỗi thắng liên tiếp
]
```

#### C. Historical Features (Lịch sử)
```python
historical_features = [
    'points_lag_1',      # Điểm 1 tuần trước
    'points_lag_3',      # Điểm 3 tuần trước
    'points_lag_6',      # Điểm 6 tuần trước
    'points_lag_12',     # Điểm 12 tuần trước
    'rank_lag_1',        # Hạng 1 tuần trước
    'rank_lag_3'         # Hạng 3 tuần trước
]
```

#### D. Statistical Features (Thống kê)
```python
statistical_features = [
    'avg_points_3m',       # Điểm trung bình 3 tháng
    'avg_points_6m',       # Điểm trung bình 6 tháng
    'avg_points_12m',      # Điểm trung bình 12 tháng
    'std_points_6m',       # Độ lệch chuẩn 6 tháng
    'std_points_12m',      # Độ lệch chuẩn 12 tháng
    'points_change_1m',    # Thay đổi điểm 1 tháng
    'points_change_6m',    # Thay đổi điểm 6 tháng
    'rank_change_1m'       # Thay đổi hạng 1 tháng
]
```

#### E. Momentum Features (Xu hướng)
```python
momentum_features = [
    'momentum_score',      # Điểm momentum
    'trend_3m',            # Xu hướng 3 tháng
    'consistency_score'    # Điểm ổn định
]
```

### 4.2 Feature Engineering cho Deep Learning

```python
# 1. Normalization (Chuẩn hóa)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Chuẩn hóa các features về cùng scale
scaler = StandardScaler()
normalized_features = scaler.fit_transform(df[all_features])

# 2. Categorical Encoding
# Draw: WS, MS, WD, MD, XD → One-hot encoding
draw_encoded = pd.get_dummies(df['draw'], prefix='draw')

# Career stage → Label encoding hoặc one-hot
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['career_stage_encoded'] = le.fit_transform(df['career_stage'])

# 3. Temporal Sequences (Cho LSTM/Transformer)
def create_sequences(df, sequence_length=12):
    """
    Tạo sequences cho mỗi cầu thủ
    Input: DataFrame đã sort theo player_id và date
    Output: X shape (n_samples, sequence_length, n_features)
            y shape (n_samples, 1)
    """
    sequences = []
    targets = []
    
    for player_id in df['player_id'].unique():
        player_data = df[df['player_id'] == player_id].sort_values('date')
        
        for i in range(len(player_data) - sequence_length):
            # Lấy 12 tuần dữ liệu
            seq = player_data.iloc[i:i+sequence_length][features].values
            # Target: điểm của tuần thứ 13
            target = player_data.iloc[i+sequence_length]['points']
            
            sequences.append(seq)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)
```

---

## 5. Implementation với PyTorch/TensorFlow

### 5.1 Complete Training Pipeline (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# ==================== Data Preparation ====================
class BWFDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Load và prepare data
df = pd.read_csv('bwf_official_enhanced.csv')

# Select features
feature_cols = [
    'points', 'rank', 'tournaments_played', 'win_rate_estimated',
    'points_lag_1', 'points_lag_3', 'points_lag_6',
    'avg_points_3m', 'avg_points_6m', 'std_points_6m',
    'momentum_score', 'trend_3m', 'consistency_score',
    'career_years', 'career_months'
]

# Target: Dự báo điểm ở tuần tiếp theo
X = df[feature_cols].values
y = df['points'].shift(-1).dropna().values  # Shift để lấy điểm tuần sau
X = X[:-1]  # Bỏ dòng cuối vì không có label

# Train/Val/Test split
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Create DataLoaders
train_dataset = BWFDataset(X_train, y_train)
val_dataset = BWFDataset(X_val, y_val)
test_dataset = BWFDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# ==================== Model Definition ====================
class BWFRankingNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# ==================== Training ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BWFRankingNet(input_size=len(feature_cols)).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(loader)

# Training loop
num_epochs = 100
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

# ==================== Evaluation ====================
model.load_state_dict(torch.load('best_model.pth'))
test_loss = validate(model, test_loader, criterion, device)
print(f'\nTest Loss: {test_loss:.4f}')

# Get predictions
model.eval()
all_predictions = []
all_actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        predictions = model(X_batch.to(device))
        all_predictions.extend(predictions.cpu().numpy())
        all_actuals.extend(y_batch.numpy())

# Calculate metrics
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(all_actuals, all_predictions)
r2 = r2_score(all_actuals, all_predictions)

print(f'MAE: {mae:.2f} points')
print(f'R² Score: {r2:.4f}')
```

### 5.2 LSTM Implementation

```python
class BWFRankingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x: (batch, sequence_length, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Lấy hidden state cuối cùng
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.fc(last_hidden)

# Chuẩn bị data dạng sequences
X_seq, y_seq = create_sequences(df_sorted, sequence_length=12)
# X_seq shape: (n_samples, 12, n_features)
# y_seq shape: (n_samples,)

# Training tương tự như FNN
```

---

## 6. So sánh DL vs ML truyền thống

### 6.1 Bảng so sánh

| Tiêu chí | Deep Learning | ML Truyền thống (RF, XGBoost) |
|----------|---------------|-------------------------------|
| **Dữ liệu cần** | Nhiều (>10k samples) | Ít-Trung bình (>1k samples) |
| **Feature Engineering** | Tự động học features | Cần thiết kế thủ công |
| **Interpretability** | Thấp (black box) | Cao (feature importance) |
| **Training time** | Chậm (giờ-ngày) | Nhanh (phút-giờ) |
| **Inference time** | Nhanh | Rất nhanh |
| **Overfitting risk** | Cao | Trung bình |
| **Temporal patterns** | Tốt (LSTM, Transformer) | Cần feature engineering |
| **Performance** | Cao (nếu đủ data) | Tốt với tabular data |

### 6.2 Kết quả thực tế với dữ liệu BWF

Dựa trên kinh nghiệm và dữ liệu của bạn:

**XGBoost/Random Forest:**
- ✅ Cho kết quả tốt với ~29,000 samples
- ✅ Training nhanh (< 1 phút)
- ✅ Dễ interpret (feature importance)
- ✅ Ít prone to overfitting

**Deep Learning:**
- ✅ Có thể học temporal patterns tốt hơn
- ✅ Tự động feature interactions
- ⚠️ Cần tuning kỹ để tránh overfit
- ⚠️ Training lâu hơn

**Khuyến nghị:**
```
Sử dụng ENSEMBLE của cả 2:
- XGBoost cho baseline tốt
- LSTM cho temporal patterns
- Kết hợp predictions (weighted average hoặc stacking)
```

---

## 7. Best Practices và Tips

### 7.1 Tránh Overfitting

```python
# 1. Regularization
model = nn.Sequential(
    nn.Linear(input_size, 128),
    nn.Dropout(0.3),  # ← Dropout
    nn.Linear(128, 64),
    nn.Dropout(0.2)
)

# Weight decay trong optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 2. Early Stopping
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

# 3. Data Augmentation (cho time series)
def augment_sequence(seq, noise_level=0.01):
    """Thêm noise nhẹ vào sequence"""
    noise = np.random.normal(0, noise_level, seq.shape)
    return seq + noise

# 4. Cross-validation
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    # Train model for this fold
```

### 7.2 Hyperparameter Tuning

```python
# Sử dụng Optuna cho automatic tuning
import optuna

def objective(trial):
    # Suggest hyperparameters
    hidden_size_1 = trial.suggest_int('hidden_size_1', 64, 256)
    hidden_size_2 = trial.suggest_int('hidden_size_2', 32, 128)
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    
    # Build model
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size_1),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size_1, hidden_size_2),
        nn.ReLU(),
        nn.Linear(hidden_size_2, 1)
    ).to(device)
    
    # Train and return validation loss
    # ... (training code)
    
    return val_loss

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Best hyperparameters:', study.best_params)
```

### 7.3 Monitoring và Visualization

```python
import matplotlib.pyplot as plt

# Track losses
train_losses = []
val_losses = []

# During training
for epoch in range(num_epochs):
    train_loss = train_epoch(...)
    val_loss = validate(...)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Curves')
plt.savefig('learning_curves.png')

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(all_actuals, all_predictions, alpha=0.5)
plt.plot([min(all_actuals), max(all_actuals)], 
         [min(all_actuals), max(all_actuals)], 
         'r--', label='Perfect prediction')
plt.xlabel('Actual Points')
plt.ylabel('Predicted Points')
plt.legend()
plt.title('Predictions vs Actual')
plt.savefig('predictions_vs_actual.png')
```

### 7.4 Tips cho BWF Ranking específicamente

1. **Multi-task Learning**
```python
class MultiTaskModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Task-specific heads
        self.points_head = nn.Linear(128, 1)  # Dự báo điểm
        self.rank_head = nn.Linear(128, 1)    # Dự báo hạng
    
    def forward(self, x):
        shared_features = self.shared(x)
        points_pred = self.points_head(shared_features)
        rank_pred = self.rank_head(shared_features)
        return points_pred, rank_pred
```

2. **Draw-specific Models**
```python
# Train riêng model cho mỗi draw
models = {}
for draw in ['WS', 'MS', 'WD', 'MD', 'XD']:
    df_draw = df[df['draw'] == draw]
    models[draw] = train_model(df_draw)
    
# hoặc dùng shared model với draw embedding
```

3. **Ensemble Predictions**
```python
# Kết hợp DL với XGBoost
dl_pred = model(X_test)
xgb_pred = xgb_model.predict(X_test)

# Weighted average
final_pred = 0.6 * dl_pred + 0.4 * xgb_pred

# Hoặc train meta-model (stacking)
```

---

## 📖 Tài liệu tham khảo

1. **PyTorch Official Tutorial**: https://pytorch.org/tutorials/
2. **Deep Learning Book** (Goodfellow et al.): https://www.deeplearningbook.org/
3. **Time Series Forecasting with Deep Learning**: 
   - https://arxiv.org/abs/1704.04110
4. **Practical Deep Learning for Coders** (fast.ai): https://course.fast.ai/

---

## 🎯 Kết luận

Deep Learning là công cụ mạnh mẽ cho bài toán dự báo BWF ranking, đặc biệt khi:
- Bạn có đủ dữ liệu (✅ ~29k samples)
- Muốn tự động học temporal patterns
- Cần dự báo nhiều draws khác nhau

**Lộ trình đề xuất:**
1. ✅ Bắt đầu với FNN đơn giản
2. ✅ Thử LSTM cho temporal modeling
3. ✅ So sánh với XGBoost baseline
4. ✅ Ensemble các models lại
5. ✅ Fine-tune hyperparameters

Chúc bạn thành công! 🏸
