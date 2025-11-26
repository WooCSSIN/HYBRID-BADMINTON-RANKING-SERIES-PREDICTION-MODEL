from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
DEFAULT_IN = BASE / 'bwf_cleaned_full_casted.csv'
DEFAULT_OUT = BASE / 'bwf_cleaned_full_ready.csv'

def prepare_ml_dataset(input_path: str | Path = DEFAULT_IN, output_path: str | Path = DEFAULT_OUT):
    #  Đọc dữ liệu
    df = pd.read_csv(input_path)

    #  Ép kiểu dữ liệu phù hợp
    # - Cột 'date' về dạng datetime để trích xuất feature thời gian
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # - Các cột phân loại nên để dạng 'category'
    categorical_cols = ["draw", "country_code", "gender", "draw_type", "event_name", "category", "draw_full_name"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # - Các cột định danh giữ dạng chuỗi
    id_cols = ["uid", "id", "name"]
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    #  Kiểm tra kết quả
    print("\n Kiểu dữ liệu sau khi chuẩn hóa:\n")
    print(df.dtypes)

    #  Lưu file mới
    df.to_csv(output_path, index=False)
    print(f"\n Đã lưu file chuẩn hóa cho ML: '{output_path}'")

    #  Thống kê nhanh
    print("\n Thống kê dữ liệu:")
    # include='all' có thể tạo nhiều output; giới hạn một chút cho readability
    try:
        print(df.describe(include='all'))
    except Exception:
        print(df.describe())


if __name__ == "__main__":
    prepare_ml_dataset()
