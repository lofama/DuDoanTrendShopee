import re
import pandas as pd
import pymongo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Kết nối tới MongoDB để lấy dữ liệu
def connect_to_mongodb():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["ShopeeDB"]
    return db


# Hàm để trích xuất tất cả các khóa duy nhất từ dữ liệu JSON
def extract_keys(data):
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    return all_keys


def convert_to_numeric(value):
    try:
        return float(re.sub(r"[^\d.]+", "", value))
    except (ValueError, TypeError):
        return 0


# Hàm ánh xạ location sang một ID từ Locationtable
def get_location_id(location):
    # Kết nối đến MongoDB
    client = connect_to_mongodb()
    db = client["ShopeeDB"]
    location_doc = db["LocationTable"].find_one({"location": location})
    if location_doc:
        return float(location_doc.get("id"))
    return 0


# Hàm làm sạch giá tiền
def clean_price(price_str):
    if isinstance(price_str, str):
        cleaned_str = re.sub(r"\D", "", price_str)
        if cleaned_str == "":
            return 0  # Trả về giá trị mặc định nếu chuỗi trống
        return int(cleaned_str)
    else:
        return price_str  # Trả về giá trị nếu không phải chuỗi


# Hàm làm sạch số lượng bán
def clean_sold(sold_str):
    if isinstance(sold_str, str):
        # Xử lý trường hợp có 'k'
        match_k = re.search(r"(\d+[,.\d+]*)k", sold_str)
        if match_k:
            return int(float(match_k.group(1).replace(",", ".")) * 1000)

        # Xử lý trường hợp có 'tr'
        match_tr = re.search(r"(\d+[,.\d+]*)tr", sold_str)
        if match_tr:
            return int(float(match_tr.group(1).replace(",", ".")) * 1000000)

        # Xử lý các trường hợp khác (số nguyên không có đơn vị)
        return int(re.sub(r"\D", "", sold_str))
    else:
        return sold_str  # Trả về giá trị nếu không phải chuỗi


def calculate_days(join_time_str):
    if isinstance(join_time_str, int):
        return join_time_str  # Nếu đã là số nguyên, trả về giá trị này

    if not isinstance(join_time_str, str):
        return 0  # Nếu không phải chuỗi hoặc số nguyên, trả về 0

    # Tìm số năm, tháng và ngày từ chuỗi
    match_years = re.search(r"(\d+)\s*năm", join_time_str)
    match_months = re.search(r"(\d+)\s*tháng", join_time_str)
    match_days = re.search(r"(\d+)\s*ngày", join_time_str)

    # Lấy số năm, tháng và ngày, nếu không có thì đặt giá trị mặc định là 0
    years = int(match_years.group(1)) if match_years else 0
    months = int(match_months.group(1)) if match_months else 0
    days = int(match_days.group(1)) if match_days else 0

    # Tính tổng số ngày (giả sử 1 năm = 365 ngày và 1 tháng = 30 ngày)
    total_days = years * 365 + months * 30 + days
    return total_days


# Lấy dữ liệu từ MongoDB
def get_data():
    db = connect_to_mongodb()
    data = list(db["dataSetShoppee"].find())
    return pd.DataFrame(data)


# Làm sạch và chuẩn bị dữ liệu
def clean_and_prepare_data(df):
    df["price"] = df["price"].apply(clean_price)
    df["sold"] = df["sold"].apply(clean_sold)
    df["ratings"] = df["ratings"].apply(clean_sold)
    df["followers"] = df["followers"].apply(clean_sold)
    df["productsCount"] = df["productsCount"].apply(clean_sold)
    df["joinTime"] = df["joinTime"].apply(calculate_days)
    df["discount"] = df["discount"].apply(clean_sold)
    df["location_id"] = df["location"].apply(clean_sold)
    return df


# Tính toán các thuộc tính dẫn xuất
def calculate_derived_features(df):
    df["rating_per_review"] = df["ratings"] / (df["productsCount"] + 1)
    df["followers_to_product_ratio"] = df["followers"] / (df["productsCount"] + 1)
    df["time_since_joining"] = df["joinTime"]
    df["sales_volume"] = df["sold"]
    # Giả sử bạn có danh sách các từ khóa xu hướng
    trending_keywords = ["từ khóa 1", "từ khóa 2", "từ khóa 3"]
    df["trending_words_count"] = df["name"].apply(
        lambda x: sum(word in x for word in trending_keywords)
    )
    return df


# Phân cụm dữ liệu
def cluster_data(df):
    features = [
        "price",
        "sold",
        "ratings",
        "followers",
        "discount",
        "location_id",
        "time_since_joining",
        "rating_per_review",
        "followers_to_product_ratio",
        "sales_volume",
        "trending_words_count",
    ]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)

    df["cluster"] = clusters

    # Trực quan hóa phân cụm
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x="price", y="sold", hue="cluster", data=df, palette="viridis")
    plt.title("Clusters based on Price and Sold Quantity")
    plt.show()

    return df


# Chạy quá trình phân tích và phân cụm
def main():
    df = get_data()
    df = clean_and_prepare_data(df)
    df = calculate_derived_features(df)
    df = cluster_data(df)
    print(df.head())


if __name__ == "__main__":
    main()
