import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Kết nối tới MongoDB để lấy dữ liệu
def connect_to_mongodb():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["ShopeeDB"]
    return db


# Lấy dữ liệu từ MongoDB
def get_data():
    db = connect_to_mongodb()
    data = list(db["CleanedProducts"].find())
    return pd.DataFrame(data)


# Hàm tính hàm phụ thuộc và vẽ đồ thị cho từng sản phẩm
def calculate_dependency(df):
    df = df[["sold", "followers", "rcount_with_context", "product_id"]].dropna()
    product_ids = df["product_id"].unique()

    for product_id in product_ids:
        product_df = df[df["product_id"] == product_id]
        X = product_df[["followers", "rcount_with_context"]]
        y = product_df["sold"]

        if len(X) < 2:
            continue  # Skip products with insufficient data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"Sản phẩm ID: {product_id}")
        print(f"Độ chính xác của mô hình: {accuracy}")

        print("Hệ số của mô hình:")
        for feature, coef in zip(X.columns, model.coef_):
            print(f"{feature}: {coef}")

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"MSE trên tập kiểm tra: {mse}\n")

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x="followers", y="sold", data=product_df)
        plt.title(f"Số lượng bán theo số lượng follow (Sản phẩm ID: {product_id})")
        plt.xlabel("Số lượng follow")
        plt.ylabel("Số lượng bán")

        plt.subplot(1, 2, 2)
        sns.scatterplot(x="rcount_with_context", y="sold", data=product_df)
        plt.title(f"Số lượng bán theo số lượt bình luận (Sản phẩm ID: {product_id})")
        plt.xlabel("Số lượt bình luận")
        plt.ylabel("Số lượng bán")

        plt.tight_layout()
        plt.show()


def main():
    df = get_data()
    calculate_dependency(df)


if __name__ == "__main__":
    main()
