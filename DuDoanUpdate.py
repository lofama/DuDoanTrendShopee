import numpy as np
import pandas as pd
import pymongo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import re
import tkinter as tk
from tkinter import ttk, messagebox


# Kết nối tới MongoDB để lấy dữ liệu
def connect_to_mongodb():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["ShopeeDB"]
    return db


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
        match = re.search(r"(\d+[,.\d+]*)k", sold_str)
        if match:
            return int(float(match.group(1).replace(",", ".")) * 1000)
        return int(re.sub(r"\D", "", sold_str))
    else:
        return sold_str  # Trả về giá trị nếu không phải chuỗi


# Hàm tính số ngày từ chuỗi joinTime
def calculate_days(join_time_str):
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


# Hàm làm sạch dữ liệu sản phẩm
def clean_data(item):
    try:
        if "price" in item:
            item["price"] = clean_price(item["price"])
        if "sold" in item:
            item["sold"] = clean_sold(item["sold"])
        if "ratings" in item:
            item["ratings"] = clean_sold(item["ratings"])
        if "followers" in item:
            item["followers"] = clean_sold(item["followers"])
        if "productsCount" in item:
            item["productsCount"] = clean_sold(item["productsCount"])
        if "joinTime" in item:
            item["joinTime"] = calculate_days(item["joinTime"])
        if "discount" in item:
            item["discount"] = (
                -int(item["discount"].strip("%")) if "discount" in item else 0
            )
        if "location" in item:
            item["location_id"] = get_location_id(item["location"])
        # Xử lý các trường khác nếu cần thiết
    except ValueError as e:
        print(f"Error processing item: {item} - {e}")
    return item


# Hàm lấy location_id từ MongoDB
def get_location_id(location):
    db = connect_to_mongodb()
    location_doc = db["LocationTable"].find_one({"location": location})
    if location_doc:
        return float(location_doc.get("id"))
    return 0


# Hàm tính tỉ lệ sale/follow
def sale_follow(item, newproduct):
    if newproduct["followers"] > 0:
        return item.get("sold", 0) / newproduct["followers"]
    else:
        return 0


# Chuẩn bị dữ liệu để dự đoán
def prepare_data(new_shop_data):
    X = []
    y = []
    product_names = []

    collect = connect_to_mongodb()
    data = list(collect["dataSetShoppee"].find())
    newproduct = clean_data(new_shop_data)

    for item in data:
        X.append(
            [
                item.get("keywords_count", 0),
                # item.get("discount", 0),
                item.get("price", 0),
                item.get("sold", 0),
                newproduct.get("responseRate", 0),
                item.get("ratings", 0),
                newproduct.get("productsCount", 0),
                newproduct.get("followers", 0),
                # sale_follow(item, newproduct),
                # item.get("tbRate/sale", 0),
            ]
        )
        product_names.append(item.get("name", ""))

        # Gán nhãn
        trending = item.get("trending", 0)
        y.append(1 if trending else 0)  # Nhãn là 1 nếu trending, ngược lại là 0

    X = np.array(X)
    y = np.array(y)

    return X, y, product_names


# Huấn luyện mô hình
def train_model():
    X, y, _ = prepare_data({})

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=0)
    model.fit(X_scaled, y)

    with open("logistic_regression_model.pkl", "wb") as model_file, open(
        "logistic_regression_scaler.pkl", "wb"
    ) as scaler_file:
        pickle.dump(model, model_file)
        pickle.dump(scaler, scaler_file)

    print("Đã lưu mô hình và scaler thành công.")


# Dự đoán sản phẩm mới
def predict_new_product(new_shop_data):
    model_filename = "CodeDLShoppee/logistic_regression_model.pkl"
    scaler_filename = "CodeDLShoppee/logistic_regression_scaler.pkl"

    with open(model_filename, "rb") as model_file, open(
        scaler_filename, "rb"
    ) as scaler_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)

    X, _, product_names = prepare_data(new_shop_data)

    X_scaled = scaler.transform(X)

    y_pred_prob = model.predict_proba(X_scaled)[:, 1]

    result_df = pd.DataFrame(
        {"product_name": product_names, "top_probability_percent": y_pred_prob * 100}
    )
    top_10 = result_df.sort_values(by="top_probability_percent", ascending=False).head(
        10
    )

    print(f"Top 10 sản phẩm có khả năng lên xu hướng:")
    print(top_10[["product_name", "top_probability_percent"]])
    return result_df.sort_values(by="top_probability_percent", ascending=False)


# Giao diện nhập dữ liệu và hiển thị kết quả
def create_gui():
    def submit_data():
        new_shop_data = {
            "name": name_entry.get(),
            "price": price_entry.get(),
            # "sold": sold_entry.get(),
            "location": location_entry.get(),
            "discount": discount_entry.get(),
            "ratings": ratings_entry.get(),
            "productsCount": products_count_entry.get(),
            "followers": followers_entry.get(),
        }

        result_df = predict_new_product(new_shop_data)
    # Clear existing entries in the tree view
        for item in tree.get_children():
            tree.delete(item)
        # Hiển thị kết quả
        for i, row in result_df.iterrows():
            tree.insert(
                "", "end", values=(row["product_name"], row["top_probability_percent"])
            )

    # Tạo cửa sổ chính
    root = tk.Tk()
    root.title("Shopee Product Trend Prediction")

    # Tạo các ô nhập liệu
    tk.Label(root, text="Name:").grid(row=0, column=0, sticky=tk.W)
    name_entry = tk.Entry(root)
    name_entry.grid(row=0, column=1)

    tk.Label(root, text="Price:").grid(row=1, column=0, sticky=tk.W)
    price_entry = tk.Entry(root)
    price_entry.grid(row=1, column=1)

    tk.Label(root, text="Sold:").grid(row=2, column=0, sticky=tk.W)
    sold_entry = tk.Entry(root)
    sold_entry.grid(row=2, column=1)

    tk.Label(root, text="Location:").grid(row=3, column=0, sticky=tk.W)
    location_entry = tk.Entry(root)
    location_entry.grid(row=3, column=1)

    tk.Label(root, text="Discount:").grid(row=4, column=0, sticky=tk.W)
    discount_entry = tk.Entry(root)
    discount_entry.grid(row=4, column=1)

    tk.Label(root, text="Ratings:").grid(row=5, column=0, sticky=tk.W)
    ratings_entry = tk.Entry(root)
    ratings_entry.grid(row=5, column=1)

    tk.Label(root, text="Products Count:").grid(row=6, column=0, sticky=tk.W)
    products_count_entry = tk.Entry(root)
    products_count_entry.grid(row=6, column=1)

    tk.Label(root, text="Followers:").grid(row=7, column=0, sticky=tk.W)
    followers_entry = tk.Entry(root)
    followers_entry.grid(row=7, column=1)

    # Tạo nút Submit
    submit_button = tk.Button(root, text="Submit", command=submit_data)
    submit_button.grid(row=8, column=0, columnspan=2)

    # Tạo bảng để hiển thị kết quả
    tree = ttk.Treeview(
        root, columns=("Product Name", "Top Probability (%)"), show="headings"
    )
    tree.heading("Product Name", text="Product Name")
    tree.heading("Top Probability (%)", text="Top Probability (%)")
    tree.grid(row=9, column=0, columnspan=2)

    root.mainloop()


if __name__ == "__main__":
    create_gui()
