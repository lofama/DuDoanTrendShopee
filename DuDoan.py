import numpy as np
import pandas as pd
import pymongo
from bson import ObjectId
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import re


# Kết nối với MongoDB để lấy dữ liệu
def connected():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["ShopeeDB"]
    collection = db["dataSetShoppee"]
    data_from_mongo = list(collection.find())
    return data_from_mongo, db


# Hàm chuyển đổi giá trị sang dạng số nếu có thể
def convert_to_numeric(value):
    try:
        if isinstance(value, ObjectId):
            return None
        num = float(value)
        return num
    except (ValueError, TypeError):
        return None


# Hàm ánh xạ location sang một ID từ Locationtable
def get_location_id(location):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
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


# Hàm làm sạch dữ liệu sản phẩm mới
def clean_data(item):
    try:
        if "price" in item:
            item["price"] = clean_price(item["price"])
        if "originalPrice" in item:
            item["originalPrice"] = clean_price(item["originalPrice"])
        if "sold" in item:
            item["sold"] = clean_sold(item["sold"])
        if "ratings" in item:
            item["ratings"] = clean_sold(item["ratings"])
        if "followers" in item:
            item["followers"] = clean_sold(item["followers"])
        if "productsCount" in item:
            item["productsCount"] = clean_sold(item["productsCount"])
        if "joinTime" in item:
            item["joinTime"] = convert_to_numeric(item["joinTime"])
        if "location" in item:
            item["location_id"] = get_location_id(item["location"])
        # Đảm bảo các đặc trưng không bị thiếu
        item["discount"] = (
            -int(item["discount"].strip("%")) if "discount" in item else 0
        )
        item["location"] = hash(item["location"]) % 10000 if "location" in item else 0
    except ValueError as e:
        print(f"Error processing item: {item} - {e}")
    return item


# Định nghĩa các đặc trưng và nhãn
def preprocess_data(data_from_mongo, db):
    X = []
    y = []
    product_names = []
    for item in data_from_mongo:
        features = {}
        for key, value in item.items():
            if key == "label":
                y.append(value)
            elif key == "location":
                location_id = get_location_id(db, value)
                if location_id is not None:
                    features["location_id"] = location_id
            elif key == "name":
                product_names.append(value)
            elif key not in ["name", "link"]:
                if key == "discount":
                    if value is not None:
                        value = value.split("%")[0]
                        num_value = -convert_to_numeric(value)
                        if num_value is not None:
                            features[key] = -num_value / 100
                        else:
                            features[key] = 0
                else:
                    num_value = convert_to_numeric(value)
                    if num_value is not None:
                        features[key] = num_value
        if features:
            X.append(features)
    # Chuyển đổi X sang DataFrame
    X_df = pd.DataFrame(X)
    print(X_df.head(5))
    # Điền giá trị thiếu bằng 0 (hoặc bất kỳ chiến lược nào khác)
    X_df.fillna(0, inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    return X_scaled, y, product_names, scaler


def hoiquyLogistic(X_train, y_train, X_test, y_test, names_test, scaler):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred_percent = y_pred_prob * 100
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Độ chính xác Hồi quy Logistic: {accuracy}")
    print("Các hệ số của mô hình:")
    print(model.coef_)
    result_df = pd.DataFrame(
        {"product_name": names_test, "top_probability_percent": y_pred_percent}
    )
    top_10 = result_df.sort_values(by="top_probability_percent", ascending=False).head(
        10
    )
    print(top_10[["product_name", "top_probability_percent"]])
    model_filename = "CodeDLShoppee/logistic_regression_model.pkl"
    scaler_filename = "CodeDLShoppee/logistic_regression_scaler.pkl"
    with open(model_filename, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(scaler_filename, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    return (model,)


def load_model_and_predict(product, model_filename):
    with open(model_filename, "rb") as model_file:
        model = pickle.load(model_file)

    product = clean_data(product)
    print(product)
    selected_keys = [
        "location_id",
        # "discount",
        "price",
        # "sold",
        "rating",
        "originalPrice",
        "ratings",
        "productsCount",
        "followers",
    ]
    # feature_values = [value for key, value in product.items() if key in selected_keys]
    feature_values = []
    for key in selected_keys:
        if key in product:
            feature_values.append(product[key])
        else:
            feature_values.append(
                0
            )  # Hoặc giá trị mặc định khác phù hợp với dữ liệu của bạn

    print(feature_values)
    feature_values = np.array(feature_values).reshape(1, -1)
    prediction = model.predict(feature_values)
    prediction_prob = model.predict_proba(feature_values)
    return prediction, prediction_prob


def main():
    model_filename = "CodeDLShoppee/logistic_regression_model.pkl"
    # Đọc mô hình từ file .pkl
    with open(model_filename, "rb") as file:
        loaded_model = pickle.load(file)
    # Dự đoán với sản phẩm mới
    new_product = {
        "name": "Áo khoác gió nam chống nước chống nắng Ultralight Windbreaker",
        "link": "",
        "price": "199.000",
        "originalPrice": "₫\n399.000",
        "sold": "Đã bán 700",
        "location": "Hồ Chí Minh",
        "rating": 4.7,
        "discount": "-50%",
        "shopName": "Ultralight Store",
        "onlineStatus": "Online 10 phút trước",
        "ratings": "1,1k",
        "responseRate": "98%",
        "joinTime": "3 năm trước",
        "productsCount": "85",
        "responseTime": "trong vài giờ",
        "followers": "88,5k",
    }
    # scaler_filename = 'CodeDLShoppee/logistic_regression_scaler.pkl'
    prediction, prediction_prob = load_model_and_predict(new_product, model_filename)
    print(f"Prediction: {prediction}, Probability: {prediction_prob}")


if __name__ == "__main__":
    main()
