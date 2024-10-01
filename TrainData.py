import re
import numpy as np
import pandas as pd
import pymongo
from bson import ObjectId
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


# Kết nối với mogoDB để lấy dữ liệu
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
def get_location_id(db, location):
    location_doc = db["LocationTable"].find_one({"location": location})
    if location_doc:
        return float(location_doc.get("id"))
    return None


def read_keywords_from_file(file_path):
    """
    Đọc danh sách từ khóa từ file.

    Parameters:
    file_path (str): Đường dẫn tới file chứa danh sách từ khóa.

    Returns:
    list: Danh sách từ khóa.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        # keywords = [line.strip().lower() for line in file if line.strip().split()]
        keywords = [
            word.strip().lower()
            for line in file
            if line.strip()
            for word in line.strip().split()
        ]
        keywords = set(keywords)
        print(keywords)
    return keywords


def count_keywords_in_name(product_name, keywords):
    """
    Đếm số lượng từ khóa xuất hiện trong tên sản phẩm.

    Parameters:
    product_name (str): Tên sản phẩm.
    keywords (list of str): Danh sách từ khóa.

    Returns:
    int: Số lượng từ khóa xuất hiện trong tên sản phẩm.
    """
    count = 0
    product_name_lower = product_name.lower()
    for keyword in keywords:
        if re.search(r"\b" + re.escape(keyword) + r"\b", product_name_lower):
            # print(keyword,end=',')
            count += 1
    return count


# Định nghĩa các đặc trưng và nhãn
# Xử lý dữ liệu
def preprocess_data(data_from_mongo, db):
    X = []
    y = []
    product_names = []

    # Đọc danh sách từ khóa từ file TuKhoa.txt
    file_path = "D:\CodeThToan\CodeDLShoppee\ThoiTrangNam.txt"
    keywords = read_keywords_from_file(file_path)
    for item in data_from_mongo:
        features = {}
        for key, value in item.items():
            if key == "lable":
                y.append(value)
            elif key == "name":
                keywords_count = count_keywords_in_name(value, keywords)
                # print(f"Số lượng từ khóa trong tên sản phẩm: {keywords_count}")
                features["keywords_count"] = keywords_count
                product_names.append(value)
            elif key not in ["name", "link"]:
                if key == "discount":
                    if value is not None:
                        value = value
                        num_value = convert_to_numeric(value)
                        if num_value is not None:
                            features[key] = num_value
                        else:
                            features[key] = 0
                else:
                    num_value = convert_to_numeric(value)
                    if num_value is not None:
                        features[key] = num_value
        if features:
            selected_keys = [
                "keywords_count",
                # "discount",
                "price",
                "sold",
                "responseRate",
                "ratings",
                "productsCount",
                "followers",
                # "sale/follow",
                # "tbRate/sale",
            ]
            feature_values = []
            for key in selected_keys:
                if key in features:
                    # if key == "price":
                    #     feature_values.append(features[key] / 1000)
                    # else:
                    #     feature_values.append(features[key])
                    feature_values.append(features[key])
                else:
                    feature_values.append(
                        0
                    )  # Hoặc giá trị mặc định khác phù hợp với dữ liệu của bạn
            X.append(feature_values)
    # Chuyển đổi X sang DataFrame
    X_df = pd.DataFrame(X)
    # Điền giá trị thiếu bằng 0 (hoặc bất kỳ chiến lược nào khác)
    X_df.fillna(0, inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    print(X_df.head(5))
    # return X_df, y, product_names, scaler
    return X_scaled, y, product_names, scaler


def hoiquyLogistic(X_train, y_train, X_test, y_test, names_test, scaler):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred_percent = y_pred_prob * 100
    # Tính toán các chỉ số đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # In ra các chỉ số đánh giá
    print(f"Độ chính xác Hồi quy Logistic: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print("Các hệ số của mô hình:")
    print(model.coef_)
    result_df = pd.DataFrame(
        {"product_name": names_test, "top_probability_percent": y_pred_percent}
    )
    top_10 = result_df.sort_values(by="top_probability_percent", ascending=False).head(
        10
    )
    print(top_10[["product_name", "top_probability_percent"]])
    filename = "CodeDLShoppee/logistic_regression_model.pkl"
    with open(filename, "wb") as file:
        pickle.dump(model, file)
    with open("CodeDLShoppee/logistic_regression_scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    return model


def DecisionTree(X_train, y_train, X_test, y_test, X_scaled, y):
    model = DecisionTreeClassifier()
    scores = cross_val_score(model, X_scaled, y, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Độ chính xác của từng fold:", scores)
    print("Độ chính xác trung bình:", np.mean(scores))
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"F1-score DecisionTree: {f1}")
    filename = "CodeDLShoppee/DecisionTree_model.pkl"
    with open(filename, "wb") as file:
        pickle.dump(model, file)
    return model


def neuralNetwork(X_train, y_train, X_test, y_test, X_scaled, y):
    model = MLPClassifier(max_iter=5000, solver="adam")
    scores = cross_val_score(model, X_scaled, y, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Độ chính xác của từng fold:", scores)
    print("Độ chính xác trung bình:", np.mean(scores))
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"F1-score neuralNetwork: {f1}")
    filename = "CodeDLShoppee/neural_network_model.pkl"
    with open(filename, "wb") as file:
        pickle.dump(model, file)
    return model


def supportVectorMachine(X_train, y_train, X_test, y_test, X_scaled, y):
    model = SVC()
    scores = cross_val_score(model, X_scaled, y, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Độ chính xác của từng fold:", scores)
    print("Độ chính xác trung bình:", np.mean(scores))
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"F1-score supportVectorMachine: {f1}")
    filename = "CodeDLShoppee/svm_model.pkl"
    with open(filename, "wb") as file:
        pickle.dump(model, file)
    return model


def randomForest(X_train, y_train, X_test, y_test, X_scaled, y):
    model = RandomForestClassifier()
    scores = cross_val_score(model, X_scaled, y, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Độ chính xác của từng fold:", scores)
    print("Độ chính xác trung bình:", np.mean(scores))
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"F1-score randomForest: {f1}")
    filename = "CodeDLShoppee/random_forest_model.pkl"
    with open(filename, "wb") as file:
        pickle.dump(model, file)
    return model


def main():
    data_from_mongo, db = connected()
    X_scaled, y, product_names, scaler = preprocess_data(data_from_mongo, db)
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X_scaled, y, product_names, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(names_train)}, Test samples: {len(names_test)}")

    logistic_model = hoiquyLogistic(
        X_train, y_train, X_test, y_test, names_test, scaler
    )
    # decision_tree_model = DecisionTree(X_train, y_train, X_test, y_test, X_scaled, y)
    # neural_network_model = neuralNetwork(X_train, y_train, X_test, y_test, X_scaled, y)
    # svm_model = supportVectorMachine(X_train, y_train, X_test, y_test, X_scaled, y)
    # random_forest_model = randomForest(X_train, y_train, X_test, y_test, X_scaled, y)


if __name__ == "__main__":
    main()
