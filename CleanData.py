import json
import re
import numpy as np
import pymongo
from sklearn.model_selection import train_test_split


# Hàm kết nối đến MongoDB và trả về đối tượng client
def connect_to_mongodb(uri="mongodb://localhost:27017/"):
    client = pymongo.MongoClient(uri)
    return client


# Hàm tải dữ liệu JSON từ tệp
def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


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


def read_code_ids():
    file_path = (
        r"D:\Propose_Products_For_Sale-master\CloneCode\id.txt"
    )
    codeid = []

    with open(file_path, mode="r", encoding="utf-8") as f:
        for line in f:
            codeid.append(
                line.strip()
            )  # Thêm mã sản phẩm vào mảng, loại bỏ dấu xuống dòng (\n)

    return codeid


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


# Hàm loại bỏ các ngoại lệ dựa trên IQR
def remove_outliers(data, key):
    values = np.array([item[key] for item in data if key in item])
    if len(values) == 0:
        return data
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [
        item
        for item in data
        if key not in item or (item[key] >= lower_bound and item[key] <= upper_bound)
    ]


# Hàm tính số lượng bán trung bình từ dữ liệu JSON đã tải
def calculate_average_sales(data):
    data = remove_outliers(data, "sold")
    total_sales = sum(clean_sold(item.get("sold", 0)) for item in data)
    average_sales = total_sales / len(data) if data else 0
    return average_sales


# Hàm tính số lượt bán trung bình trên số follow của shop
def calculate_average_sales_per_follower(data):
    data = remove_outliers(data, "sold")
    data = remove_outliers(data, "followers")
    total_sales_per_follower = sum(
        clean_sold(item.get("sold", 0)) / max(clean_sold(item.get("followers", 1)), 1)
        for item in data
    )
    average_sales_per_follower = total_sales_per_follower / len(data) if data else 0
    return average_sales_per_follower


# Hàm tính trung bình lượt bình luận trên số lượt bán nhân rate trung bình <5
def calculate_average_comments_per_sales_rate(data):
    data = remove_outliers(data, "sold")
    total_comments_per_sales_rate = 0
    count = 0
    for item in data:
        if item.get("rating", 0) <= 5:
            total_comments_per_sales_rate += (
                item.get("rcount_with_context", 0)
                / clean_sold(item.get("sold", 0))
                * item.get("rating")
            )
            count += 1
    average_comments_per_sales_rate = (
        total_comments_per_sales_rate / count if count > 0 else 0
    )
    return average_comments_per_sales_rate


# Hàm gán nhãn
def isTrend(product, data):
    tbRate = calculate_average_comments_per_sales_rate(data)
    tbinhSold = calculate_average_sales(data)
    tbFollow = calculate_average_sales_per_follower(data)
    return check_conditions(product, tbFollow, tbinhSold, tbRate)


def check_conditions(item, tbFollow, tbinhSold, tbRate):
    conditions_met = 0
    codeid = read_code_ids()
    for code in codeid:
        if code in item["link"]:
            print(item["link"])
            conditions_met += 1
    if item["sale/follow"] >= tbFollow:
        conditions_met += 1
    if item["sold"] >= tbinhSold:
        conditions_met += 1
        # return 1
    if item["tbRate/sale"] >= tbRate:
        conditions_met += 1
    if conditions_met >= 2:
        return 1
    else:
        return 0


# Hàm làm sạch dữ liệu
def clean_data(data):
    dataSet = []
    for item in data:
        try:
            if "price" in item:
                item["price"] = clean_price(item["price"])
            if "originalPrice" in item:
                item["originalPrice"] = clean_price(item["originalPrice"])
            if "sold" in item:
                item["sold"] = clean_sold(item["sold"])
                # Làm sạch discount
            if "discount" in item:
                item["discount"] = (
                    convert_to_numeric(item["discount"]) / 100
                )  # Chuyển đổi về dạng thập phân
            # Làm sạch responseRate
            if "responseRate" in item:
                item["responseRate"] = (
                    convert_to_numeric(item["responseRate"]) / 100
                )  # Chuyển đổi về dạng thập phân
            if "location" in item:
                item["location"] = get_location_id(item["location"])
            if "ratings" in item:
                item["ratings"] = clean_sold(item["ratings"])
            if "followers" in item:
                item["followers"] = clean_sold(item["followers"])
            if "joinTime" in item:
                item["joinTime"] = calculate_days(item["joinTime"])
            if "productsCount" in item:
                item["productsCount"] = clean_sold(item["productsCount"])
            if "rating_count" in item and len(item["rating_count"]) == 5:
                item["4-5_star_ratings"] = (
                    item["rating_count"][4] + item["rating_count"][3]
                )
            item["sale/follow"] = 1.0 * item["sold"] / item["followers"]
            item["tbRate/sale"] = (
                item.get("rcount_with_context", 0)
                / clean_sold(item.get("sold", 0))
                * item.get("rating")
            )
        except ValueError as e:
            print(f"Error processing item: {item} - {e}")
        dataSet.append(item)
    return dataSet


# Hàm tạo dataset với nhãn 'isTrend' dựa trên số lượng bán
def create_dataset(data):
    dataset = []
    tbRate = calculate_average_comments_per_sales_rate(data)
    tbinhSold = calculate_average_sales(data)
    tbFollow = calculate_average_sales_per_follower(data)
    print("average_sales_per_follower: ", tbFollow)
    print("tbinhSold: ", tbinhSold)
    print("calculate_average_comments_per_sales_rate", tbRate)
    count = 0
    # Define columns to drop
    columns_to_drop = [
        "shopName",
        "ratings",
        "responseRate",
        "joinTime",
        "productsCount",
        "responseTime",
        "followers",
    ]
    columns_to_keep = [
        "name",
        "link",
        "price",
        "originalPrice",
        "sold",
        "location",
        "rating",
        "discount",
        "ratings",
        "responseRate",
        "joinTime",
        "productsCount",
        "followers",
        "rating_total",
        "rating_count",
        "4-5_star_ratings",
        "rcount_with_context",
        "sale/follow",
        "tbRate/sale",
    ]
    # Drop unnecessary columns
    # data_filtered = data.drop(columns=columns_to_drop, errors='ignore')
    for item in data:
        if isTrend(item, data):
            # print(item['name'])
            count += 1
            # Remove unnecessary columns from the item
        filtered_item = {
            key: value for key, value in item.items() if key in columns_to_keep
        }

        if all(
            key in item
            for key in [
                "name",
                "link",
                "price",
                "originalPrice",
                "sold",
                "location",
                "rating",
                "discount",
                "ratings",
                "responseRate",
                "joinTime",
                "productsCount",
                "followers",
                "rating_total",
                "rating_count",
                "rcount_with_context",
                "sale/follow",
                "tbRate/sale",
            ]
        ):
            # is_trend = 1 if item['sold'] > 1000 else 0
            is_trend = isTrend(item, data)
            filtered_item["lable"] = is_trend
            dataset.append(filtered_item)
            """
            dataset.append({
                'name': item['name'],
                'location': item['location'],
                'discount': item['discount'],
                'price': clean_price(item['price']),
                'link': item['link'],
                'sold': clean_sold(item['sold']),
                'rating': item['rating'],
                'originalPrice': clean_price(item['originalPrice']),
                'shopName': item['shopName'],
                'ratings': clean_sold(item['ratings']),
                'responseRate': item['responseRate'],
                'joinTime': item['joinTime'],
                'productsCount': clean_sold(item['productsCount']),
                'responseTime': item['responseTime'],
                'followers': clean_sold(item['followers']),
                'label': is_trend
            })"""
    print("Có số sản phẩm đạt:", count)
    return dataset


# Hàm chính
def main():
    # Kết nối đến MongoDB
    client = connect_to_mongodb()
    db = client["ShopeeDB"]
    collection = db["CleanedProducts"]
    dataset_collection = db["dataSetShoppee"]  # Collection để lưu trữ dataset
    # Tải dữ liệu JSON từ tệp
    # data = load_json_data('D:/CodeThToan/CodeDLShoppee/Infor_product_and_shop_detail.json')
    data = load_json_data("D:/CodeThToan/CodeDLShoppee/DataNew.json")

    # Trích xuất và in ra các khóa duy nhất
    unique_keys = extract_keys(data)
    print("Unique keys found in JSON data:")
    print(unique_keys)

    # Làm sạch dữ liệu
    cleaned_data = clean_data(data)

    # Chèn dữ liệu đã làm sạch vào MongoDB
    for item in cleaned_data:
        existing_product = collection.find_one({"link": item["link"]})
        if existing_product:
            # print(f"Product already exists with link: {item['link']}")
            print(1, end=".")
        else:
            collection.insert_one(item)

    # Tạo dataset và lưu vào MongoDB
    dataset = create_dataset(cleaned_data)
    for item in dataset:
        existing_dataset_item = dataset_collection.find_one({"link": item["link"]})
        if existing_dataset_item:
            # print(f"Dataset item already exists with link: {item['link']}")
            print(2, end=".")
        else:
            dataset_collection.insert_one(item)
            # print(3, end=".")
    print()
    print("Dataset saved successfully to dataSetShoppee collection.")


# Gọi hàm chính
if __name__ == "__main__":
    main()
