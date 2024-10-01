import json
import pymongo

# Kết nối tới MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["ShopeeDB"]
collection = db["CleanedProducts"]


# Hàm để đọc dữ liệu từ bảng LocationTable
def read_location_table():
    location_table = list(db["LocationTable"].find({}, {"_id": 0}))
    return location_table


# Hàm để so sánh tên địa điểm đã chuẩn hóa với dữ liệu từ LocationTable
def compare_location(location_name):
    normalized_name = normalize_location(location_name)
    location_table = read_location_table()

    for loc in location_table:
        if normalized_name == loc["location"]:
            return loc["id"]

    return None  # Trả về None nếu không tìm thấy


# Hàm để chuẩn hóa tên địa điểm
def normalize_location(location):
    # return location.strip().lower()
    return location


# Hàm để trích xuất các địa điểm duy nhất từ dữ liệu JSON
def extract_unique_locations(data):
    all_locations = set()
    for item in data:
        if "location" in item and item["location"]:
            all_locations.add(item["location"])
    return list(all_locations)


# Hàm để tạo và lưu bảng location
def create_location_table(locations):
    print(locations)
    existing_locations = {loc["location"] for loc in read_location_table()}
    location_table = []
    if len(existing_locations) == 0:
        print("Không có")
    for loc in locations:
        if loc not in existing_locations:
            loc_record = {
                "id": compare_location(loc) or len(existing_locations) + 1,
                "location": loc,
            }
            location_table.append(loc_record)
            existing_locations.add(loc)
        else:
            print(f"{loc} already exists in the database.")

    if location_table:
        db["LocationTable"].insert_many(location_table)
        print("Location table created and stored successfully.")

# Hàm tải dữ liệu JSON từ tệp
def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

# Hàm chính để thực hiện quá trình trích xuất và tạo bảng location
def main():
    data = load_json_data("D:/CodeThToan/CodeDLShoppee/DataNew.json")
    locations = extract_unique_locations(data)
    create_location_table(locations)


if __name__ == "__main__":
    main()
