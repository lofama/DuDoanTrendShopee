import pandas as pd
import pymongo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import re

# Kết nối đến MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["ShopeeDB"]
collection = db["dataSetShoppee"]

# Lấy dữ liệu từ bộ sưu tập MongoDB
data_from_mongo = list(collection.find())

# Lấy tên sản phẩm từ MongoDB
product_names = [item['name'] for item in data_from_mongo if 'name' in item]

# Chuyển đổi danh sách tên sản phẩm thành DataFrame
df = pd.DataFrame(product_names, columns=['product_name'])

# Định nghĩa stop words cho tiếng Việt
# Đọc stop words từ file
with open('D:/CodeThToan/CodeDLShoppee/stop_word.txt', 'r', encoding='utf-8') as file:
    stop_words_vietnamese = file.read().splitlines()

def read_keywords_from_file(file_path):
    """
    Đọc danh sách từ khóa từ file.

    Parameters:
    file_path (str): Đường dẫn tới file chứa danh sách từ khóa.

    Returns:
    list: Danh sách từ khóa.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        # keywords = [line.strip().lower() for line in file if line.strip().split()]
        keywords = [word.strip().lower() for line in file if line.strip() for word in line.strip().split()]
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
        if re.search(r'\b' + re.escape(keyword) + r'\b', product_name_lower):
            print(keyword,end=',')
            count += 1
    return count



# Hàm để tìm từ khóa xu hướng cho một danh mục
def find_trending_terms(category_df):

    # Đọc danh sách từ khóa từ file TuKhoa.txt
    file_path = 'D:\CodeThToan\CodeDLShoppee\ThoiTrangNam.txt'
    keywords = read_keywords_from_file(file_path)
    for i in category_df['product_name']:
        keywords_count = count_keywords_in_name(i, keywords)
        print(f"Số lượng từ khóa trong tên sản phẩm: {keywords_count}")
    # Tokenize tên sản phẩm và tính toán tần suất từ xuất hiện (Term Frequency)
    vectorizer = CountVectorizer(stop_words=stop_words_vietnamese)
    X = vectorizer.fit_transform(category_df['product_name'])

    # Tạo DataFrame từ kết quả tokenization
    word_freq_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # Tính toán tổng số lần xuất hiện của mỗi từ
    word_freq = word_freq_df.sum(axis=0).sort_values(ascending=False)

    # Lấy 10 từ khóa xu hướng
    top_terms = word_freq.head(10)
    return top_terms
top_terms = find_trending_terms(df  )
print(top_terms)
