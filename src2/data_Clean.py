import pandas as pd

train_file_path = 'C:/MyProject_Study04/AirfarePrediction_AI/data/raw/Data_Train.xlsx'
test_file_path = 'C:/MyProject_Study04/AirfarePrediction_AI/data/raw/Test_set.xlsx'

#Doc File
try:
    train_data = pd.read_excel(train_file_path)
    test_data = pd.read_excel(test_file_path)

    print("Train Data:")
    print(train_data.head(3))
    print("\nTest Data:")
    print(test_data.head(3))

except FileNotFoundError as e:
    print(f"File khong ton tai: {e}")
except Exception as e:
    print(f"Co loi xay ra: {e}")

# Kiem tra cac gia tri thieu trong du lieu Train
print("\nKiem tra cac gia tri thieu trong Train Data:")
print(train_data.isnull().sum())

# Kiem tra cac gia tri thieu trong du lieu Test
print("\nKiem tra cac gia tri thieu trong Test Data:")
print(test_data.isnull().sum())

# Loai bo gia tri thieu
train_data = train_data.dropna()
test_data = test_data.dropna()

print("\nDu lieu test sau khi lam sach:")
print(train_data.info())

print("\nDu lieu test sau khi lam sach:")
print(test_data.info())

train_data.to_csv('../data/processed/cleaned_train_data.csv', index=False)
test_data.to_csv('../data/processed/cleaned_test_data.csv', index=False)

print("\nLam sach du lieu Xong!")
