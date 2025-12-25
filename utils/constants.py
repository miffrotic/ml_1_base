from pathlib import Path


BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "selling_price_model.pkl"
PAGES_PATH = BASE_DIR / "pages"
DF_TRAIN_URL = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
DF_TEST_URL = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv"
