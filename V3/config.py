import os

AMAZON_API_KEY = "eeda37d1-b608-408c-aafd-9f9d48a63d64"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "fashionfowardfinder.keras") 
MODEL_HISTORY_SAVE_PATH = os.path.join(SCRIPT_DIR, "training_history.pkl")
BASE_PATH = "D:/Project/"
TEST_IMAGE_PATH = BASE_PATH + "img/Classic_Faded_Jeggings/img_00000055.jpg" 
BATCH_SIZE = 32
EPOCHS = 10
MAX_HEIGHT = 224
MAX_WIDTH = 224
MAX_SUGGESTIONS = 10
ATTRIBUTES_SENSIBILITY = 0.5


CATEGORY_NAMES = {
    1: {'desc': 'Anorak', 'cat_type': 1}, 2: {'desc': 'Blazer', 'cat_type': 1}, 3: {'desc': 'Blouse', 'cat_type': 1},
    4: {'desc': 'Bomber', 'cat_type': 1}, 5: {'desc': 'Button Down', 'cat_type': 1}, 6: {'desc': 'Cardigan', 'cat_type': 1},
    7: {'desc': 'Flannel', 'cat_type': 1}, 8: {'desc': 'Halter', 'cat_type': 1}, 9: {'desc': 'Henley', 'cat_type': 1},
    10: {'desc': 'Hoodie', 'cat_type': 1}, 11: {'desc': 'Jacket', 'cat_type': 1}, 12: {'desc': 'Jersey', 'cat_type': 1},
    13: {'desc': 'Parka', 'cat_type': 1}, 14: {'desc': 'Peacoat', 'cat_type': 1}, 15: {'desc': 'Poncho', 'cat_type': 1},
    16: {'desc': 'Sweater', 'cat_type': 1}, 17: {'desc': 'Tank', 'cat_type': 1}, 18: {'desc': 'Tee', 'cat_type': 1},
    19: {'desc': 'Top', 'cat_type': 1}, 20: {'desc': 'Turtleneck', 'cat_type': 1}, 21: {'desc': 'Capris', 'cat_type': 2},
    22: {'desc': 'Chinos', 'cat_type': 2}, 23: {'desc': 'Culottes', 'cat_type': 2}, 24: {'desc': 'Cutoffs', 'cat_type': 2},
    25: {'desc': 'Gauchos', 'cat_type': 2}, 26: {'desc': 'Jeans', 'cat_type': 2}, 27: {'desc': 'Jeggings', 'cat_type': 2},
    28: {'desc': 'Jodhpurs', 'cat_type': 2}, 29: {'desc': 'Joggers', 'cat_type': 2}, 30: {'desc': 'Leggings', 'cat_type': 2},
    31: {'desc': 'Sarong', 'cat_type': 2}, 32: {'desc': 'Shorts', 'cat_type': 2}, 33: {'desc': 'Skirt', 'cat_type': 2},
    34: {'desc': 'Sweatpants', 'cat_type': 2}, 35: {'desc': 'Sweatshorts', 'cat_type': 2}, 36: {'desc': 'Trunks', 'cat_type': 2},
    37: {'desc': 'Caftan', 'cat_type': 3}, 38: {'desc': 'Cape', 'cat_type': 3}, 39: {'desc': 'Coat', 'cat_type': 3},
    40: {'desc': 'Coverup', 'cat_type': 3}, 41: {'desc': 'Dress', 'cat_type': 3}, 42: {'desc': 'Jumpsuit', 'cat_type': 3},
    43: {'desc': 'Kaftan', 'cat_type': 3}, 44: {'desc': 'Kimono', 'cat_type': 3}, 45: {'desc': 'Nightdress', 'cat_type': 3},
    46: {'desc': 'Onesie', 'cat_type': 3}, 47: {'desc': 'Robe', 'cat_type': 3}, 48: {'desc': 'Romper', 'cat_type': 3},
    49: {'desc': 'Shirtdress', 'cat_type': 3}, 50: {'desc': 'Sundress', 'cat_type': 3}
}

ATTRIBUTE_NAMES = {
    1: {'desc': 'floral', 'apply_to': []}, 2: {'desc': 'graphic', 'apply_to': []}, 3: {'desc': 'striped', 'apply_to': []},
    4: {'desc': 'embroidered', 'apply_to': []}, 5: {'desc': 'pleated', 'apply_to': []}, 6: {'desc': 'solid', 'apply_to': []},
    7: {'desc': 'lattice', 'apply_to': []}, 8: {'desc': 'long sleeve', 'apply_to': [1]}, 9: {'desc': 'short sleeve', 'apply_to': [1]},
    10: {'desc': 'sleeveless', 'apply_to': [1]}, 11: {'desc': 'maxi length', 'apply_to': [3]}, 12: {'desc': 'mini length', 'apply_to': [3]},
    13: {'desc': 'no dress', 'apply_to': [3]}, 14: {'desc': 'crew neckline', 'apply_to': [1]}, 15: {'desc': 'v neckline', 'apply_to': [1]},
    16: {'desc': 'square neckline', 'apply_to': [1]}, 17: {'desc': 'no neckline', 'apply_to': [1]}, 18: {'desc': 'denim', 'apply_to': []},
    19: {'desc': 'chiffon', 'apply_to': []}, 20: {'desc': 'cotton', 'apply_to': []}, 21: {'desc': 'leather', 'apply_to': []},
    22: {'desc': 'faux', 'apply_to': []}, 23: {'desc': 'knit', 'apply_to': []}, 24: {'desc': 'tight', 'apply_to': []},
    25: {'desc': 'loose', 'apply_to': []}, 26: {'desc': 'classic', 'apply_to': []}
}

