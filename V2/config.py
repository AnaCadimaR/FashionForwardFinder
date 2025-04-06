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
ATTRIBUTES_SENSIBILITY = 0.6

CATEGORY_NAMES = {
    1:'Anorak', 2:'Blazer', 3:'Blouse', 4:'Bomber', 5:'Button-Down', 6:'Cardigan', 7:'Flannel', 8:'Halter', 9:'Henley', 10:'Hoodie',
    11:'Jacket', 12:'Jersey', 13:'Parka', 14:'Peacoat', 15:'Poncho', 16:'Sweater', 17:'Tank', 18:'Tee', 19:'Top', 20:'Turtleneck', 21:'Capris',
    22:'Chinos', 23:'Culottes', 24:'Cutoffs', 25:'Gauchos', 26:'Jeans', 27:'Jeggings', 28:'Jodhpurs', 29:'Joggers', 30:'Leggings', 31:'Sarong',
    32:'Shorts', 33:'Skirt', 34:'Sweatpants', 35:'Sweatshorts', 36:'Trunks', 37:'Caftan', 38:'Cape', 39:'Coat', 40:'Coverup', 41:'Dress',
    42:'Jumpsuit', 43:'Kaftan', 44:'Kimono', 45:'Nightdress', 46:'Onesie', 47:'Robe', 48:'Romper', 49:'Shirtdress', 50:'Sundress'}

ATTRIBUTE_NAMES = {
    1: "floral", 2: "graphic", 3: "striped", 4: "embroidered", 5: "pleated", 6: "solid", 7: "lattice",
    8: "long_sleeve", 9: "short sleeve", 10: "sleeveless", 11: "maxi length", 12: "mini length", 13: "",
    14: "crew neckline", 15: "v neckline", 16: "square neckline", 17: "", 18: "denim", 19: "chiffon",
    20: "cotton", 21: "leather", 22: "faux", 23: "knit", 24: "slim fit", 25: "oversize", 26: "classic"}
