import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "fashionfowardfinder.keras") 
BASE_PATH = "D:/Project/"
BATCH_SIZE = 32
EPOCHS = 10
MAX_HEIGHT = 224
MAX_WIDTH = 224