import matplotlib.pyplot as plt
import pickle
import pandas as pd
from config import *

def plot_training_history(history):
    """
    Plots training curves from a Keras history-like dictionary.

    Expected keys:
    - 'loss'
    - 'category_output_accuracy'
    - 'category_output_loss'
    - 'attribute_output_binary_accuracy'
    - 'attribute_output_loss'
    """

    epochs = list(range(1, len(history['loss']) + 1))

    plt.figure(figsize=(18, 5))

    # 1. Combined Losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['loss'], label='Total Loss', linewidth=2)
    plt.plot(epochs, history['category_output_loss'], label='Category Loss', linestyle='--')
    plt.plot(epochs, history['attribute_output_loss'], label='Attribute Loss', linestyle='--')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 2. Category Accuracy + Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['category_output_accuracy'], label='Accuracy', color='green', linewidth=2)
    plt.plot(epochs, history['category_output_loss'], label='Loss', color='red', linestyle='--')
    plt.title("Category Output Performance")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # 3. Attribute Accuracy + Loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['attribute_output_binary_accuracy'], label='Accuracy', color='blue', linewidth=2)
    plt.plot(epochs, history['attribute_output_loss'], label='Loss', color='orange', linestyle='--')
    plt.title("Attribute Output Performance")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def generate_training_table(history, output_path="static/training_report.xlsx"):
    """
    Converts a Keras-like history dict to a pandas DataFrame and saves it to Excel.
    
    Parameters:
    - history: dict containing training history
    - output_path: where to save the Excel file
    """
    num_epochs = len(history['loss'])

    table_data = {
        'Epoch': list(range(1, num_epochs + 1)),
        'Total Loss': history['loss'],
        'Category Accuracy': history['category_output_accuracy'],
        'Category Loss': history['category_output_loss'],
        'Attribute Accuracy': history['attribute_output_binary_accuracy'],
        'Attribute Loss': history['attribute_output_loss'],
    }

    df = pd.DataFrame(table_data)
    df.to_excel(output_path, index=False)
    print(f"✅ Training report saved to: {output_path}")
    return df

with open(MODEL_HISTORY_SAVE_PATH, "rb") as f:
    history_data = pickle.load(f)

print(f"History Data: {history_data}")
    
df = generate_training_table(history_data)

print(df.head())
plot_training_history(history_data)