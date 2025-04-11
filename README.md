
#Table of Contents
1.	About the Project
2.	Key features
3.	Dataset
4.	Technologies Used
5.	How to Run
6.	Files
7.	Results

# FashionForwardFinder
The project develops a service using Neural Networks, applying recognition and classification of images, and similarity-search algorithms. 
Aims to meet the increasing demand for personalized suggestions in shopping experiences focused on e-commerce.

Key Features:

Deep Learning Model: Uses Keras Model/ResNet-50 for extracting deep visual features from images.
Similarity Matching: Applies Levenshtein distance for comparing product attributes.
Efficient Recommendation System: Generates real-time fashion suggestions based on user input.
Web-Based Interface: Displays results using HTML and serves recommendations in JSON format for seamless integration.
The project is ideal for fashion retailers, e-commerce platforms, and AI researchers interested in fashion technology advancements.

Dataset:

We use the DeepFashion dataset to train our model. One of its four benchmarks is the "Category and Attribute Prediction Benchmark." 
The dataset contains high-quality images of various clothing items, along with annotations and metadata that help in understanding 
visual similarities and fashion trends. 

You can access the dataset from the following link: https://drive.google.com/drive/folders/1AZTym6VgWcI-P5MDDwxaajMJxuVk1mZ0?usp=sharing.
There you will find the following:

-Images, string type, known in the data as 'filename', taken from Project/img (in the Google Drive link).

-Categories are integers (50), it contains the clothing category ID. Located in:
Project/Category and Attribute Prediction Benchmark/Anno_fine/train_cate.txt (in the Google Drive link).

-Attributes are binary; they look for the presence of a specific attribute. Located in 
Project/Category and Attribute Prediction Benchmark/Anno_fine/train_attr.txt (in the Google Drive link).

Technologies used:
NumPy/Pandas
TensorFlow.keras- ResNet50 
Scikit-Learn
Os
Matplotlib/Seaborn 
Requests
Levenshtein
JSON/HTML
Flask/OpenPyXL
Among others.


Files:

Version 1:
The V1 file contains our first approaches to data understanding and data processing.
Seen in the Colab file format: 'ProjectClassificator.v0' and 'ProjectClassificator.vresnet', both are the first versions run to choose which algorithm will be used for this project’s model, it contains the data loading and preprocessing, as well as its behavior through 10 epochs. 
As for the similarity search and the output with the recommendations, the file 'ProjectRecommendation1' is in the Colab files format. It contains our first approach to using the connection to Amazon with a string sample used in Canopy’s playground.

Version 2:
The V2 file contains our second approach to handling the data in local Python format.
The files are: ‘project_classificator_v1’ is a file in the Colab version that contains the model, the data preprocessing, training of the model, and the prediction for one specific image. The file ‘project_classificator_v2’  is an improved version of the model, its data preprocessing, and training of the model. (This is the version that we used until the final version, and it is already run in a local Python environment)
The  ‘model_with_recommendations’ contains the connection we use to connect with Amazon through Canopy, with its respective functions of the image preprocessing, and calculation of similarities. The  ‘test_model’ file contains the tests run to see the performance of the model on the prediction for a specific image, the code that saves the training history, and the graphs plotted to see the losses and accuracies of the model.
The files ‘config’, and ‘utilities’ are helper files. The first one has the Configurations of the model where the established parameters used in the model can be found. The last one contains the common functions used in the project.
Version 3 
The V3 file contains our third approach developing this project.
The files in this folder are ‘model’ which contains the dataset ‘train’ files loaded to train the model. The  ‘model_eval’ contains the model’s performance evaluation we decide to run for the model. Then, the ‘model_with_recommendations’ contains the connection of the model with Amazon through Canopy with the functions needed to connect it. The ‘test_model’ file contains a prediction we did using a specific image seeing the performance of the model from the prediction of the categories and accuracies until the ten recommendations similar to the input.
Related to ‘config’ and ‘utilities’ this are helper files. The first one has the Configurations of the model where can be found the stablished parameters used in the model. And the last one contains the common functions used in the project.
 ‘training_history.pkl’ is a file that helps us to save the training of our model through the history. This has the training until this third version of the project’s model.

Final Version
This folder contains the final version of the coding used within the project. The files and its content is as follows:

model
“””LOADING DATASET, DATA PREPROCESSING, MODEL TRAINING, SAVING MODEL’S HISTORY”””
model_with_recommendations
"""IMPLEMENTATION OF RECOMMENDATION V1. CONNECTING THE MODEL TO AMAZON: PROOF OF CONCEPT (POC)"""
model_training_history
"""GENERATING THE REPORT'S TRAINING HISTORY: GRAPHS AND EXCEL'S TRAINING"""
model_eval
"""EVALUATING MODEL'S CONFIDENCE. GENERATING EVALUATION'S REPORT AND GRAPH"""
model_unit_test.file
"""TESTING THE MODEL PREDICTION AND THE SEARCH_KEYWORD"""
config.file
"""Configurations of the model. Established parameters"""
utilities.file
"""COMMON FUNCTIONS USED IN THE PROJECT"""
demo.file
"""PROOF OF CONCEPT #2"""
The same code files can be seen in ipynb format in the folder Colab version within this Final Version folder.

To see the model and its history, the following link contains the final project's model version:
https://drive.google.com/drive/folders/1AZTym6VgWcI-P5MDDwxaajMJxuVk1mZ0?usp=drive_link (FashionForwardFinder.keras and training_hhistory.pkl)

# Instructions to Run Python Files

## Prerequisites

To execute this project, you need to install the following tools:

### 1. Python  
TensorFlow supports Python versions *3.8 to 3.12. We recommend installing **Python 3.11*.  
Download the appropriate installer for your system from the official Python website:  
[Python 3.11.5 Download](https://www.python.org/downloads/release/python-3115/)

### 2. Visual Studio Code  
A lightweight and free-to-use code editor.  
Download the suitable installer from the official Visual Studio Code website:  
[Visual Studio Code Download](https://code.visualstudio.com/download)

---

## Installing Dependencies  

After installing Python, open a terminal and run the following command to install the required Python packages:

sh
py -3.11 -m pip install numpy tensorflow matplotlib pillow

---

## Configuring the Base Path  

Modify the BASE_PATH variable in the script to match the location of your local project directory.

### Example:

python
BASE_PATH = "C:/Users/YourUsername/Documents/ProjectData/"

For Google Colab users, you might set it as:

python
BASE_PATH = "/content/drive/MyDrive/Project/"

---

## Running the Project  

To train the model, follow these steps:

1. Open *Visual Studio Code*.
2. Open a new terminal inside Visual Studio Code.
3. Navigate to the project directory.
4. Run the following command:

sh
py -3.11 .\projectclassificator_v1.py

This will start the training process for your model.

to execute second attempt to train the model 
py -3.11 .\project_classificator_v2.py

this will save the resulting model in the same directory the files is being executed.
to test the the exported model you need to execute the file  test_model

py -3.11 .\test_model.py

config.py is the file that contains all the global config variables defined across the proyect
these define some of the parameter to train the model

MODEL_SAVE_PATH -> defined the name of the resulting model (this is a full model save)
BASE_PATH -> the base path the project will start looking for the train files and images
BATCH_SIZE -> batch size used to train the model (32 is the suggested value)
EPOCHS -> defines the quantity of epochs to train the model (10 is the suggested value)
MAX_HEIGHT -> image resize value for height  (restnet suggest a value of 224)
MAX_WIDTH -> image resize value for width  (restnet suggest a value of 224)

#Results

After training the model, open a terminal and run the following command to install the required Python packages to plot graphs and Excel versions with the results:

py -3.11 -m pip install scikit-learn
py -3.11 -m pip install seaborn
py -3.11 -m pip install openpyxl


#Demo

To have access to the demo, it is needed to run a terminal with the Flask package.
py -3.11 -m pip install flask


COMMANDS:

py -3.11 .\model.py
py -3.11 .\model_eval.py
py -3.11 .\model_with_recommendations.py
py -3.11 .\model_unit_test.py
PY -3.11 .\utilities.py
py -3.11 .\config.py
py -3.11 .\demo.py

