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

Files:

Version 1:
The V1 file contains our first approaches to the data understanding and data processing. Both for the modeling of the data (feature extraction) 
seen in the Colab files format: 'ProjectClassificator.v0' and 'ProjectClassificator.vresnet', 
as well as for the similarity search and the output with the recommendations 'ProjectRecommendation1' in the Colab files format.

Version 2:
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


This will start the training process for your model.

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
MAX_WIDTH -> image resize value for width  (restnet suggest a value of 224)

