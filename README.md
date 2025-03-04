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

