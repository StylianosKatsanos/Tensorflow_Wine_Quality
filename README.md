# Tensorflow_Wine_Quality
Python script that utilizes the tensorflow library to create a prediction model based on a csv file about features and quality of wines


Based on the red wine csv file from the modeling data with title:
"Modeling wine preferences by data mining from physicochemical properties."
published in 2009
also accessible from the uci Machine Learning Repository,
url : "https://archive.ics.uci.edu/ml/datasets/wine+quality"

This Python scripts uses the tensorflow library to compile a Neural Network 
using a portion of the data, and evaluating it with the use of a test sample

The scripts starts with the processing of the data and the creation of the trainning and test samples

It first normalizes the data and it passes them through a Network with 2 layers

After compiling the model and trainning it with the data,
plots are used to visualize the correlation between the Mean Absolute Error and the Mean Square Error values and each epoch

Finally, predictions are generated with the test data and compared with their true values through plotting. 
