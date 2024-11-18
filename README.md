# House Price Prediction Using Linear Regression
This project implements a linear regression model to predict house prices based on features such as square footage, the number of bedrooms, and bathrooms. The goal is to use the training dataset to build a model and generate predictions for the test dataset.

**Project Structure**
Files:
* train.csv: Training dataset containing house features and corresponding prices.
* test.csv: Test dataset with house features (without prices) for prediction.
* sample_submission.csv: Sample format for the submission file.
Outputs:
* submission.csv: File containing predicted house prices for the test dataset.

**Steps to Run the Project**
Environment Setup:

Open the project in Google Colab.
Install necessary libraries if not already installed (e.g., scikit-learn, pandas, numpy, seaborn, matplotlib).
Upload Datasets:

Upload train.csv, test.csv, and sample_submission.csv to Colab.
Execute Code:

Run the provided code sequentially.
The workflow includes loading the data, preprocessing, training a linear regression model, evaluating performance, and predicting house prices.
Visualizations:

Pair plots, correlation heatmaps, and residual plots are included to help analyze data and model performance.
Generate Submission:

After predictions are made on the test dataset, download the submission.csv file.
Dependencies
Ensure the following libraries are installed:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
  
To install missing libraries, run:

bash

`!pip install pandas numpy matplotlib seaborn scikit-learn`

**Key Features**
Data Preprocessing:

Handles missing values in the training data.
Splits data into training and validation sets.
Model Training:

Implements a LinearRegression model from scikit-learn.
Evaluates model performance using Mean Squared Error (MSE).
Visualizations:

Analyzes feature relationships and residual distributions.
Compares actual vs. predicted values.
How to Interpret Results
Model Performance:

The Mean Squared Error (MSE) on the validation set indicates how well the model fits the training data.
Visualizations:

Scatter plots and heatmaps reveal data trends.
Residual analysis checks the assumptions of linear regression.

**Predictions:**
submission.csv contains predicted house prices for the test dataset, formatted for submission.
