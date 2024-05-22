# HeartFailurePrediction


## A Web Application to predict Heart-Disease using SVM  (Deployed on Heroku)


### 1.Project Requirements or Dependencies
* Anaconda Python (to get ML Libraries)
* Pip install flask (For Front-end)
### 2. Load Dataset
Heart-Disease-Prediction Original Data Set
### 3.Build and Train the model using SVM

Using SVM (Support Vector Machines) we build and train a model using human cell records, and classify cells to predict whether the samples are Effected or Not-Affected.

### 4.Flask Creation

1.	Heart-Disease-Prediction.ipynb — This contains code for the machine learning model to predict heart disease based on the class.
2.	app.py — This contains Flask APIs that receives cells details through GUI or API calls, computes the predicted value based on our model and returns it
3.	templates & static  — This folders contains the HTML template and CSS styling to allow user to enter cells details and displays the predicted output.

### 5.Backend creation using model.pkl file

Use this pretrained model and connect it with our Flask application.
Use this for prediction for model and to show the output

### 6. Adding form to flask app
 
### 7.Integrating web application with machine learning backend.


