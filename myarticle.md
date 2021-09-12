## Introduction 
Machine Learning is the process of training machines to become intelligent and solve tasks with minimal supervision. You might be amazed at how YouTube intelligently recommends videos for you or how Spotify magically suggests new songs to you. What powers the intelligent recommendation or magically suggested songs is simply machine learning. Humans learned how to walk by observing how others walked and tried to replicate the process. Similarly, machines learn to perform tasks by iteratively following the patterns in a given dataset and then forming a mathematical function. Data scientists can use the mathematical function to make predictions on new datasets. As shown in the image below,  training datasets are fed into the machine learning model. The model learns the patterns and develops a mathematical function (h). The mathematical function is then applied to a new dataset to make predictions. You can read this [article](https://www.analyticsvidhya.com/machine-learning/) to gain a better understanding of machine learning.
<figure>
<img src="https://user-images.githubusercontent.com/38056084/132355664-318bf4a5-81f2-40ef-b18f-4cf9cf5a6122.png",alt="Model training image" width="600" height="500">
<figcaption>Fig.1 How machine learning works <a href="https://www.google.com/url?sa=i&url=https%3A%2F%2Fmachinelearningmedium.com%2F2017%2F08%2F10%2Fmodel-representation-and-hypothesis%2F&psig=AOvVaw2-DcUOC1eomtACOEfe_cUb&ust=1631108564271000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCPiwkIP_7PICFQAAAAAdAAAAABAI">source</a></figcaption>
</figure>
<!-- ![image](https://user-images.githubusercontent.com/38056084/132355664-318bf4a5-81f2-40ef-b18f-4cf9cf5a6122.png) -->
 
 The Youtube recommendation model was built and then deployed on Youtube for users to enjoy. Imagine if Google built the recommendation model without deploying it on Youtube? That would be awkward, and we wouldn't be able to enjoy Youtube recommendations as it it today. Unfortunately, most data scientists are comfortable with just building models without deploying them to production for end-users to enjoy. Machine Learning Operations (MLOps) is simply the process of shipping your machine learning models to production (this is just a basic definition). According to Wikipedia MLOps is  
> MLOps is a set of practices that aims to deploy and maintain machine learning models in production reliably and efficiently.

As shown in the image below, MLOps is an iterative method, that encompasses the:

  - Design Phase
  - Model Development Phase
  - Operations Phase

![image](https://user-images.githubusercontent.com/38056084/132360717-0fed95b8-e37b-4795-b298-f7c9d6b21af5.png)
Fig.2 MLOps Phases [source](https://www.google.com/url?sa=i&url=https%3A%2F%2Fml-ops.org%2Fcontent%2Fmlops-principles&psig=AOvVaw1SVPdhuwUk_caQVpTfQJNm&ust=1631110733104000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCKC-uo2H7fICFQAAAAAdAAAAABAD)




## **Design Phase** 
Under the Design Phase we have other sub-phases, which are:
1. Requirement Engineering 
2. ML Use-Cases and Prioritization
3. Data Availability Check 
### **Requirements Engineering**

 According to Javatpoint 
 > Requirements Engineering (RE) refers to the process of defining, documenting, and maintaining requirements in the engineering design process. 
 
 Requirement engineering helps an individual/ company understand the desires of the customers. It involves understanding the customers' needs through analysis, assessing the feasibility of a solution/product, settling for a better solution, clearly framing the solution, validating the specifications, managing any requirements before transforming it into a working system. 
 
 ![image](https://user-images.githubusercontent.com/38056084/132573929-5e729bd6-d3ec-46d0-846d-e686830ca741.png)

 Fig.3 Requirement Engineering Process [source](https://www.javatpoint.com/software-engineering-requirement-engineering)

The first step in any Data Science/ Machine learning lifecycle is requirement engineering. This phase is where you understand your customers' needs. For instance, Spotify was able to build a recommendation model because this is what most customers need. You can read more about requirements engineering [here](https://www.javatpoint.com/software-engineering-requirement-engineering) 

 ### **ML Use Cases and Prioritization**
This phase is where Machine Learning practitioners prioritize the impact of any proposed machine learning solution on their business. This phase is concerned with analyzing the importance of a proposed machine learning solution towards their goals. The truth is some proposed machine learning solutions aren't profitable to an organization. In this phase, companies drop those solutions.
You can read this [article](https://www.linkedin.com/pulse/identifying-prioritizing-artificial-intelligence-use-cases-srivatsan/) from Srivatsan Srinivasan,a Google Developer Expert; the report will give you more understanding of prioritizing Machine Learning Use Cases.

### **Data Availability Check**

This phase deals with the process of checking if the data that will be used to train the model is available. It's obvious that you can't train any machine learning model without data. 

## **Model Development Phase**

The model development phase deals with training the model with the data, performing feature engineering, testing the performance of the model using different metrics like accuracy, precision, recall e.t.c


## **Operations Phase**

This phase deals with deploying the model to production, monitoring the performance of the model in the production environment. You can either deploy the model using a continuous integration/continuous development (CI/CD) pipeline or without. 

The focus of the articles in this series is on building the model and deploying the model with Docker and Google Cloud Platform. 

Machine Learning models are built to solve problems,hence before building any machine learning model, a problem statement needs to be defined.

## Problem Statement
You just resumed, as a datascientist at moowale, moowale is a real estate company that lists houses for sale. You have been tasked with the responsibility of building a machine learning model that can be used to predict the price of any building based on some given parameters. The model will be deployed to make it easy for customers to get predicted prices based on information that they input into moowale website.

*N.B moowale is a Yoruba word meaning I am in search of a house [to buy].*


## Data Gathering
Now that you have a well defined problem sta- [Introduction](#introduction)
- [Introduction](#introduction)
- [**Design Phase**](#design-phase)
  - [**Requirements Engineering**](#requirements-engineering)
  - [**ML Use Cases and Prioritization**](#ml-use-cases-and-prioritization)
  - [**Data Availability Check**](#data-availability-check)
- [**Model Development Phase**](#model-development-phase)
- [**Operations Phase**](#operations-phase)
- [Problem Statement](#problem-statement)
- [Data Gathering](#data-gathering)
- [Model Building](#model-building)
- [Conclusion](#conclusion)


## Model Building
It's time to build our model, but before we train our model we will need to explore the datastet (this is called exploratory data analysis). To follow along you will need to have an Integrated Development Environment like Jupyter Notebook/Lab (this is what most datascientist use). You can learn more about jupyter notebook with this [link](https://www.youtube.com/watch?v=HW29067qVWk).
Now that you have jupyter notebook, you can open it and follow along. 
To explore the dataset, we will be using some python libraries which we will import.

```py
# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 

```
Pandas will be used to read the dataset, numpy is a numerical module, matplotlib is a plotting module and %matplotlib inline helps jupyter notebook to render plots within the notebook. 

Now that the necessary modules have been imported, the next step is to load and visualize the dataset.

```py
# reading the dataset with pandas
dataset = pd.read("train.csv")

# Exploring the first five rows
data.head() 
```
![image](https://user-images.githubusercontent.com/38056084/132589280-efbb4a6e-82c3-4ede-8e60-14c4eeb8432e.png)
Image of the first five rows (Image by Author)

```py
# printing the number of columns 
print(f"There are {len(data.columns)} columns in the dataset")
```
![image](https://user-images.githubusercontent.com/38056084/132666670-3cde770a-9fc6-4724-94ed-560729fb6ecb.png)

There are 81 columns(or features)  in the dataset, not all columns are useful for model building. You will need to rely on picking the necessary columns to be used to train the model. Datascientist either relies on domain knowledge gained from experience or by consulting with a domain expert in the field of interest to choose useful columns. For this task, I will just pick columns based on assumptions. Columns that will be chosen are `'MSSubClass',"LotFrontage','YrSold','SaleType','SaleCondition','SalePrice`
```py
data_new = data[['MSSubClass','LotFrontage','YrSold','SaleType','SaleCondition','SalePrice']]
```
```py
# checking for missing values
data_new.info()
```
![image](https://user-images.githubusercontent.com/38056084/132671268-d46c1d17-b2f9-4cff-97b1-e8d9296a159a.png)

The LotFrontage has missing values, since the column is of float datatype, the missing values can be filled with the column mean, column median, or a constant number.

```py
data_new['LotFrontage'] = data_new['LotFrontage'].fillna(-9999)

# check for missing values again
data_new.info()
```
![image](https://user-images.githubusercontent.com/38056084/132673332-ea688182-a712-45e9-b0c3-3cea297d6e34.png)

There are no more missing values, the next phases in model building.

If you look closely at the `data_new.info()` result you will notice that we have two columns with object datatype.
Machine Learning models expect all columns to be in numerical format, hence the need to convert the columns to numerical. LabelEncoder is widely used to convert categorical columns to numerical columns, you can read more about LabelEncoder [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
```py
# using label encoder to encode the categorical columns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# saletype label encoder
lb_st = LabelEncoder() 
# salecondition label encoder
lb_sc = LabelEncoder()

lb_st.fit(data_new['SaleType'])
lb_sc.fit(data_new['SaleCondition'])

data_new['SaleType'] = lb_st.transform(data_new['SaleType'])
data_new["SaleCondition"] = lb_sc.transform(data_new['SaleCondition'])
```
The `.fit()` method encodes each category to a number and the `.transform()` method transforms the category to number (you need to fit before you can transform).

Now that all columns are in numerical format, it's time to build the machine learning model. Machine Learning Engineers/ Datascientists usually split dataset into train and test set (the train set is used to train the model while the test set is used to evaluate the performance of the model). To split the dataset into train and test, the features that will be used to train the model must be separated from the target.
```py
# Separating the dataset into features(X) and target(y)
X = data_new.drop("SalePrice",axis=1)
y = data_new['SalePrice']
``` 
The features are named X and the target is named y because machine learning is basically mathematical functions applied on data. Recall from mathematics that  *f*X= y, the target(y) is computed by applying a mathematical function on your features (X) (this is what happens in machine learning or during model training)

```py
# splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y)
```
Now that we have our train and test dataset, it's time to build our model. There are two major types of machine learning tasks, Classification and Regression. Regression deals with numerical targets while classification deals with categorical targets. 
The target is SalePrice which is numerical,hence a regression model will be used. 
There are different types of regression models, Linear Regression is the simplest to understand and beginner-friendly regression model. LinearRegression model will be used in this project to keep it simple), you can use other regression models like RandomForestRegressor etc.
```py
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

# training the model
lr.fit(X_train,y_train)
```
Recall that after building your model, you need to evaluate the performance of the model. Several metrics can be used to evaluate machine learning models, for regression models metrics such are mean squared error, mean absolute error, root mean squared error e.t.c. Root Mean Squared Error metric will be used to evaluate the model.

```py
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(lr.predict(X_test),y_test))

# this will give approximately 73038
```

The Root Mean Squared Error is quite large, machine learning engineers/ datascientist tries to reduce the error by trying other models and applying feature engineering. I won't be diving into that, the aim of the article is not to minimize the error.

Now that we have our model, to use the model in production you will need to save the model, the two label encoder that was initialized earlier. 

```py
# joblib is ued to save object for later use
import joblib
joblib.dump(lr,'lr_model')

joblib.dump(lb_sc,'lb_sc')

joblib.dump(lb_st,'lb_st')
```
`joblib.dump` accepts two default parameters the object that you want to save and the name that you wish to save the object with.

Now that we have our model saved, we have finally come to the end of the first article in this series.

## Conclusion
This article has introduced you to machine learning, the importance of machine learning operation in the Machine Learning Lifecycle. How to frame a problem statement, gather data, build a linear regression model, evaluate the model and finally save the model for future use. In Part2 of this series, you will learn how to build machine learning APIs with flask, test the APIs and containerize the API with docker. 