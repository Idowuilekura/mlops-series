- [**Introduction**](#introduction)
  - [**Model Deployment**](#model-deployment)
    - [**What on earth is Flask?**](#what-on-earth-is-flask)
      - [**Getting Started with Flask**](#getting-started-with-flask)
      - [**Getting Started with Swagger**](#getting-started-with-swagger)
    - [**Deploying Machine Learning Model with Flask and Swagger**](#deploying-machine-learning-model-with-flask-and-swagger)
  - [**Conclusion**](#conclusion)
    - [**Reference/ Resources**](#reference-resources)
## **Introduction**
In the previous [article](https://dev.to/idowuilekura1/mlops-deploying-machine-learning-models-with-docker-and-google-cloud-platform-part-1-37m2), we learned about Machine Learning, the importance of Machine Learning Operations (MLOps) in the Machine Learning Lifecycle. We also learned how to frame a problem statement, gather required data, build a Machine Learning model, evaluate the model and save the model for future use. 


In this article, we will look into the process of deploying a Machine Learning Model with Flask and Swagger.

### **Model Deployment**
We have already discussed Model Deployment in the previous article, but to give a recap. Model Deployment is the process of making Machine Learning models accessible to others through different interfaces like Websites, Mobile Phones, Embedded Systems e.t.c. 
Flask and Swagger will be used to deploy the model in this article, but what are Flask and Swagger?

#### **What on earth is Flask?**
According to Wikipedia 
> Flask is a micro web framework written in Python. It is classified as a microframework because it does not require particular tools or libraries.

Flask is simply a framework that helps python developers to build websites, without the need to learn Javascript. With flask, full-fledged websites can be built with python, HTML & CSS  

##### **Getting Started with Flask**
Now that we understand the functionality of flask, let us dive into a practical example.
You will need an Integrated Development Environment like [Visual Studio Code](https://code.visualstudio.com/), [PyCharm](https://pycharm-community-edition.en.softonic.com/download) e.t.c to follow along. You will also need to install Flask, you can install Flask by navigating to your command prompt/terminal and type 
```py
pip install flask
```  
Now that we have the necessary tools to work with flask, let's dive into flask. We will write a simple program that will return your name with a greeting. The code below will 
- Import the Flask class from the flask module
-  Initialize a Flask() object with the name of the current file `__name__`, `__name__` will help python to locate where to load other dependencies into (in this case into our script). You can read this article from [StackOverflow](https://stackoverflow.com/questions/39393926/flaskapplication-versus-flask-name#:~:text=1%20Answer&text=__name__%20is%20just,files%2C%20instance%20folder%2C%20etc.) to understand the purpose of `__name__`
-  declare a variable to store my name(replacee with your name)
-  initialize a decorator, the decorator will be called when users click on the homepage URL of the website  (this is what is called when you click on www.google.com). You can read this [article](https://www.datacamp.com/community/tutorials/decorators-python) from Datacamp to learn more about Decorators.
- define a function that binds to the decorator (the decorator will return the function whenever it is called). The function can return a HTML File or just a text.
- write an if conditional statement that restricts the execution of the application to the current script(I.e if you import the script into another script the application won't run). You can read this [article](https://www.geeksforgeeks.org/what-does-the-if-__name__-__main__-do/#:~:text=if%20__name__%20%3D%3D%20%E2%80%9Cmain%E2%80%9D%3A%20is%20used%20to,run%20directly%2C%20and%20not%20imported.) from GeeksForGeeks to understand the importance of `if __name__== '__main__'`  
```py
# Importing the Flask module from flask library
from flask import Flask

# initializing our application with the name of our module
app = Flask(__name__)
# Variable to store my name( change to your name)
my_name = 'idowu'

# Initializing the homepage route
@app.route('/')
def greet(): # function that will be returned
    return 'Hello ' + my_name

if __name__=='__main__':
    app.run(debug=True) 
```

Copy the code above and save it inside a file, you can name the file my_flask.py . Now navigate to where you saved the file using the command line, and type python my_flask.py.
```py
python my_flask.py
```  
or just by clicking on the run icon if you are using Visual Studio Code.


![image](https://user-images.githubusercontent.com/38056084/135066441-db79fa6c-41b1-44c6-97a6-e02577eda28e.png)
Fig. 1 How to run python file with VSCode (Image by Author)

After runing the file either with the command line/terminal or with VSCode you will be presented with this screen.

![image](https://user-images.githubusercontent.com/38056084/135067355-dfd5fbfe-59e9-433d-824c-cb2c95643e49.png)
Fig. 2 Image displaying information about flask (Image by Author)

Click on the link (http://127.0.0.1:5000/),once you click on the link you will see the screen below.
![image](https://user-images.githubusercontent.com/38056084/135072139-4e8a7318-257b-4c89-bc83-506836e3ef35.png)
Fig.3 Image showing the output of the greet function (Image by Author)

When you clicked on the link (http://127.0.0.1:5000/), flask automatically called the `@app.route('/')` decorator, this is because you are trying to access the homepage (which is `/`). The @app.route decorator will automatically call the `greet` function and the output of the function will be returned. You will notice that, we hardcoded the variable name into our script, which is not intuitive. What if we want to accept the name from users and return `Hello + user_name`, 
we can do this by creating another index for the name.
We need to rewrite the programs in our script.
The code below will 
- Bind a welcome function to the home route (`/`) 
- Bind a different function greet_name to the `/greet/name` decorator. This decorator will allow users to insert their names ( this is similar to www.google.com/search/query). www.google.com is the home route url while `/search` is for the search route.`


```py
from flask import Flask
app = Flask(__name__)

@app.route("/")
def welcome():
    return 'Hello welcome to my first flask website'

@app.route("/greet/<name>")
def greet_name(name):
    return 'Hello ' + name

if __name__=='__main__':
    app.run(debug=True)
```
Delete the previous codes in my_flask.py and copy the above code into my_flask.py (ensure you save the file)

Now try to run `my_flask.py` and click on the homepage url.
```
python my_flask.py
```
You will be presented with the screen below
![image](https://user-images.githubusercontent.com/38056084/135098506-fd19dad8-11de-4ad6-9d5c-78c94183e589.png)
Fig. 4 Home page of the website (Image by Author)

Now, for users to insert their names, the users will need to add `/greet/their_name` after the homepage url i.e (http://127.0.0.1:5000/greet/their_username).

While still on the homepage, try and add `/greet/your_name[replace with your name]`
![image](https://user-images.githubusercontent.com/38056084/135100295-ee41b9b9-31c0-4750-92ec-8e7ef79c6946.png)
Fig. 5 Image showing the response from greet route


You will agree with me that this process is quite tedious. What if we can have an interface that accepts responses from a form, sends the responses to flask and return the responses. To achieve this, we can build a form with HTML that accepts parameters. Luckily, instead of writing a HTML program, we can leverage the Swagger module to do that.

> Swagger allows you to describe the structure of your APIs so that machines can read them. The ability of APIs to describe their own structure is the root of all awesomeness in Swagger. Why is it so great? Well, by reading your API’s structure, we can automatically build beautiful and interactive API documentation.
[Source](https://swagger.io/docs/specification/2-0/what-is-swagger/)

##### **Getting Started with Swagger**
You will need to install flasgger module, which can be done with 
```py
pip install flasgger
```
The code below will
- Import the request and Flask module from flask. 
- Import swagger from flasgger
- Initialize the Flask object 
- Wrap the flask object application with Swagger. This helps our application to inherit properties from Swagger.
- create a decorator for our homepage url
- create a decorator for our greet url. If you notice we have something different from what we have been defining previously. The `@app.route()` decorator for the greet url takes in a methods parameter with the value Get. There are two major methods that `@app.route` receives
  - The Get Method (The get method is used when you want to receive something, when you navigate to www.google.com you are indirectly calling the Get method to return the HTML file for the homepage)
  - The Post Method is used to send information to a server. You can read this [article](https://www.geeksforgeeks.org/get-post-requests-using-python/) from GeeksforGeeks to learn more about Post and Get Methods.
- Inside the `greet_name()` function, you will have to use docstring to define the Swagger interface. 
  - The interface expects the;
    - title of the query
    - information about the query
    - parameters that will be inputted. The paramter fileds expects  
      - the name of the field
      - the mode in which the parameter will be entered, it can be manually inserted as values(query) or by inserting a path to the value.
      - the type of the parameter ( it can be an integer or string)
      - if the parameter is compulsory or not( if it is required then the parameter can't be ommitted) 
    - the responses. 
  
- *N.B* Make sure you indent the docstring with four spaces, also you will need to indent the details under the parameters and responses else swagger won't render the display.
- To access the user_name under the parameters, you will need to use the request module to access the parameter
- Lastly, you can return the greeting and the user_name.

Copy the code below into my_flask.py
```py
from flask import Flask, request 
from flasgger import Swagger 
app = Flask(__name__)
Swagger(app)

@app.route("/")
def welcome():
    return 'Hello, welcome to my first flask website'


@app.route('/greet', methods=["Get"])
def greet_name():
    """Greetings
    This is using docstrings for specifications.
    ---
    parameters:
        - name: user_name
          in: query
          type: string
          required: true
    responses:
      200:
         description: greetings with user's name
    """

    user_name = request.args.get("user_name")

    print(user_name)

    return "Hello " + user_name

if __name__=='__main__':
    app.run(debug=True)
```

Now, try to run my_flask.py with 
```py
python my_flask.py
```
![image](https://user-images.githubusercontent.com/38056084/135226479-150500d8-8e41-46e9-80d2-33d2eb986939.png)
Fig. 6 Image showing the homepage (Image by Author)

To access the Swagger User Interface, append `/apidocs` after the homepage url i.e `http://127.0.0.1:5000/apidocs`

![image](https://user-images.githubusercontent.com/38056084/135227423-a33e3573-1585-4c68-8169-3701cb696601.png)
Fig. 7 Image showing the Swagger UI (Image by Author)

To interact with the UI, click on the GET button, which will present you with this screen 

![image](https://user-images.githubusercontent.com/38056084/135227925-9fa390c7-f847-48c8-add8-7a6937adf95c.png)
Fig. 8 Image showing the UI under the GET button. 

If you notice, you can't insert any parameter inside the user_name, to insert value into the user_name field you will need to click on the Try it out button. 

![image](https://user-images.githubusercontent.com/38056084/135227925-9fa390c7-f847-48c8-add8-7a6937adf95c.png)

You can now insert any name into the user_name field and once you are click on the `Execute` button.

![image](https://user-images.githubusercontent.com/38056084/135229761-36f5881f-f2bd-4072-aac4-be759ac5c752.png)
Fig. 9 Image showing the Response 

The response body display's the response from the greet function. Now that we have all the prerequisite for deploying our model, we can move n to cracking our main task which is to deploy a house price prediction model for Mowaale. 

#### **Deploying Machine Learning Model with Flask and Swagger**
To successfuly deploy the model, we will need to build a simple pipeline that will receive users inputs and make prediction.
We need to rewrite our Swagger program, to accommodate more infromation.
The code below will
- import Flask & request from flask
- import Swagger from flasgger
- import joblib(this will be useful later)
- load the previously saved label encoder for sale condition and save into lb_salecond variable 
- load the previsouly saved label endcoder for sale type and save into lb_saletype variable
- load the previoulsy saved linear regression model and save into model variable
- Initialize our Flask application object
- Wrap the application object with Swagger 
- define a route for the homepage.
- define another route for the predict_price index.
- define the name of each parameters and other information that was discussed previously. The default is to specify a default parameter and the enum is to create a dropdown list of values.
- use the request module to get each parameters that was inputed.
- transform the salecondition from a string to a number (this was discussed in part 1)
- transform the sale type from a string it a number.
- store all the parameters into a list
- use the model to make predictions and return the predicted price for the house.
Copy the code into my_flask.py
```py
from flask import Flask, request 
from flasgger import Swagger
import joblib 
lb_salecond = joblib.load('lb_sc')
lb_saletype = joblib.load('lb_st') 
model = joblib.load('lr_model')
app = Flask(__name__)
Swagger(app)


@app.route("/")
def welcome():
    return 'Hello welcome to my first flask website'

['Normal', 'Partial', 'Abnorml', 'Family', 'Alloca', 'AdjLand']
@app.route('/predict_price', methods=["Get"])
def predict_prices():
    """Welcome to Moowale House Prediction Website
    This is using docstrings for specifications.
    ---
    parameters:
        - name: MSSubClass
          in: query
          type: integer
          required: true
        - name: LotFrontage
          in: query
          type: integer
          required: true
        - name: Year_sold
          in: query
          type: integer
          required: true
        - name: Sale_type
          in: query
          type: string
          required: true
          default: New
          enum: ['WD', 'New', 'COD', 'ConLD', 'ConLI', 'ConLw', 'CWD', 'Oth', 'Con']
        - name: Sale_condition
          in: query
          type: string
          default: Normal
          enum: ['Normal', 'Partial', 'Abnorml', 'Family', 'Alloca', 'AdjLand']
    responses:
      200:
         description: House Price Prediction
    """

    mssubclass = request.args.get("MSSubClass")
    lotfrontage = request.args.get("LotFrontage")
    year_sold = request.args.get("Year_sold")
    saletype = request.args.get("Sale_type")
    salecondition = request.args.get("Sale_condition")

    label_encode_sc = lb_salecond.transform([salecondition])
    label_encode_st = lb_saletype.transform([saletype])

    columns_list = [mssubclass,lotfrontage,year_sold,label_encode_sc,label_encode_st]

    price = round(model.predict([columns_list])[0])
    return f'The predicted price is ${price:.0f}'

if __name__=='__main__':
    app.run(debug=True)
```
Run my_flask.py,click on the url and attach `/apidocs` after the homepage url. You will see the screen below,

![image](https://user-images.githubusercontent.com/38056084/135260572-c54e24f5-6a57-4adc-98f9-94d6af7ce503.png)
Fig. 10 Image showing the UI of moowale (Image by Author) 

Click on Try it out, and insert these values for MSSUbClass insert 20, for Lotfrontage insert 80, for Year_sold insert 2007, for Sale type & SaleCondition leave as the default and click on Execute.

![image](https://user-images.githubusercontent.com/38056084/135262953-c9f028e6-eec0-48f8-9371-012a5651a94d.png)
Fig.11 Image showing the prediction (Image by Author)

Now that we have our model deployed, we have come to the end of the second article in this series.

### **Conclusion**
This article has introduced you to the process of deploying machine learning models with flask and building interactive visuals with Swagger. In part3 of this series, you will learn how to containerize the Api with Docker and Deploy on Google Cloud Platform. 
You can connect with me on [Linkedin](https://www.linkedin.com/in/ilekuraidowu/)  

#### **Reference/ Resources**
Video from Krish Naik [channel](https://www.youtube.com/watch?v=8vNBW98LbfI&list=PLZoTAELRMXVNKtpy0U_Mx9N26w8n0hIbs&index=3)

[Flasgger](https://github.com/flasgger/flasgger)
