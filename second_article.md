## **Introduction**
In the previous article, we learnt about Machine Learning, the importance of Machine Learning Operations (MLOps) in Machine Learning Lifecycle. We also learnt how to frame a problem statement, gather required data, build Machine Learning model, evaluate the model and save the model for future use. 


In this article, we will look into the processes of deploying a Machine Learning Model with Flask and how to containerize Machine Learning Apis with Docker. 

### **Model Deployment**
We have already discussed about Model Deployment in the previous article, but just to give a recap. Model Deployment is the process of making Machine Learning models accessible to others through different interfaces like Website, Mobile Phones, Embedded Systems e.t.c. 
Flask and Swagger will be used to deploy the model in this article, but what is Flask and Swagger?

#### **What on earth is Flask?**
According to Wikipedia 
> Flask is a micro web framework written in Python. It is classified as a microframework because it does not require particular tools or libraries.

Flask is simply a framework that helps python developers to build websites, without the need to learn Javascript. With flask, you can build a full-fledged website with python, HTML & CSS.  

##### **Getting Started with Flask**
Now that we understand the functionality of flask, let us dive into a practical example.
You will need an Integrated Development Environment like [Visual Studio Code](https://code.visualstudio.com/), [PyCharm](https://pycharm-community-edition.en.softonic.com/download) e.t.c to follow along. You will also need to install Flask, you can install Flask by navigating to your command prompt/terminal and type 
```py
pip install flask
```  
Now that we have all the neccessary tools to work with flask, let's dive into flask. We will write a simple program that will return your name with a greeting. The code below will 
- Import the Flask class from the flask module
-  Initialize a Flask() object with the name of the current file `__name__`, `__name__` will help python to locate where to load other dependencies into (in this case into our script). You can read this article from [StackOverflow](https://stackoverflow.com/questions/39393926/flaskapplication-versus-flask-name#:~:text=1%20Answer&text=__name__%20is%20just,files%2C%20instance%20folder%2C%20etc.) to understand the purpose of `__name__`
-  declare a variable to store my name
-  initialize a decorator, the decorator will be called when users click on the homepage url of the website  (this is what is called when you clcik on www.google.com).
- define a function that binds to the decorator (the decorator will return the function whenever it is called). The function can return an HTML File or just a text.
- write an if conditional statement that retricts the execution of the application to the current script(i.e if you import the script into another script the application won't run). You can read this [article](https://www.geeksforgeeks.org/what-does-the-if-__name__-__main__-do/#:~:text=if%20__name__%20%3D%3D%20%E2%80%9Cmain%E2%80%9D%3A%20is%20used%20to,run%20directly%2C%20and%20not%20imported.) from GeeksForGeeks to understand the importance of `if __name__== '__main__'`  
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

Copy the code above and save it inside a file, you can name the file my_flask.py. Now navigate to where you saved the file using the command line, and type python my_flask.py.
```py
python my_flask.py
```  
or just by clicking on the run icon if you are using Visual Studio Code.


![image](https://user-images.githubusercontent.com/38056084/135066441-db79fa6c-41b1-44c6-97a6-e02577eda28e.png)
Fig. 1 How to run python file with VSCode (Image by Author)

After runing the file either with the command line/terminal or with VSCode you will be presented with this screen.

![image](https://user-images.githubusercontent.com/38056084/135067355-dfd5fbfe-59e9-433d-824c-cb2c95643e49.png)
Fig. 2 Image displaying information about flask (Image by Author)

Click on the link (http://127.0.0.1:5000/), once you click on the link you will presented with this screen below.
![image](https://user-images.githubusercontent.com/38056084/135072139-4e8a7318-257b-4c89-bc83-506836e3ef35.png)
Fig.3 Image showing the output of the greet function (Image by Author)
When you clicked on the link (http://127.0.0.1:5000/), flask automatically called the `@app.route('/')` decorator, that is becasue you are trying to access the homepage (which is `/`). The @app.route decorator will automaticall call the `greet` function and the output of the function will be returned. You will notice that, we had to manually handcode the variable name into our script, which is not intuitive. What if we want to accept the name from users and return `Hello + user_name`, we can do this by creating a HTML form that accepts users name, send the name to our script and return `Hello + name`. 