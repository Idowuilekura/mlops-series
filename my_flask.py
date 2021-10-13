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