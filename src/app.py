from json import loads, dumps
from flask import Flask, make_response, request
from model import Model

model = Model('../notebooks/model.joblib', 
              '../notebooks/params.pkl')
app = Flask(__name__)

@app.route("/")
def ping():
    return "<h1 style='color:blue'>It works!</h1>"

@app.route("/predict", methods=['POST'])
def predict_endpoint():    
    data = loads(request.data, strict=False)['data']
    preds = model.predict(data)
    return make_response(dumps(preds))

if __name__ == "__main__":
    app.run(debug=True)