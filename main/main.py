from flask import Flask, request, render_template, jsonify
from flask import Response
import os
from flask_cors import CORS,cross_origin
import requests
from training_model.training import module
from spacy_file.spacy_dir import spa_
from sentence_transformers import SentenceTransformer



app = Flask(__name__)
CORS(app)

model = SentenceTransformer('all-MiniLM-L6-v2')
@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    svm = module.training_module()
    text = ' '.join(data)
    word = spa_.spacy_tokenizer(text)
    emb = model.encode(word)
    values = svm.predict(emb.reshape(1, -1))
    if values == 0:
        return jsonify("bussiness")
    elif values == 1:
        return jsonify("tech")
    elif values==2:
        return jsonify("politics")
    elif values ==3:
        return jsonify("sport")
    else:
        return jsonify("entertainment")


@app.route('/predict',methods=['POST'])
def predict():
    data = list(request.form.values())
    svm=module.training_module()
    text = ' '.join(data)
    word=spa_.spacy_tokenizer(text)
    emb=model.encode(word)
    values = svm.predict(emb.reshape(1, -1))
    if values==0:
        a="bussiness"
    elif values==1:
        a="tech"
    elif values==2:
        a="politics"
    elif values ==3:
        a="sport"
    else:
        a="entertainment"
    return render_template("home.html", prediction_text="The News is of Following {}".format(a))


port = int(os.getenv("PORT",5000))
if __name__ == "__main__":
    app.run(port=port,debug=True,host="0.0.0.0")





