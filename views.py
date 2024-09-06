from flask import Blueprint, render_template, request
import time
import requests


views = Blueprint(__name__, "views")


@views.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        question = request.form['question']
        candidate_code = request.form['candidate_code']
        ai_response = request.form['ai_response']
        # output = predict_genres(question, candidate_code, ai_response)[0]
        # confidence_list = output['confidences']
        # labels = [elem['label'] for elem in confidence_list]
        time.sleep(2)
        return render_template("result.html", score="85")
    else:
        return render_template("index.html")


def predict_genres(question, candidate_code, ai_response):
    response = requests.post("https://zzarif-ai-detector.hf.space/run/predict", json={
        "data": [
            question,
            candidate_code,
            ai_response
        ]
    }).json()
    data = response["data"]
    return data
