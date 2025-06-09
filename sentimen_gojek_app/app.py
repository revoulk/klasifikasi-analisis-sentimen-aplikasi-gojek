from flask import Flask, render_template, request
import pandas as pd
from catboost import CatBoostClassifier

app = Flask(__name__)

# Load model
model = CatBoostClassifier()
model.load_model("model/model_gojek_sentiment.cbm")

@app.route('/', methods=['GET', 'POST'])
def index():
    hasil_prediksi = None
    ulasan = ''
    if request.method == 'POST':
        ulasan = request.form['ulasan']
        if ulasan.strip():
            df_input = pd.DataFrame({'content': [ulasan]})
            prediksi = model.predict(df_input)[0][0]
            hasil_prediksi = prediksi
    return render_template('index.html', ulasan=ulasan, hasil=hasil_prediksi)

if __name__ == '__main__':
    app.run(debug=True)
