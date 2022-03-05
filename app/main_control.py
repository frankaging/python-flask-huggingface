import datetime
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin

# import objects from the Flask model
# from keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

goemotions_tokenizer = AutoTokenizer.from_pretrained(
    "monologg/bert-base-cased-goemotions-original",
    cache_dir = "../../huggingface_cache/"
)
goemotions_model = AutoModelForSequenceClassification.from_pretrained(
    "monologg/bert-base-cased-goemotions-original",
    cache_dir = "../../huggingface_cache/"
)
goemotions_pipe = TextClassificationPipeline(model=goemotions_model, tokenizer=goemotions_tokenizer)


@app.route('/', methods=['GET'])
def root():
    # For the sake of example, use static information to inflate the template.
    # This will be replaced with real information in later steps.
    response = jsonify({'Hello World!': 'You Got In!'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/tweet', methods=['POST'])
def createTweet():
    print("requesting data: ", request.json)
    content = request.json['mse']
    hasil = goemotions_pipe(content)
    tweet = {'content': content, 'sentiment': hasil[0]['label'], 'details': hasil[0]}
    ret = jsonify({'data': tweet})
    ret.headers.add('Access-Control-Allow-Origin', '*')
    return ret


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')
