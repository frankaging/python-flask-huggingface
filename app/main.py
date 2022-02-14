import datetime
from flask import Flask, jsonify, request, render_template
# import objects from the Flask model
# from keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)


@app.route('/')
def root():
    # For the sake of example, use static information to inflate the template.
    # This will be replaced with real information in later steps.
    dummy_times = [datetime.datetime(2018, 1, 1, 10, 0, 0),
                   datetime.datetime(2018, 1, 2, 10, 30, 0),
                   datetime.datetime(2018, 1, 3, 11, 0, 0),
                   ]

    return render_template('index.html', times=dummy_times)

@app.route('/tweet', methods=['POST'])
def createTweet():
    content = request.json['content']
    hasil = pipe(content)
    tweet = {'content': content,'date': request.json['date'], 'id':    request.json['id'], 'sentiment': hasil[0]['label']}
    return jsonify({'data': tweet})


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='0.0.0.0', port=5000, debug=True)
