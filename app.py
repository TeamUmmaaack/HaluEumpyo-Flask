import os
import json
import torch
from torch import nn
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import sys
from os import path

app = Flask(__name__)

@app.route('/test', methods=['POST'])
def test():
    content = json.loads(request.get_data('content'))
    print(content['content'])
    sys.path.append(path.join(path.dirname(__file__)))
    from sentiment_analysis_koBERT import predict
    data = predict(content['content'])
    return jsonify({
        'emotion_id': data
    })


if __name__ == '__main__':
    PORT = 50051

    app.run(host="0.0.0.0", debug=True, port=int(os.environ.get("PORT", 5000)))
