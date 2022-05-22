import os
import json
import torch
from torch import nn
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from sentiment_analysis_koBERT import predict

app = Flask(__name__)


@app.route('/emotion', methods=['POST'])
def test():
    content = json.loads(request.get_data('content'))
    print(content['content'])
    data = predict(content['content'])
    return jsonify({
        'emotion': data
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=int(os.environ.get("PORT", 5000)))
