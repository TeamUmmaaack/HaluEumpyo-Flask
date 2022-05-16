import numpy as np
import json
from flask import Flask, request, jsonify
from analysis_BERT_main import predict


app = Flask(__name__)


@app.route('/test', methods=['POST'])
def test():
    content = json.loads(request.get_data(), encoding='utf-8')
    print(content)
    data = predict(content)
    return jsonify({
        'emotion_id': data
    })


if __name__ == '__main__':
    PORT = 50051

    app.run(host="0.0.0.0", debug=True, port=PORT)
