import os
import json
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from APIService.convert_emotion import convert_emotion
from model.sentiment_analysis_koBERT import recommend

app = Flask(__name__)


@app.route('/emotion', methods=['POST'])
def test():
    content = json.loads(request.get_data('content'))
    print(content['content'])
    data = recommend(content['content'])
    return_data = data.iloc[0, 0:5]
    music_id = int(return_data['id'])
    emotion_id = int(return_data['감정'])
    converted_emotion_id = convert_emotion(emotion_id)
    return jsonify({
        'emotion': converted_emotion_id,
        'musicId': music_id
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=int(os.environ.get("PORT", 5000)))
