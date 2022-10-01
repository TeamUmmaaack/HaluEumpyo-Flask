from lib2to3.pytree import convert
import os
import json
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from APIService.convert_emotion import convert_emotion
from model.music_recommendation import recommend

app = Flask(__name__)


@app.route('/emotion', methods=['POST'])
def test():
    content = json.loads(request.get_data('content'))
    print(content['content'])
    datas = recommend(content['content'])
    emotions_id = convert_emotion(datas["emotion"])
    recommended_data = datas["recommended_musics"]
    music_id_list = []
    for data in recommended_data:
        music_id_list.append(int(data['id']))

    return jsonify({
        'emotion': emotions_id,
        'recommended_music': music_id_list[0],
        'similar_musics': music_id_list[1:]
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=int(os.environ.get("PORT", 5000)))
