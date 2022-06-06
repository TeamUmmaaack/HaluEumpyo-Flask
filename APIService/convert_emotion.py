def convert_emotion(emotion_id):
    if emotion_id == 0:  # 행복
        return 1
    elif emotion_id == 1:  # 슬픔
        return 2
    elif emotion_id == 2:  # 분노
        return 4
    elif emotion_id == 3:  # 휴식
        return 7
