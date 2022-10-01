def convert_emotion(emotion_id):
    if emotion_id == 0:  # 행복
        return 1
    elif emotion_id == 1:  # 슬픔
        return 2
    elif emotion_id == 2:  # 분노
        return 4
    elif emotion_id == 3:  # 중립
        return 7
    elif emotion_id == 4:  # 혐오
        return 5
    elif emotion_id == 5:  # 놀람
        return 3
    else:  # 공포
        return 6
