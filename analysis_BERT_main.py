# torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
import torch
import numpy as np
from Bert_model import BERTClassifier
from dataset import BERTDataset, tok

# kobert

# GPU 사용
device = torch.device("cuda:0")

# BERT 모델 불러오기 필수
bertmodel = get_pytorch_kobert_model()


# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

# 학습 모델 로드
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
model = torch.load('data/KoBERT_model.pt')
model.load_state_dict(torch.load('data/model_state_dict.pt'))


def new_softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = (exp_a / sum_exp_a) * 100
    return np.round(y, 3)


# 예측 모델 설정


def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(
        dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(
        another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

    test_eval = []
    emotion_arr = []
    for i in out:
        logits = i
        logits = logits.detach().cpu().numpy()
        min_v = min(logits)
        total = 0
        probability = []
        logits = np.round(new_softmax(logits), 3).tolist()
        for logit in logits:
            # print(logit)
            probability.append(np.round(logit, 3))

        if np.argmax(logits) == 0:
            test_eval.append("행복이")
            emotion = "행복"
            emotion_arr.append(1)

        elif np.argmax(logits) == 1:
            test_eval.append("슬픔이")
            emotion = "슬픔"
            emotion_arr.append(2)

        elif np.argmax(logits) == 2:
            test_eval.append("분노가")
            emotion = "분노"
            emotion_arr.append(4)

        elif np.argmax(logits) == 3:
            test_eval.append("특정한 감정이 안")
            emotion = "중립"
            emotion_arr.append(7)

        probability.append(emotion)
    # print(probability)

    print(">> 일기에서 " + test_eval[0] + " 느껴집니다.")
    return emotion_arr[0]
