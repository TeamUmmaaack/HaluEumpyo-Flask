import gluonnlp as nlp
import numpy as np
import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils.utils import get_tokenizer
from torch.utils.data import Dataset

from Bert_model import BERTClassifier

kobert_model, vocab = get_pytorch_kobert_model()
model = BERTClassifier(kobert_model, dr_rate=0.5, num_classes=4)

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

emotion_clsf_weights_file = "data/model_state_dict.pt"
model.load_state_dict(torch.load(
    emotion_clsf_weights_file, map_location=device))
model.eval()

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return self.sentences[i] + (self.labels[i],)

    def __len__(self):
        return len(self.labels)


# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5


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
        another_test, batch_size=batch_size, num_workers=0)

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
            else:
                test_eval.append("안녕")
            # print(probability)
        print(">> 일기에서 " + test_eval[0] + " 느껴집니다.")
        return emotion_arr[0]
