from konlpy.tag import Mecab
from scipy.sparse.dia import dia_matrix
import re
import scipy.sparse as sp
import pickle
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import gluonnlp as nlp
import numpy as np
import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils.utils import get_tokenizer
from torch.utils.data import Dataset

from model.classifier import BERTClassifier

kobert_model, vocab = get_pytorch_kobert_model()
model = BERTClassifier(kobert_model, dr_rate=0.5, num_classes=4)

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

emotion_clsf_weights_file = "checkpoint/model_state_dict.pt"
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
num_epochs = 5
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
    test = BERTDataset([data], 0, 1, tok, max_len, True, False)
    dataloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, num_workers=0)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        for logits in out:
            logits = logits.detach().cpu().numpy()
            logits = np.round(new_softmax(logits), 3).tolist()
            probability = []
            for logit in logits:
                probability.append(np.round(logit, 3))

            emotion = np.argmax(logits)
            probability.append(emotion)

    return probability


pd.set_option('display.max_rows', 320)
pd.set_option('display.max_colwidth', None)

root_path = 'data'
txt_file = root_path + "/korean_stopword_list_100.txt"
music_file = root_path + "/melon_crawling_by_sentiment_converted.csv"

melon = pd.read_csv(
    music_file, encoding='utf-8')


m = Mecab()

stop_words = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를',
              '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게', '좀', '잘', '걍', '으로', '자', '하다']

with open(txt_file, encoding='utf-8') as f:
    for i in f:
        stop_words.append(i.strip())
stop_words = set(stop_words)


# 일반명사, 고유명사, 동사, 형용사
def tokenize(raw, pos=["NNG", "NNP", "VV", "VA"], stopword=stop_words):
    return [word for word, tag in m.pos(raw) if len(word) > 1 and tag in pos and word not in stopword]


tf = TfidfVectorizer(
    tokenizer=tokenize, ngram_range=(1, 2), max_features=200000, sublinear_tf=True)

mtx_path = 'model.mtx'
tf_path = 'tf.pickle'


def learn_lyrics(tf):
    X = tf.fit_transform(melon['가사'])
    tf.n_docs = len(melon['가사'])
    with open(mtx_path, "wb") as fw:
        pickle.dump(X, fw)
    with open(tf_path, "wb") as fw:
        pickle.dump(tf, fw)


def recommend(more):
    with open(mtx_path, "rb") as fr:
        X = pickle.load(fr)
    with open(tf_path, "rb") as fr:
        tf = pickle.load(fr)

    example_vector = tf.transform([more])
    cos_similar = linear_kernel(example_vector, X).flatten()
    cos_similar = list(enumerate(cos_similar))
    sim_rank_idx = sorted(cos_similar, key=lambda x: x[1], reverse=True)
    sim_rank_idx = sim_rank_idx[:30]

    music_indices = [idx[0] for idx in sim_rank_idx]
    result = melon.loc[music_indices]
    result['점수'] = [idx[1] for idx in sim_rank_idx]
    emotion = predict(more)[-1]
    if emotion in [0, 1, 2, 3]:
        result = result[result['감정'] == emotion]

    return [result.iloc[0], result.iloc[1], result.iloc[2]]


learn_lyrics(tf)
