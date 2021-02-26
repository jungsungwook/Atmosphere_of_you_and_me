# -*- coding: utf-8 -*-
from konlpy.tag import Okt
import nltk
import json
import os
import numpy as np
import keras
from pprint import pprint
import test_1
okt = Okt()
def read_data(filename):
    with open(filename, 'r',encoding='utf-8-sig') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        # txt 파일의 헤더(id document label)는 제외하기
        data = data[0:]
    return data

train_data = read_data('Learning_data.txt')
test_data = read_data('test_data.txt')


def tokenize(doc):
    #print(doc,"->",okt.pos(doc,norm=True,stem=True))########################Test Debug.Log[1010]########################
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

if os.path.isfile('train_docs.json'):
    with open('train_docs.json','r',encoding="utf-8-sig") as f:
        train_docs = json.load(f)
    with open('test_docs.json','r',encoding="utf-8-sig") as f:
        test_docs = json.load(f)
else:
    train_docs = [(tokenize(row[0]), row[1]) for row in train_data]
    test_docs = [(tokenize(row[0]), row[1]) for row in test_data]
    # JSON 파일로 저장
    with open('train_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    with open('test_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")

tokens = [t for d in train_docs for t in d[0]]
text_tokens = nltk.Text(tokens, name='NMSC')
cnt=0
for i in text_tokens.vocab().most_common(100):
    tmp=i[0].split('/')
    if tmp[1] in ['Noun','Verb','Adjective','Unknown','KoreanParticle']:
        cnt=cnt+1
        print (cnt,"등으로 많이 나온 단어 : ",i[0],"/ 횟수 : ",i[1])
#pprint(text_tokens.vocab().most_common(100)) ################Test Debug_Log[0099]#################
selected_words = [f[0] for f in text_tokens.vocab().most_common(100)]

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]
x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')

y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')



#모델의 구성
model = keras.models.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(100,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
#모델의 학습 과정 설정하기
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
            loss=keras.losses.binary_crossentropy,
            metrics=[keras.metrics.binary_accuracy])
#모델을 학습시키기
model.fit(x_train, y_train, epochs=10, batch_size=50)
#모델을 평가하기
results = model.evaluate(x_test, y_test)
print("[결과값] : ", results)


def analyzeMsg(msg):
    token = tokenize(msg)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score > 0.5):
        print("[{}]는 {:.2f}% 확률로 긍정적 대화입니다.\n".format(msg, score * 100))
    else:
        print("[{}]는 {:.2f}% 확률로 부정적 대화입니다.\n".format(msg, (1 - score) * 100))

rText = open('testTalk.txt', mode='rt', encoding='utf-8-sig')
lines = rText.readlines()
rText.close()
textArr=[]
test_1.DisposeLine(lines,textArr)
for arr in textArr:
    analyzeMsg(arr[2])

