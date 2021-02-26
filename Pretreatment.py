# -*- coding: utf-8 -*-
from pygrok import Grok
import konlpy
from konlpy import jvm
import pprint
from konlpy.tag import Okt
import re
import random

okt = Okt()
third = ""
textArr = []
textArr2 = []
list_positive = []
list_negative = []
list_neutral = []
Learning_data = []
Verified_data = []
ALL = 0
date_pattern = '%{YEAR:년도}년 %{MONTHNUM:월}월 %{MONTHDAY:일}일'
grok = Grok(date_pattern)


def strip_e(st):
    RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    return RE_EMOJI.sub(r'', st)


def DisposeLine(_lines, arr):
    global third
    for line in _lines:
        if (line != '\n'):  # 해당 줄에 글씨가 있는 경우
            if (grok.match(line)):  # date_pattern과 동일하면
                tmplist = []
                first = line.split(',')  # first에 날/나머지로 분리
                if (len(first) >= 2):  # 분리가 2분할 이상인경우는 한명의 대화로 인식
                    tmplist.append(first[0])  # 날짜들어가는곳
                    second = ""
                    for k in range(1, len(first)):  # 세컨드에 나머지 한줄로넣기
                        second += first[k]
                    secondlist = second.split(':')  # 세컨드를 이름/대화내용으로 분리
                    if (len(secondlist) >= 2):
                        tmplist.append(secondlist[0])  # 이름들어가는곳
                        third = ""
                        for j in range(1, len(secondlist)):
                            third += secondlist[j]
                            third = deleteNewline(third)
                        tmplist.append(third)  # 대화내용들어가는곳
                    arr.append(tmplist)
            else:
                tmpline_ = deleteNewline(line)
                third += tmpline_
                arr[-1][2] = third


def deleteNewline(str_):
    tmpline = ""
    splitlist = str_.split('\n')
    for q in range(0, len(splitlist)):
        tmpline += splitlist[q]
    return tmpline


# print(textArr)
# 여기까지 데이터 전처리
# 여기서부터 자연어 전처리'
def hanjiyoon(arr):
    current_who = arr[0][1]
    current_msg = ""
    for now_date, now_who, now_msg in arr:
        if current_who == now_who:
            current_msg += now_msg
        else:
            tmpList = []
            for i in okt.pos(current_msg):
                if i[1] in ['Noun', 'Verb', 'Adjective', 'Unknown', 'KoreanParticle']:
                    tmpList.append(i[0])  # 분석대상 단어만 집어넣음
            result_pos = naive_bayes_classifier(tmpList, list_positive, ALL)
            result_neg = naive_bayes_classifier(tmpList, list_negative, ALL)
            if (result_pos > result_neg):
                Learning_data.append([current_msg, 1])
            elif (result_neg > result_pos):
                Learning_data.append([current_msg, -1])
            else:
                Learning_data.append([current_msg, 0])
            current_who = now_who
            current_msg = now_msg


def getting_list(filename, listname):
    while 1:
        line = filename.readline()
        line_parse = okt.pos(line)
        for i in line_parse:
            if i[1] in ['Noun', 'Verb', 'Adjective', 'Unknown', 'KoreanParticle']:
                listname.append(i[0])
        if not line:
            break
    return listname


def naive_bayes_classifier(test, train, all_count):
    counter = 0
    list_count = []
    for i in test:
        for j in range(len(train)):
            if i == train[j]:
                counter = counter + 1
        list_count.append(counter)
        counter = 0
    list_naive = []
    for i in range(len(list_count)):
        list_naive.append((list_count[i] + 1) / float(len(train) + all_count))
    result = 1
    for i in range(len(list_naive)):
        result *= float(round(list_naive[i], 6))
    return float(result) * float(1.0 / 3.0)


if __name__ == '__main__':
    f_pos = open('positive.txt', 'r', encoding='UTF8')
    f_neg = open('negative.txt', 'r', encoding='UTF8')
    f_neu = open('neutral.txt', 'r', encoding='UTF8')
    list_positive = getting_list(f_pos, list_positive)
    list_negative = getting_list(f_neg, list_negative)
    list_neutral = getting_list(f_neu, list_neutral)
    f_pos.close()
    f_neg.close()
    f_neu.close()
    ALL = len(set(list_positive)) + len(set(list_negative)) + len(set(list_neutral))
    rText = open('testmodel.txt', mode='rt', encoding='utf-8-sig')
    lines = rText.readlines()
    rText.close()
    DisposeLine(lines, textArr)
    hanjiyoon(textArr)
    Ldata = open('test_data.txt', 'w', encoding='utf-8-sig')
    for data in Learning_data:
        randomtemp=random.randint(0,100)
        if(randomtemp==950):
            n=int(input(data[0]+"\n은 긍정적입니까? (1=긍정/0=중립/-1=부정) :"))
            Verified_data.append([data[0],n])
            continue
        tmpstr = strip_e(data[0]) + "\t" + str(data[1])
        Ldata.write(tmpstr.strip())
        Ldata.write('\n')
    Ldata.close()





