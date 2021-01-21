---
layout: post
title:  "[논문 요약]Enriching word Vectors with Subword Information - fasttext"
date:   2020-09-21
excerpt: "fasttext 논문을 읽고 요약 및 코드 구현"
NLP_review: true
tag:
- NLP
comments: false
---

## 0\. 들어가며

fasttext 에 관련된 논문은 [여기](https://arxiv.org/abs/1607.04606) 에서 확인할 수 있습니다.

## 1\. Abstract

본 논문에서는 각 단어가 **bag of character n-grams** 으로 표현되는 skip-gram 모델을 바탕으로 새로운 접근법을 제안합니다. <br>

기존 모델은 단어 형태로 vector를 나타내며, 형태학적 특징을 고려하지 못합니다.


> 반면, fasttext는 개별 단어가 아닌 n-gram의 Charaters를 Embedding
각 단어는 Embedding된 n-gram의 합으로 표현됩니다!

**그 결과 : 빠르고 좋은 성능**

## 2\. Introduction

Introduce an extension of the continuous skipgram model which takes into account subword information<br>

- 개별 단어를 기반하는 Skip-gram의 메커니즘을 확장시켜 subword information (n-gram characters)을 도입해 Fasttext model 제시
- 기존 모델(word2vec)과 달리 언어의 형태학적(Morpological) 특징 파악 가능

## 3\.General model

#### Skip gram

w1, ..., wT  단어의 시퀀스가 큰 훈련 말뭉치 로 주어졌을 때,<br>
skipgram model의 목적 : 로그 우도함수를 최대화시키는 것<br>
![capture1](https://user-images.githubusercontent.com/28617444/94133653-e692fc80-fe9b-11ea-8d49-29fe88fda09f.PNG)<br>

wt가 주어졌을 떄, wc(context words)가 관찰될 확률은 매개변수로 지
scoring function: (word, context) 쌍을 점수로 map시키는 문맥의 가능성을 softmax로 정의<br>![capture2](https://user-images.githubusercontent.com/28617444/94133809-1b9f4f00-fe9c-11ea-90b8-98aa032551b7.PNG)


**그러나, wt 단어가 주어졌을 떄, 문맥 단어 only one context word(wc)를 예측하는 것이 문제**<br>

#### Subword model
 : character n-grams을 이용하는 이 모델에 대한 설명

 1. 기존 Skipgram model
  : 각 단어에 대해 구별되는 벡터 표현을 사용<br>
      -->단어의 내부 구조를 무시(한계)
      - 이 정보를 고려하기 위해, 다른 scoring funtion s를 제안
      <br>![image](https://user-images.githubusercontent.com/28617444/94134637-3de59c80-fe9d-11ea-884e-5d7d6959df2f.png)

 2. 각 단어 w는 bag of character n-gram 으로 표현
   <br>즉, 단어의 n-gram을 추출하여 각각의 vector들의 합으로 표현

 3. 다룬 문자 sequence와 구별하게 해주는 단어의 시작과 끝에 특수 기호 <(접두사)and >(접미사)를 추가
 4. n그램 집합에 w라는 그 단어 자체도 포함
 5. OOV (out of vocabulary) 해결

 특정 단어가 vocabulary에 존재하지 않더라도,
 <br> **character n-gram으로 새로운 단어에 대한 vector를 예측 가능**

Example
> where 및 n = 3이라는 단어를 예로 들면, <br>
character n-grams : <wh, wh, her, ere, re> <br>
special sequence : < where >


## 4\. 실습

<br> fasttext 관련 코드 [github주소](https://github.com/facebookresearch/fastText)입니다.<br>

먼저, fasttext 라이브러리를 설치 위한 리눅스 형식의 설치법입니다.

```python
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
$ sudo python setup.py install
```
위의 코드는 [이 사이트](https://fasttext.cc/docs/en/support.html) 에서 구체적으로 확인할 수 있습니다.

그러나 위의 깃허브 주소에서 권장하는 fasttext 라이브러리 설치 방법은 윈도우에서 사용하기는 까다로워 보입니다.
따라서,
[다음 사이트](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext)에 들어가 파이썬 버전에 맞는 whl파일을 다운받습니다.

    > Example)
    python3.6/64bit : fasttext-0.9.1-cp36-cp36m-win_amd64.whl
    python3.7/64bit : fasttext-0.9.1-cp37-cp37m-win_amd64.whl
    python3.8/64bit : fasttext-0.9.1-cp38-cp38-win_amd64.whl

  > 저는 파이썬 3.7 버전과 윈도우 64비트를 사용중이라 "fasttext‑0.9.2‑cp37‑cp37m‑win_amd64.whl" 이 파일을 다운받았습니다!

디렉토리에 다운 받은 파일을 넣고, cmd 창에 "**pip install fasttext‑0.9.2‑cp37‑cp37m‑win_amd64.whl**" 를 입력하시면 fasttext 라이브러리가 설치됩니다.

확인을 위해, 쥬피터 노트북이나 cmd 창에서 python을 열고 import fasttext를 하시면 에러 없는 것을 확인할 수 있습니다!


## 1\. fasttext 파라미터

fasttext에서 제공하는 [코드](https://fasttext.cc/docs/en/support.html) 를 실습해 보았습니다.

1. train_unsupervised('data.txt') <br>
파라미터로 모델 지정 가능 ('skipgram', 'cbow')

#### train_unsupervised parameters
```
input             # training file path (required)
model             # unsupervised fasttext model {cbow, skipgram} [skipgram]
lr                # learning rate [0.05]
dim               # size of word vectors [100]
ws                # size of the context window [5]
epoch             # number of epochs [5]
minCount          # minimal number of word occurences [5]
minn              # min length of char ngram [3]
maxn              # max length of char ngram [6]
neg               # number of negatives sampled [5]
wordNgrams        # max length of word ngram [1]
loss              # loss function {ns, hs, softmax, ova} [ns]
bucket            # number of buckets [2000000]
thread            # number of threads [number of cpus]
lrUpdateRate      # change the rate of updates for the learning rate [100]
t                 # sampling threshold [0.0001]
verbose           # verbose [2]
```

  2. train_supervised('data.train.txt')

#### train_supervised parameters
```
input             # training file path (required)
lr                # learning rate [0.1]
dim               # size of word vectors [100]
ws                # size of the context window [5]
epoch             # number of epochs [5]
minCount          # minimal number of word occurences [1]
minCountLabel     # minimal number of label occurences [1]
minn              # min length of char ngram [0]
maxn              # max length of char ngram [0]
neg               # number of negatives sampled [5]
wordNgrams        # max length of word ngram [1]
loss              # loss function {ns, hs, softmax, ova} [softmax]
bucket            # number of buckets [2000000]
thread            # number of threads [number of cpus]
lrUpdateRate      # change the rate of updates for the learning rate [100]
t                 # sampling threshold [0.0001]
label             # label prefix ['__label__']
verbose           # verbose [2]
pretrainedVectors # pretrained word vectors (.vec file) for supervised learning []
```

Model object functions
```
get_dimension           # Get the dimension (size) of a lookup vector (hidden layer).
                        # This is equivalent to `dim` property.
get_input_vector        # Given an index, get the corresponding vector of the Input Matrix.
get_input_matrix        # Get a copy of the full input matrix of a Model.
get_labels              # Get the entire list of labels of the dictionary
                        # This is equivalent to `labels` property.
get_line                # Split a line of text into words and labels.
get_output_matrix       # Get a copy of the full output matrix of a Model.
get_sentence_vector     # Given a string, get a single vector represenation. This function
                        # assumes to be given a single line of text. We split words on
                        # whitespace (space, newline, tab, vertical tab) and the control
                        # characters carriage return, formfeed and the null character.
get_subword_id          # Given a subword, return the index (within input matrix) it hashes to.
get_subwords            # Given a word, get the subwords and their indicies.
get_word_id             # Given a word, get the word id within the dictionary.
get_word_vector         # Get the vector representation of word.
get_words               # Get the entire list of words of the dictionary
                        # This is equivalent to `words` property.
is_quantized            # whether the model has been quantized
predict                 # Given a string, get a list of labels and a list of corresponding probabilities.
quantize                # Quantize the model reducing the size of the model and it's memory footprint.
save_model              # Save the model to the given path
test                    # Evaluate supervised model using file given by path
test_label              # Return the precision and recall score for each label.
```

## 2\. fasttext 실습
```python
import fasttext
model = fasttext.train_unsupervised('review.sorted.uniq.refined.tsv.text.tok',model='skipgram', epoch=5,lr = 0.1)
print(model.words)   # list of words in dictionary
```
![image](https://user-images.githubusercontent.com/28617444/95245656-5e730680-084e-11eb-97bd-d35ad0b2c4fb.png)

```python
print(model['행사']) # get the vector of the word '행사'
```
단어 '행사' 에 대한 vector를 도출할 수 있습니다. <br><br>
![image](https://user-images.githubusercontent.com/28617444/95245774-8e220e80-084e-11eb-8b96-325a8abb6e01.png)

## Importance of character n-grams

서브단어 정보를 사용하면 모르는 단어의 벡터도 도출할 수 있습니다.
```python
model.get_word_vector("보아즈")
```
![image](https://user-images.githubusercontent.com/28617444/95246650-beb67800-084f-11eb-9fb0-e69c8f588db6.png)

이처럼 데이터에 없는 단어(vocabulary에 없는 단어)도 벡터 출력 가능합니다.<br>
이를 out of vocabulary (oov)라 합니다.

## Nearest neighbor queries

word vector의 퀄리티를 평가하며 벡터의 의미 정보 유형을 직관적으로 보여줍니다.

```python
model.get_nearest_neighbors('반짝반짝')
```
![image](https://user-images.githubusercontent.com/28617444/95246297-3afc8b80-084f-11eb-8261-c8651fa57df2.png)

## Measure of similarity
단어 사이의 유사성을 계산해 최근접 이웃을 찾을 수 있을 수 있습니다.
모든 단어들을 계산해서 가장 유사한 단어 10개를 나타내고 해당 단어가 있다면, 상단에 표시되고 유사도는 1입니다.

## Word analogies
다음과 같이 세 단어의 관계를 유추할 수 있습니다.

```python
model.get_analogies("저렴","싸","할인")
```
![image](https://user-images.githubusercontent.com/28617444/95246464-7dbe6380-084f-11eb-80ac-8b01a474512f.png)
