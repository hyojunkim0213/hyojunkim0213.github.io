---
layout: post
title:  "[논문 요약]Enriching word Vectors with Subword Information - fasttext"
date:   2020-09-21
excerpt: "fasttext 논문을 읽고 요약 및 정리하고자 한다."
NLP_review: true
tag:
- NLP
comments: false
---

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Frlagywns0213.github.io%2Fnlp%2FNLP-fasttext%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=%EC%A1%B0%ED%9A%8C%EC%88%98&edge_flat=false)](https://hits.seeyoufarm.com)
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
