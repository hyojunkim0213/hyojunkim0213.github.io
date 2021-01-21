---
layout: post
title:  "Pretraining Bert Model"
date:   2020-12-21
excerpt: "pretrained-Bert 모델 사용이 아닌 직접 사전학습 프로세스를 정리하고자 한다."
NLP_review: true
tag:
- NLP
comments: false
---

## 0\. 개요

Bert 모델의 장점은 pretrained model을 직접 불러와서 fine-tuning 함으로써 짧은 시간 안에, bert model의 complexitiy를 사용할 수 있는 것이다.<br>
그러나, pretrained model (Bert-Multilingual model, kobert 등) 의 경우 다양한 분야에서 bert 모델을 사용하기에는 다음과 같은 한계가 존재한다.
- 구체적인 용어의 부재

예를 들어, 의료 분야에서의 의학적인 용어는 bert 모델이 인식할 수 없다. <br>

즉, **데이터 분야별로 특화된 pretrained-model** 을 구축한다면 추후 연구 활용에 매우 효과적일 것으로 사료된다. <br>
따라서 bert model 을 사전시키는 process를 정리하고자 한다.

## 1\. vocab 형성

- Sentencepiece 라이브러리 사용

구글의 Sentencepiece 라이브러리를 통해 단어장을 구축한다.
버트 model의 특징 중 하나인 **word piece**
```python
import sentencepiece as spm

corpus = "harrypotter.txt"
prefix = "harry_word"
vocab_size = 4000
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +
    " --model_type=word"
    " --max_sentence_length=999999" + # 문장 최대 길이
    " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" + # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]") # 사용자 정의 토큰

```

나는 먼저 Selenium을 통하여 네이버 지도에서 "카페명", "주소", "위도","경도"를 크롤링한 후 이를 지도 시각화하기로 구상하였다.

그러나, 네이버 지도 api를 사용하는 것이 아니라 단순히 네이버 지도 사이트에서 크롤링을 하기에 "위도", "경도"에 대한 데이터를 수집할 수가 없었다.
