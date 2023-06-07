# Transformer_English-to-French-Translation
딥러닝을 이용한 영어-프랑스어 기계 번역 모델 구현

<br>

<img width="800" alt="스크린샷 2023-06-07 19 36 00" src="https://github.com/monzheld/Transformer_English-to-French-Translation/assets/93971420/b4c3d184-5c83-4650-872f-3cf8788634f7">


<br>

## 제작 기간

- 제작 기간: 2022.03.17 ~ 2022.03.22
- 개선 기간: 2023.03.02 ~


<br>

## 디렉토리 구조

<br>

```bash
├── Section4
│   ├── en_fr_tokenizers       ﹒﹒﹒ 영어, 프랑스어 Tokenizer
│   └── Project_Section4.ipynb ﹒﹒﹒ 원본 코드
├── model
│   ├── transformer.py         ﹒﹒﹒ Transformer 모델
├── data_loader.py             ﹒﹒﹒ 데이터 전처리 
├── evaluate.py                ﹒﹒﹒ 테스트 데이터셋 성능 평가 
├── train.py                   ﹒﹒﹒ 모델 훈련 및 저장
├── translate.py               ﹒﹒﹒ 입력 받은 문장의 번역 결과 출력
├── translator.py              ﹒﹒﹒ 추론 모델 생성 및 저장 
``` 

<br>



## 개선 사항
- [x] 데이터 업데이트
- [x] ipynb에서 py 파일로 수정
- [x] 평가지표로 BLEU Score 사용 
- [x] Post-Layer Normalization 에서 Pre-Layer Normalization 방식으로 수정  


<br>



## 데이터셋
- French - English
- 데이터셋 크기: (208906, 2)
- [데이터 출처](http://www.manythings.org/anki/)


