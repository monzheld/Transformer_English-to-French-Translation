# Transformer_English-to-French-Translation
딥러닝을 이용한 영어-프랑스어 기계 번역 모델 

<br>

## 제작 기간
2022.03.17 ~ 2022.03.22

<br>

## 프로젝트 기획 배경 
프랑스어를 전공하며 사람들이 언어 장벽 없이 자유롭게 소통할 수 있을 만큼의 정확한 번역 모델을 만드는 데 기여하고 싶다는 목표를 가졌고, <br>
이를 위해 딥러닝을 이용한 기계 번역 모델을 직접 구현해 보기 위해 기획

<br>

## 가설 검정 

- 10만 개의 데이터로 학습된 번역 모델은 문법적 특성을 지키지 못한다. -> **False**


<br>

## 사용 기술

#### EDA
- pandas
- numpy

#### 딥러닝 모델링 
- text.BertTokenizer
- Transformer

<br>

## 제작 과정

#### 텍스트 전처리 
- tensorflow의 `text.BertTokenizer`를 사용
- 데이터셋에서 직접 어휘를 생성해 언어별로 CustomTokenizer 생성

#### 딥러닝 모델 생성 
- `Transformer` 사용
- positional encoding / padding mask / look ahead mask / scaled dot product attention / Multi-Head (Self) Attention / FFNN (Feed Forward Neural Network) / Encoder Layer / Decoder Layer / Encoder / Decoder
- 하이퍼파라미터
   - num_layers = 6, d_model = 256, dff = 2048, num_heads = 8, dropout_rate = 0.1


#### 모델 비교

- EPOCHS = 30
   - Loss: 0.6392
   - Accuracy: 0.8205
   
- EPOCHS = 100
   - Loss: 0.2563
   - Accuracy: 0.9213


#### 모델 평가
- EPOCHS = 30 모델과 EPOCHS = 100 모델별로 Train 데이터셋에서 추출한 5개 문장과 Test 데이터셋에서 추출한 5개 문장의 번역 결과 비교 


<br>

## 결론

### 딥러닝 모델
- Transformer

### 평가지표 
- Loss
- Accuracy

### 모델 비교 

- EPOCHS = 30
   - Loss: 0.6392
   - Accuracy: 0.8205
   
- EPOCHS = 100
   - Loss: 0.2563
   - Accuracy: 0.9213

=> 모델을 30 epochs 와 100 epochs로 학습시킨 결과,
Accuracy는 10% 정도 차이 나지만 **Loss**는 **100 epochs 만큼 학습시킨 모델이 38% 적은 것**으로 나타남.
번역 결과를 봤을 때도 **100 epochs 만큼 학습 시킨 모델이 실제 번역과 조금 더 유사하게 번역**했지만
프랑스어 사전에 없는 단어들을 만들어내기도 함.

### 한계점 및 추후 해결 방안

- 번역 결과 출력 
   - 프랑스어의 악센트(accent), 대문자 표현 불가
   
- 모델 성능 평가지표 
   - 추후 기계 번역 결과와 사람이 직접 번역한 결과가 얼마나 유사한지 비교하는 대표적인 기계 번역 성능 평가 방식인
**BLEU Score**를 사용해 성능을 다시 측정해 볼 것

- 일방향 번역
   - 추후 영어 <-> 프랑스어 양방향 번역뿐만 아니라
**제로샷 학습**(Zero-shot learning), **mT5 모델**을 이용해 한국어 <-> 프랑스어 번역이 가능한 다국어 번역 모델로 발전시킬 것
