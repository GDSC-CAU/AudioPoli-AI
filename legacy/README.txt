========== 실험 환경 & Requirements ==========

GPU : Geforce RTX3090 24GB
OS : Ubuntu 18.04.05 LTS

CUDA 11.1
cuDNN 8.0.4

Python 3.6.9
Tensorflow 1.15.4
keras 2.2.4
librosa 0.8.0
seaborn 0.7.1
numpy 1.17.0
pandas 0.25.3
matplotlib 3.3.4
sklearn 0.20.0

※아래 도커 환경에서 실행 권장(실험 환경과 동일, librosa 라이브러리만 추가 설치필요)
Docker 버전 : 20.10.1
1. 도커 이미지 다운
	$ docker pull nvcr.io/nvidia/tlt-streamanalytics:v3.0-dp-py3
2. 도커 컨테이너 실행
	$ docker run --gpus all --name test -d -it -v <AI모델 폴더 경로>:/workspace -p 8888:8888 nvcr.io/nvidia/tlt-streamanalytics:v3.0-dp-py3 jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
3. 도커 컨테이너 터미널 접속
	$ docker exec -it <생성된 도커 컨테이너 주소> /bin/bash
	=> 터미널안에서 아래 사용 방법으로 실행


========== 사용 방법 ==========

1. testset 폴더에 [위급상황 음향/음성 데이터셋]을 저장
2. hdf.py 실행
	=> $python hdf.py
	=> testset에 있는 데이터가 클래스별 HDF5 파일 형식으로 testset_hdf 폴더에 저장
	=> testset_hdf에 예시로 이미 11684개의 데이터로 만든 HDF5 형식 파일 존재(예시파일)
3. model_eval.py 실행
	=> $python model_eval.py
	=> save_result 폴더에 confusion matrix 그래프 결과 저장(이미 예시 그래프 존재)

※ 출력 결과값 직전에 에러메세지 "Internal: Invoking ptxas not supported on Windows
Relying on driver to perform ptx compilation. This message will be only logged once."가
나올경우 환경세팅 문제로 모델 결과값에 영향을 줄 수 있음.

========== 출력 결과 예시 ==========

Prediction response time :  7.046435594558716
Test accuracy: 0.9771482369051695

Category Classification Report

              precision    recall  f1-score   support

           A       0.99      0.99      0.99       728
           B       0.99      0.99      0.99       513
           C       0.98      0.98      0.98       455
           D       0.98      0.99      0.98      1023
           E       0.98      0.98      0.98      1148
           F       0.99      0.99      0.99       399
           G       0.98      0.99      0.98      1409
           H       0.98      0.99      0.99      1092
           I       1.00      1.00      1.00       661
           J       1.00      1.00      1.00       227
           K       0.87      0.96      0.91       723
           L       0.98      0.99      0.98       708
           M       0.97      0.80      0.88       561
           N       1.00      1.00      1.00       977
           O       0.99      0.99      0.99       760
           P       0.99      0.96      0.97       300

   micro avg       0.98      0.98      0.98     11684
   macro avg       0.98      0.97      0.98     11684
weighted avg       0.98      0.98      0.98     11684

Confusion Matrix graph saved ./save_result/