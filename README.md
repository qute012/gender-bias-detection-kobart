# gender-bias-detection-kobart
SKT-AI에서 공개된 Bart 모델을 이용하여 성 차별 감지 competition에 참가해보았습니다.

# Requires
[koco] 데이터 셋과 사전 학습된 [KoBart]의 설치가 필요합니다.
```
pip install koco
pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
```
Kobart를 설치할 때, Pytorch 1.7.1 버전이 자동적으로 설치됩니다. 자신의 cuda 환경에 맞추어서 재설치가 필요할 수 있습니다.

# Usage
아래의 명령어를 실행하면, 매 에포크마다 좋은 정확도를 가진 모델을 models/ 경로에 저장하도록 되어있습니다.
```
python KGBD_task.py
```

[koco]: https://github.com/inmoonlight/koco
[KoBart]: https://github.com/SKT-AI/KoBART

# Score
| Dataset | Score |
| ------- | ----- |
| Dev     | 0.62  |
