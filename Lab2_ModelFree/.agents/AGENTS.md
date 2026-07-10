
---
환경정보
구글 클래스룸에 업로드된 HW1_1.json과 HW1_2.json 두 개의 파일은 15×15
맵입니다. 먼저 두 파일을 다운로드받아 Lab2_ModelFree/env/maps/ 경로에
옮겨주세요. 제공된 맵을 기반으로 SARSA와 Q-Learning 알고리즘을 수정·적용하여
제한된 에피소드 내에 높은 성공률을 달성하는 것이 목표입니다.

1. SARSA 알고리즘 (총점의 40%)
실습 시간에 사용한 SARSA 코드에서 다음 조건을 만족하도록 파라미터를
조정하거나 알고리즘을 수정합니다:
● 최대 에피소드 수 (max_episodes) : 500
에피소드당 최대 스텝 수 (max_steps) : 500

❖ 코드 실행 방법
- 학습:
python train.py --algo sarsa --map HW1_1.json
python train.py --algo sarsa --map HW1_2.json
- 테스트 (렌더링):
python render.py --policy policy_sarsa_None.pkl --map HW1_1.json
python render.py --policy policy_sarsa_None.pkl --map HW1_2.json

2. Q-Learning 알고리즘 (총점의 40%)
실습 시간에 사용한 Q-Learning 코드에서 다음 조건을 만족하도록 파라미터를
조정하거나 알고리즘을 수정합니다.
● 최대 에피소드 수 (max_episodes) : 500
● 에피소드당 최대 스텝 수 (max_steps) : 100
❖ 코드 실행 방법
- 학습:
python train.py --algo q_learning --map HW1_1.json
python train.py --algo q_learning --map HW1_2.json
- 테스트 (렌더링):
python render.py --policy policy_q_learning_None.pkl --map HW1_1.json
python render.py --policy policy_q_learning_None.pkl --map HW1_2.json

3. 보고서 (총점의 20%)
A4 기준 1-2페이지 이내의 보고서를 작성하며, 아래 내용을 포함합니다:
● SARSA 알고리즘에서 수정한 내용과 그 이유
● Q-Learning 알고리즘에서 수정한 내용과 그 이유
● 자체 실험 결과 분석 및 설명
● 실행 방법에 대한 설명

[채점 기준]
조교 컴퓨터에서 동일한 조건으로 각 알고리즘을 10회 실행하여 성능을
평가합니다.
● 5회 이상 성공: 기본 통과 (90점)
● 8회 이상 성공: 우수 통과 (100점)
● 5회 미만 성공: 수정한 알고리즘의 내용을 바탕으로 정성적으로 평가하여
점수 부여

[제출물]
● 수정된 알고리즘 코드 (SARSA 및 Q-Learning)
● 보고서 (PDF 형식)

---

(1) 채점 기준
조교 컴퓨터에서 랜덤 시드를 다양하게 설정해 학습 코드를 10번 돌린 후 각각 마지막에 학습된 policy를 가지고 목적을 달성하는지 검사합니다.
채점 기준에 대해서 많이 궁금해 하셔서 테스트용으로 기존 실습 코드에 Lab2_ModelFree/eval.py를 추가했습니다. 실습자료에 배포된 대로 실습자료를 업데이트하시거나, eval.py만 다운로드받으신 뒤 아래 명령어로 테스트가능합니다.
python eval.py --algo sarsa --map HW1_1.json

(2) max_steps
HW1 조건에 SARSA는 max_steps 500 / Q-learning은 max_steps 100으로 적혀있습니다.
python 실행 명령어의 argument에는 max_steps가 없으니 sarsa.py의 line 16과 q_learning.py의 line 16에 있는 max_steps를 적절히 조절해주시길 바랍니다.

(3) 코드 수정 영역
sarsa.py의 sarsa함수, q_learning.py의 q_learning함수를 수정하시면 됩니다. 그 외의 경우를 수정하실 일은 없어보이나 수정하실 경우 보고서에 언급해주시길 바랍니다.

(4) 제출 방법
Lab2_ModelFree 폴더와 보고서를 함께 업로드해주시면 됩니다.

---
 
공지된 코드 수정 영역에 대해서 수정할 경우 자유도가 너무 높다는 문의가 있어서, 추가 안내를 드리려고 합니다.
실습 시간에 같이 살펴봤던 코드 부분은 SARSA, Q-Learning 알고리즘의 본질이라 수정할 부분이 따로 없습니다.

과제를 통해서 다양하게 테스트해볼 수 있는 부분들은 실습자료에 적혀있는 학습 관련 설정 변수(gamma, alpha 등)과 SARSA, Q-Learning 함수 초기에 설정하는 변수들(Initial epsilon, min_epsilon, decay_rate) 등을 다양하게 바꿔가면서 보는 결과들입니다. 이 변수들을 조정하면서 주어진 태스크를 성공하는게 강화학습 튜닝의 과정이라고 이해해주시면 되겠습니다.

그리고 q_learning.py에서 현재 max_steps가 500인데 line 16참고하셔서 100으로 바꿔서 결과 보셔야하는점 리마인드드립니다.

