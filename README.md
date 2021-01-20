# prac_deeplearningzerotoall
Tensorflow ,Google Colab

# 함수 정리
- tf.Variable(3)
변수 생성 및 초기값 부여. 초기값 3
- tf.constant([3.0, 3.0])
상수 선언. 1x2 matrix
- tf.enable_eager_execution()
TensorFlow supports eager execution and graph execution. In eager execution, operations are evaluated immediately. In graph execution, a computational graph is constructed for later evaluation.
그래프를 생성하지 않고 함수를 바로 실행하는 명령형 프로그래밍 환경으로 진행하기. TF2.0에서는 기본적으로 True
- tf.GradientTape()
Record operations for automatic differentiation. with구문과 쓰인다. with tf.GradientTape() as tape:
- tape.gradient(y,x)
y 함수를 x에 대해 미분하고 x를 대입한 값을 반환한다. x가 list로 주어지면 튜플로 반환한다.
- a.assign_sub(b)
python에서 a-=b 와 같은 연산.

