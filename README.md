# prac_deeplearningzerotoall
-----------
Tensorflow ,Google Colab
  
<br/>

# 궁금증과 해답
---
- data를 입력할 때 np.array([],dtype=np.float32)에서 dtype 명시 지정해주는 이유?  
    * 기본적으로 np.array()에서 생성할 때  지정해주지 않으면 (그 자료형 중에 가장 minimum한 것으로)float16으로 저장된다.
    * tf.random.normal이 dtype=tf.dtypes.float32 이기 때문에 나중에 tf.matmul() 등의 연산 시 오류가 발생할 수 있다.
  
- 왜 tf.Variable()에서 name을 부여하는가?  
    * 파이썬 코드의 변수 이름은 프로그램이 종료되면 사라지지만, 지정해놓은 tensor의 name은 파일에 저장된다.
    * https://stackoverflow.com/questions/33648167/why-do-we-name-variables-in-tensorflow/46419671

<br/>

# 함수 정리
------------
- tf.Variable(3)  
    - 변수 생성 및 초기값 부여. 초기값 3
      
- tf.constant([3.0, 3.0])  
    - 상수 선언. 1x2 matrix  
      
- tf.enable_eager_execution()  
    - TensorFlow supports eager execution and graph execution. In eager execution, operations are evaluated immediately. In graph execution, a computational graph is constructed for later evaluation.  
    - 그래프를 생성하지 않고 함수를 바로 실행하는 명령형 프로그래밍 환경으로 진행하기. TF2.0에서는 기본적으로 True  
      
- tf.GradientTape()  
    - Record operations for automatic differentiation. with구문과 쓰인다. with tf.GradientTape() as tape:  
      
- tape.gradient(y,x)  
    - y 함수를 x에 대해 미분하고 x를 대입한 값을 반환한다. x가 list로 주어지면 튜플로 반환한다.
      
- a.assign_sub(b)  
    - python에서 a-=b 와 같은 연산.  
      
- variable.numpy()
    - tensor는 .numpy()를 호출하여 ndarray로 변환가능
    
- tf.random.normal([1], -100., 100.)
    - Outputs random values from a normal distribution.
    - [1] : shape, -100 : mean평균, 100 : stddev표준편차

<br/>
