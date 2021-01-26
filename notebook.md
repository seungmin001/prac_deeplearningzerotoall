# 공부한 것들 정리  
<br/>

## 궁금증과 해답
---
- data를 입력할 때 np.array([],dtype=np.float32)에서 dtype 명시 지정해주는 이유?  
    * 기본적으로 np.array()에서 생성할 때  지정해주지 않으면 (그 자료형 중에 가장 minimum한 것으로)float16으로 저장된다.
    * tf.random.normal이 dtype=tf.dtypes.float32 이기 때문에 나중에 tf.matmul() 등의 연산 시 오류가 발생할 수 있다.
  
- 왜 tf.Variable()에서 name을 부여하는가?  
    * 파이썬 코드의 변수 이름은 프로그램이 종료되면 사라지지만, 지정해놓은 tensor의 name은 파일에 저장된다.
    * https://stackoverflow.com/questions/33648167/why-do-we-name-variables-in-tensorflow/46419671  
    
- dataset iterator  
   * tf.data.Iterator is the primary mechanism for enumerating elements of a tf.data.Dataset.
   * It supports the Python Iterator protocol, which means it can be iterated over using a for-loop: **for element in dataset**
   * or by fetching individual elements explicitly via get_next(): **iterator = iter(dataset) .. iterator.get_next()**  
   
- dataset 왜 batch를 전체범위로 설정해두고 for loop로 iterate하면서 학습해야하나? 그냥 전체 data만 가지고 학습하면 안되나?  
   - dataset은 tensorflow API로서 입력 파이프라인을 제공하는데, data의 크기가 클 수록 연산속도 향상에 많은 도움이 되어 사용하는 것이 좋다.
   -  tf.data.Dataset.from_tensor_slices(),  tf.data.Dataset.from_tensors() 차이? : https://stackoverflow.com/questions/49579684/what-is-the-difference-between-dataset-from-tensors-and-dataset-from-tensor-slic
   

- maxtrix multiplication vs element-wise multiply  
   - tf.matmul vs tf.multiply(or *)
   - matmul은 행렬곱의 규칙을 지키는 행렬끼리
   - dot연산은 shape가 같은 것들이 각 요소끼리 곱
   
- softmax : cross-entropy cost function에서 왜 one-hot encoding을 하지 않고 softmax 값을 갖고 하는가?  
   - softmax는 확률으로 어느 정도 모델에 부합하는지 측정할 수 있기 때문에 발전시키기 좋다.
   - one-hot은 1,0,0 식으로 극으로 나눈 결과여서 softmax에 비해 정밀한 cost 값을 구하지 못한다.  
   
- softmax : cross-entropy cost function에서 reduce_sum( input_tensor, axis=1)을 하는 이유?  
   - y * log(s(h)) 결과 shape = (8,3). y와의 dot연산결과에 따라 각 행마다 하나의 열만 0이 아닌 값을 가진다.  
   원래의 cost 공식에 의하면 이 상태에서  reduce_mean을 해서 기존 y와의 얼마만큼 차이인지를 나타내는 cost 값을 구해야한다.
   - 근데 (8,3)에서 각 row당 2개의 col 값은 0이다.  
   이 상태에서 reduce_mean을 할 경우 항 개수인 24로 나누게 되는데, 0인 값은 y값과의 비교에 이유가 없는 값이므로(y값에서도 0이니까) 이들은 제외해야하고 data개수인 8로 나누어야한다.
   - 따라서 열 개수를  1으로 줄인다음 reduce_mean을 해야하기 때문에 reduce_sum axis=1을 해서 col을 하나로 만든 다음에, 평균값을 낸다.
   - sum이므로 0인 열들은 영향을 주지 않게된다.
   
- cost 값의 변화를 관찰할 때, print문에 function을 쓰는 경우 cost_func()을 호출해서 값을 알아본다. 혹은 cost값 저장 변수를 따로 선언해 저장해 놓는다. -> lab6에서 cost.numpy()를 print해서 값이 변하지 않는 상황이 발생했는데 이를 알아채지 못해서 많은 시간을 소비함.  



<br/>

## 함수 정리(tensorflow)
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
      
- (tf variable)a.assign_sub(b)  
    - python에서 a-=b 와 같은 연산.  
      
- variable.numpy()
    - tensor는 .numpy()를 호출하여 ndarray로 변환가능
    
- tf.random.normal([1], -100., 100.)
    - Outputs random values from a normal distribution.
    - [1] : shape, -100 : mean평균, 100 : stddev표준편차

- tf.data.dataset.from_tensor_slices()
    - Dataset : tf에서 입력 파이프라인을 만들 수 있는 built-in-API
    - 사용하려는 데이터로부터 Dataset 인스턴스를 만든다.
    - tf.data.Dataset.from_tensor_slices((features,labels)) 한 개 이상의 numpy배열을 넣는 것도 가능하다.
    - Creates a Dataset whose elements are slices of the given tensors.  
The given tensors are sliced along their first dimension. This operation preserves the structure of the input tensors, removing the first dimension of each tensor and using it as the dataset dimension. All input tensors must have the same size in their first dimensions.

- dataset.batch(batch_size) : batch - 한 번에 학습되는 data 개수. data를 batch_size 개수만큼씩 묶어서 outer dimension을 생성하고 반환환다.
- dataset.shuffle(buffer_size) : 처음부터 buffer_size 개수까지 element를 섞는다.
- dataset.repeat(count) : data를 count만큼 반복하여 늘린다. count가 3이면 전체 dataset에 같은 data가 3번 반복되어 저장됨.

    
- dataset.element_spec  
    - The type specification of an element of this dataset.
    - dataset의 원소의 타입 상세정보
    
- optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)  
   - Gradient descent (with momentum) optimizer.
   - w = w - learning_rate * g (momentum == 0) 수행
   - opt.minimize(loss(callable인..), update_var_list) <br/> == GradientTape작업 + opt.apply_gradients(grads_and_vars=zip(processed_grads, update_var_list))  
   
- tf.reduce_sum(값, axis=1)
   - axis를 기준으로 합을 구하는 함수.
   - axis=1 : column 기준 합치기. (8,3) 에서 axis=1 적용하여 sum하면 col이 하나로 합쳐져서 (8,1) 됨.
   - axis 기준에 관한 포스트 : http://taewan.kim/post/numpy_sum_axis/  
   
- tf.nn.softmax(   logits, axis=None, name=None  )  
   - softmax 값 반환

<br/>

## 함수 정리(numpy)

- Numpy array(ndarray)의 여러가지 size함수  

  - Number of dimensions of numpy.ndarray: ndim
  - Shape of numpy.ndarray: shape
  - Size of numpy.ndarray (total number of elements): size
  - Size of the first dimension of numpy.ndarray: len()
  - https://note.nkmk.me/en/python-numpy-ndarray-ndim-shape-size/
  
<br/>
