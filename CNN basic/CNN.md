# 목차

1. **Convolution 연산**
2. **Back Propagation**
3. **Cost Function**
4. **Optimization 용어 정리**
5. **Optimizer의 종류**
   - Gradient Descent
     - Stochastic gradient descent
     - Momentum
     - Nesterov accelerated gradient
     - Adagrad
     - Adadelta
     - RMSprop
     - Adam

# 1. Convolution 연산

- **정의**

  고정된 크기의 kernel을 입력 데이터 상에서 움직여가면서 선형모델과 합성함수가 적용되는 구조로 신호를 kernel을 이용해 국소적으로 증폭 또는 감소시켜 정보를 추출/필터링하는 작업이다. 

  - 장점

    데이터 입력에 따라 kernel의 크기가 변하지 않는다. 

![image](https://user-images.githubusercontent.com/71866756/144757713-86bded0b-92fc-4224-b5bd-12822504ed28.png)

- **출력 크기 계산**

  - 입력 크기 : (H,W) , 커널 크기 : (Kh, Kw), 출력 크기 : (Oh, Ow)

    ![image](https://user-images.githubusercontent.com/71866756/144757738-3cb0c1b2-71eb-40dd-8778-526cc9148cbc.png)

  - 채널이 여러개인 경우 커널의 채널 수 = 입력의 채널 수!!

    ![image](https://user-images.githubusercontent.com/71866756/144757755-e516f7a7-0f6e-439d-96e9-22d55ae4106a.png)

    - 출력은 커널개수만큼 나온다. 

- 용어 정리

  - Stride

    커널을 입력 데이터에 convolution연산할 시 몇 칸씩 뛰어넘을지를 의미

    ![image](https://user-images.githubusercontent.com/71866756/144757767-9ba785e7-1ee1-47ac-a616-d5390fb68b2d.png)

  - Padding

    출력 크기 계산 수식을 이용하면 출력의 크기는 입력보다 작아지게 된다. 따라서 이 경우는 Padding이 없는 경우이며, 만약 padding이 있는 경우에는 입력의 최외곽 일정 크기만큼을 일정한 수로 채우는 것을 의미한다. 

    <img src="../../../../AppData/Roaming/Typora/typora-user-images/image-20211206013834179.png" alt="image-20211206013834179" style="zoom:67%;" />

  - Example

    

![image](https://user-images.githubusercontent.com/71866756/144757798-477e4037-fb61-4bca-8953-0546454a4abf.png)

# 2. Back Propagation

# 3. Cost Function

- **Regression Task**

  - MSE( Mean Square Error )

  ![image](https://user-images.githubusercontent.com/71866756/144757811-f1036d9a-8ecf-4a12-8cd0-56c4fe9d7f4c.png)

- **Classification Task**

  - CE( Cross-Entropy )

  ![image](https://user-images.githubusercontent.com/71866756/144757823-b31b23b9-be35-4809-bd71-d7ea1e763768.png)

- **Probabilistic Task**

  - MLE( Maximum Likelihood Estimation)

  

  ![image](https://user-images.githubusercontent.com/71866756/144757835-eb608cb0-053a-4803-bcaa-2b6579126082.png)

  

# 4. Optimization

- **Parameter, Hyper parameter**

  - Parameter

    최적해에서 찾고 싶은 값으로 weight, bias 등이 있다. 

  - Hyper parameter

    사용자가 지정하는 값으로 learning rate 등이 있다. 

- **Generalization**

  trainig error와 test error사이의 차이를 generalization gap이라고 한다. Generalization Performance가 좋다라는 것은 학습한 결과가 test data에서도 잘 동작하는 것을 의미한다. 

  
![image](https://user-images.githubusercontent.com/71866756/144757915-aa963733-a769-4b35-9fe5-7a62a174534e.png)

- **Underfitting & Overfitting**

  - Underfitting

    학습결과가 학습 데이터의 경향을 잘 표현하지 못하는 것 

  - Overfitting

    학습결과가 학습 데이터에 너무 치중되어 나타난 것으로 test data로 모델을 돌렸을 시 결과가 좋지 못할 가능성이 높다. 

- **Cross-validation**

  해당 모델의 Generalization Performance를 높이기 위한 방법으로 하나의 데이터셋을 train data와 validation data, test data로 구분하는 것을 의미한다. 

- **Bias & Variance**

  ![image](https://user-images.githubusercontent.com/71866756/144757934-01852996-8f87-4edb-a66b-d0c1680bc3cf.png)

  - Bias와 Variance는 Tradeoff 관계이다. 

    cost function의 경우 bias, variance, noise를 포함하고 있기 때문에 bias가 낮은 경우 그에 따라 자연스럽게 variance는 높을 수 밖에 없다. 그 반대 역시 마찬가지이다. 

    ![image](https://user-images.githubusercontent.com/71866756/144757964-03569ce3-a75b-4335-8f19-028212382fb1.png)

- Bootstrapping

  학습 데이터가 주어졌을 때, 그 학습데이터를 여러개의 sub data로 나누어 여러 모델을 만드는 것을 의미한다. 

  - Bagging( bootstrapping aggregating )

    bootstrapping을 통해 학습한 여러 모델들의 output의 평균을 내는 것으로 보통 한개의 모델을 이용하는 것보다 이런 모델들의 output의 평균을 이용하는 것이 더 좋은 경우가 많다. 

  - boosting

    한 모델을 학습할 때, 결과가 제대로 나오지 않는 데이터셋에 대해서 또 다른 모델을 만들고, 여러개의 모델을 sequence로 연결하여 강한 모델을 만드는 방법

  ![image](https://user-images.githubusercontent.com/71866756/144757995-308a5a0c-0237-4bb0-8599-e0dae2912515.png)

  

# 5. Optimizer의 종류

- **Gradient Descent**

  하나의 샘플 데이터로 gradient를 업데이트하는 것

  ![image](https://user-images.githubusercontent.com/71866756/144758090-898246d5-30a7-4a8a-9397-a0c8059eba0a.png)

  - mini-batch gradient descent

    전체 데이터가 아닌, 그보다 작은 batch 사이즈로 gradient를 업데이트 하는 것

  - batch gradient descent

    데이터 전체를 이용하여 한번에 업데이트 하는 것

  - batch 사이즈에 따른 특징

    batch 사이즈가 매우 큰 경우, 오른쪽 그림처럼 sharp minimum이 된다. 즉, 같은 위치에서 training한 결과는 minimum이지만, test한 결과가 매우 높은 값을 나타낼 수 있다. 

    batch 사이즈가 작은 경우, 왼쪽 그림처럼 gradient차이가 크지 않기 때문에 training에서의 좋은 결과는 test에서도 좋은 결과로 나올 가능성이 높다. 

  ![image](https://user-images.githubusercontent.com/71866756/144758555-7c761519-85e1-4849-94f1-c6ff398f8e4e.png)

- **Momentum**

  베타값이 들어가서 현재 weight를 이전 batch size만큼 학습했을 때의 weight에서 특정값을 곱하여 뺀다. 즉, 과거의 gradient를 어느정도 유지하여 gradient의 변화 폭이 커도 어느정도 학습이 잘 된다는 장점이 있다. 

  ![image](https://user-images.githubusercontent.com/71866756/144758124-980e7e0f-9408-4316-91bf-c7f6e2b1cb2c.png)

- **Nesterov Accelerated Gradient**

  Momentum과 비슷하지만, lookahead로 한번 더 이동하여 그 위치에서 계산한 값을 현재 계산에 넣어주어 좀 더 빠르게 minimum을 찾을 수 있다.

  ![image](https://user-images.githubusercontent.com/71866756/144758145-20c7dfff-1933-4541-80ab-b080f64496d6.png)

- **Adagrad**

  값이 많이 변한 parameter들에 대해서는 적게 변화시키고, 값이 적게 변한 parameter들에 대해서는 많이 변화시키는 방법으로 이전 parameter들의 변화를 기록하고 있어야 하며, 이 값은 학습이 진행될 수록 축적이 되어 커지기 때문에 오랜 시간 학습을 진행할 경우 weight의 변화가 줄어들어 학습이 제대로 안될 가능성이 있다. 

  ![image](https://user-images.githubusercontent.com/71866756/144758151-bf70d5ab-fc1f-4a48-a92e-ef83dcf8469b.png)

- **Adadelta**

  Adagrad의 문제를 해결하기 위해서 window값을 지정해준다. 즉, Adagrad처럼 값이 축적되어 커지진 않고, 만약 parameter의 개수가 클 경우에 메모리 문제가 발생할 수 있으니 감마를 이용하여 어느정도 완화시켜준다. Adadelta의 경우 learning rate가 없다는 단점이 있다. 

  - EMA ( Exponential Moving Average )

    EMA는 이동평균법으로 평균과의 가장 큰 차이점은 시간이라는 개념이다. 평균은 같은 시간대에서 산출되는 것이 흔한 반면, 이동평균은 동일대상을 서로 다른 시점에서 구한다는 점이 차이점이다. moving average filter를 생각하면 이해가 빠르다. 

  ![image](https://user-images.githubusercontent.com/71866756/144758225-35daeabe-4e5c-4920-b7aa-540e8567b1bc.png)

- **RMSprop**

  Adadelta에 step size, 즉 learning rate을 추가한 방법이다. 

  ![image](https://user-images.githubusercontent.com/71866756/144758204-610fe652-0d4b-41b7-aa22-1877989adb27.png)

- **Adam**

  현재 가장 무난하게 사용되고 있는 방법으로 Momentum과 EMA( 이동 평균법 )을 합친 방법이다. 

  입실론은 0으로 나눠지는 것을 막기 위한 요소로 Adam의 가장 중요한 요소이다. 

  ![image](https://user-images.githubusercontent.com/71866756/144758181-b5cfac99-18e4-4da7-83bb-1fc6916fcc86.png)

  
