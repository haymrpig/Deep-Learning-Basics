# 다양한 MODEL

- **목차**
  1. [VGGNet](#1-vggnet)
  2. [ResNet](#2-resnet)
  2. [AlexNet](#3-alexnet)
  2. [DenseNet](#4-densenet)
  2. SENet
  2. MobileNet
  2. SqueezeNet
  2. AutoML(NAS,NASNet)

# 1. VGGNet

- **목적**

   VGGNet은 large-scale image 인식에서의 정확도에 대한 depth의 영향을 파악하기 위해 만들어진 모델이다. AI 모델의 정확도를 향상시키기 위해서 고려되는 요소들로는 적절한 window size, stride, parameter setting 등 다양하다. 이와 더불어 model의 depth가 미치는 영향 또한 무시할 수 없다. VGGNet에서는 이러한 Convolution network의 architecture design의 중요성을 파악하여 여러 depth의 모델을 이용하여 성능을 측정하였다. 

- **용어 정리**

  - scale jittering : 이미지 크기를 고정된 크기로 조절하거나, (min,max)같은 특정 범위의 사이즈로 조절하여 CNN의 입력 사이즈에 맞게 무작위로 crop하는 방법

  - single-scale training : 이미지를 고정시킨 사이즈로 학습

  - multi-scale training : (min,max)값 사이로 이미지를 크기를 변경하면서 학습

  - multi-crop : data augmentation시 이미지를 랜덤하게 자르는 것을 의미 ( 연산량이 증가한다. )

    ex) 좌상단, 우상단, 좌하단, 우하단, 중앙 + 좌우반전 -> 이미지 10배로 늘림

  - dense evaluation : FC layer로 max pooling을 통해 정보를 추출하면 표현력이 떨어지므로, max pooling을 dense하게 적용하는 방식

    ex ) stride를 2->1로 바꾸는 등

  - receptive field : conv을 여러번 진행하면 이전 이미지에서의 더 많은 픽셀값의 정보를 담고 있게 된다. 

    ex) conv 3x3을 두번 진행시 마지막에 나오는 1pixel은 이전 이미지의 3x3 pixel의 정보를 담고 있고, 그 이전의 5x5pixel의 정보를 담고 있는 것이 된다. 

- **Architecture**

  - input image
    - input image size 224x224 RGB
  - Conv layer
    - 3X3의 아주 작은 convolution filter 사용( stride 1, padding 적용하여 깊게 쌓을 수 있음 )
    - 1x1의 convolution filter를 사용하기도 한다. ( nonlinear 특성 강조 )

  - Pooling layer

    - 5개의 max-pooling layer 사용 ( 2x2 window, stride 2 )

  - FC( fully-connected ) layer

    - 2개의 fully connected layer ( 4096 channel ), 1개의 fully connected layer ( 1000 channel )

  - 그 외

    - hidden layer의 activation function은 ReLU 사용

    - architecture의 다른 parameter들을 fix한 상태로 convolution layer를 추가하여 depth 증가

    - A모델 구성 후, A 모델에 학습된 layer들을 B,C,D,E에 가져다 씀으로 최적의 초기값으로 학습을 좀 더 용이하게 진행 

    - 224x224이미지로 rescale시, 비율을 유지하면서 rescale을 진행하는 방식인 isotropically-rescaling을 진행 ( 이미지 비율이 1:1이 아닌 경우, random crop )

    - dense, multi-crop evaluation 사용

      

  

- **종류**

  VGG11, VGG13, VGG16, VGG19 ( layer수에 따라 나뉨 )

  ![image](https://user-images.githubusercontent.com/71866756/147485941-829a239b-3950-48d3-bb7c-03f406cacaf2.png)


- **결과**

  - single scale image

  ![image](https://user-images.githubusercontent.com/71866756/147485983-fc610bc5-f361-403f-a1cc-f144ef12db52.png)

  - multi scale image

  ![image](https://user-images.githubusercontent.com/71866756/147486003-992aa8ae-3c0f-4346-a09e-cd146faecc2a.png)

  layer가 깊어질수록 더 나은 성능을 보였으며, C와 D의 경우, hidden layer에서 conv1, conv3를 사용했냐의 차이인데, conv1을 사용했을 때가 conv3를 사용한 경우보다 성능이 좋지 못함

  -> conv을 이용한 spatial context( 공간적 맥락 )를 잘 파악하는 것이 중요하다.  



# 2. ResNet

- **목적**

  layer를 무작정 깊게 쌓는다고 해서 성능이 좋아지는 것이 아니란 것을 실험적으로 도출하게 되었고, 층을 깊게 쌓으면서 좋은 성능을 내기 위해 다른 방법이 필요했다. ResNet은 이 필요성에 맞춰 새로운 방식을 도입하였다. 그것은 Residual Block의 출현이다. residual block은 입력값을 출력값에 더해줄 수 있도록 지름길 (shortcut)을 만들어준 것이다. 

  ![image](https://user-images.githubusercontent.com/71866756/147486174-c800f379-19e4-4772-90c5-6acac0c58bda.png)

  기존의 신경망은 x를 target y로 매핑하는 함수 H(x)를 얻는 것이 목적이었지만, ResNet은 F(x)+x를 최소화하는 것을 목적으로 한다. x는 현시점에서 변하지 않는 값으로 F(x)를 최소화하는 것을 목적으로 한다. 즉, H(x)-x를 최소화하는 것으로 이를 잔차 (residual)이라고 한다.

  이 방식을 통해 gradient vanishing/exploding 문제를 해결하였다. 

- **Architecture**

  - input image
    - input image size 224*224 RGB image
  - VGG-19 base
    - VGG-19를 기본 베이스 뼈대로 하여 convolution layer와 shortcut을 추가한 것
    - shortcut에서 channel 수를 맞춰서 합을 구함

  ![image](https://user-images.githubusercontent.com/71866756/147486208-64249115-26d8-421f-b9ef-d8dffe080c46.png)![image](https://user-images.githubusercontent.com/71866756/147486238-9ce63988-f1bc-488b-bc30-e6d2288c1328.png)

  

- **결과**

  ![image](https://user-images.githubusercontent.com/71866756/147486268-89251097-344d-4137-a40a-9331d64d974d.png)

  plain구조에서 18층의 error가 34층의 error보다 낮은 것을 알 수 있다. 하지만 ResNet에서는 34층의 결과가 더 좋은 것으로 미루어 보아 residual 방식의 효과를 알 수 있다. 



# 3. AlexNet

- **목적**

  AlexNet은 대량의 이미지 분류를 위해 만들어진 모델이다. 기본 뼈대는 LeNet-5로 되어있지만, 두개의 GPU를 사용하여 연산속도를 증가시켰다. 또한 AlexNet에서는 dropout기법을 사용했는데, 이 AlexNet 이후 CNN 구조의 GPU 구현과 dropout 적용이 보편화되었다. 

  AlexNet은 overfitting을 방지하기 위해 다양한 기법들을 사용하였다. 

- **용어 정리**

  - label-preserving transformation

    : data augmentation 기법 중 상하반전과 같은 기법을 사용할 때, label이 그대로 유지되는 것을 의미한다. MNIST의 경우 6을 반전시키면 9가 나오기 때문에 label이 유지되지 않는다. 

- **Architecture**

  ![image](https://user-images.githubusercontent.com/71866756/149939996-b05cf8a8-b6e1-4a86-bc26-e21aa3ae8513.png)


  - input image

    ImageNet dataset을 이용하였는데, image의 크기는 227x227로 고정되어있다. 이는 FC layer의 입력 크기를 맞추기 위해서이다. resize 방식은 이미지의 넓이와 높이 중 더 짧은 쪽을 227로 고정시키고 중앙 부분을 자르는 center crop을 이용하였다. 

  - conv layer

    총 5개의 conv layer가 사용되었다. 

    - conv1 

      96 kernels of size 11x11, stride=4, padding=0

    - conv2

      256 kernels of size 5x5, stride=1, padding=2

    - conv3

      384 kernels of size 3x3, stride=1, padding=1

    - conv4

      384 kernels of size 3x3, stride=1, padding=1

    - conv5

      256 kernels of size 3x3, stride=1, padding=1

    두 개의 GPU로 나눠서 학습했지만, 2번째에서 3번째 conv layer에서는 두개의 GPU가 서로 연산을 뒤섞어서 진행하였다. 

  - pooling layer

    Overlapping pooling 기법을 사용하였다. 요즘에는 보통 pooling하는 size와 stride를 일치시키지만, 이 모델에서는 overlapping하여 pooling을 진행한다. 

    ![image](https://user-images.githubusercontent.com/71866756/149940054-fce9241d-61a0-4821-a426-fe113bf469e4.png)

  - FC layer

    세개의 fully connected layer로 구성되어있다. 1번째와 2번째에 dropout 기법을 이용하였다. 

  - 그 외

    - data augmentation

      : label-preserving transform

    - Dropout

      : dropout을 통해 training 시간을 줄이며, overfitting을 방지할 수 있다.

    - ReLU

    - RGB intensify (PCA)

    - Multi GPUs

    - Local Response Normalization

      ![image](https://user-images.githubusercontent.com/71866756/149940119-22d3b4da-bfce-44ad-a11a-9d794918b9cc.png)

      : 강한 자극이 주변의 약한자극을 전달하는 것을 막는 효과를 준다. conv filter의 결과가 매우 높다면 그 주변 conv filter의 결과값이 상대적으로 작아진다. 

      -> 극단적인 경우 dropout 효과를 일으킬 수 있다. 

- **결과**

  ![image](https://user-images.githubusercontent.com/71866756/149940158-ab9df865-678e-4939-a722-f2056527ddc7.png)





# 4. DenseNet

- **목적**

  이전의 연구에서 증명됐듯이, input과 output에 가까운 layer들의 connection이 짧을 수록 training에 효율적이며, 더 정확하고, 대체로 더 깊어질 수 있다.  따라서 DenseNet에서는 feed forward 방향으로 모든 layer들이 연결되어 있으며, 각 layer의 feature map은 그 다음 layer의 input으로 들어가게 된다. 이러한 구조로 ResNet보다 더 적은 parameter들로 CIFAR-10, CIFAR-100, SVHN, ImageNet 데이터들을 학습시킨 결과 의미있는 향상을 보였다. 

  이러한 dense한 연결성 때문에 DenseNet이라 부르게 되었다. 

  

- **용어 정리**

  

- **Architecture**

  ![image](https://user-images.githubusercontent.com/71866756/149940719-9f4c37fe-291d-479a-95fa-56219a20488a.png)

  - input image

    - input image size 224x224 RGB

      

  - Dense block

    - 이전 layer들의 feature map이 입력으로 들어가기 위해서 concatenation이 적용되고, concatenation이 적용되기 위해서는 feature map의 사이즈가 동일해야 한다. 따라서 down sampling이 필요하며, 이 과정을 반복적으로 수행하는 것이 dense block이다. 

    - 총 3개의 Dense block으로 구성

      

  - transition layers

    - Dense block 사이의 layer를 지칭한다. 

    - Batch normalization , 1x1 conv, 2x2 average pooling으로 구성되어 있다. 

      

  - Bottleneck layers

    - 1x1 conv로 구성되어 있으며, 3x3 conv 이전에 feature map의 사이즈를 줄이기 위한 layer이다.  

    - 1x1 conv를 통해 4*k의 feature map을 생성한다. ( k는 임의 조정 ) 

      

  - Conv

    - 3x3 conv의 경우, zero-padding으로 feature map의 사이즈는 동일하게 유지한다.

       

  - Growth rate

    - dense block 내의 레이어는 k개의 feature map을 생성하고, 이 k를 growth rate이라고 한다. 

    - growth rate은 각 레이어가 전체에 어느정도 기여를 할 것인지 결정한다. 

      

  - 마지막 단

    - global average pooling과 softmax 적용

      

  - 그 외

    - bottleneck layer와 transition layer에서 feature map의 사이즈를 줄이지만, Dense block내에서의 layer들 사이의 connection을 통해 layer의 output feature map 개수를 크게 줄일 수 있다 (다음 layer의 input으로 이전 layer들에서의 feature map을 concatenate하기 때문에).
    - dense block에서 이전 layer들의 feature map을 재사용함으로 output feature map을 크게 줄이고, size도 줄일 수 있기 때문에 computational efficiency를 높일 수 있다. (parameter 개수가 줄어드는 것 -> 딥러닝에서 중요한 것은 parameter의 수를 줄이면서 좋은 성능을 내는 것이기 때문에 바람직하다.)
    - shortcut을 만드는 것으로, 트레이닝 시간을 줄이고, 확률적으로 layer를 drop하는 것이 generalize performance의 향상으로 이어진다. 
    - 최종 결과물이 모든 feature map을 기반으로 결정을 내리기 때문에 좋은 결과를 낼 가능성이 높아진다. 



- **종류 및 구조**

  ![image](https://user-images.githubusercontent.com/71866756/149940234-e2c5d1a0-8cc1-414b-a71f-4e01d83bb6b8.png)

  

- **결과**

  ![image](https://user-images.githubusercontent.com/71866756/149940324-b0202350-c0bf-436a-bf0f-b2cd762b42f1.png)

  ![image](https://user-images.githubusercontent.com/71866756/149940382-69e301d6-071e-46b9-bb4a-e851462f3532.png)

  - 위 설명에서는 data augmentation 없이도 다른 network보다 우수한 성능을 낸다고 나와있다. (dropout은 사용)

  - 위에서 '+'는 표준 data augmentation인 mirroring과 shifting을 의미한다. 

  - optimizer로는 SGD를 사용 (weight decay=10^-4, momentum=0.9)

  - `CIFAR와 SVHN` : 초기 learning rate=0.1이지만, epoch의 50%일 때, 10으로 나누고, 75%일 때 10으로 다시 나눠준다.

    `ImageNet`  : 초기 learning rate=0.1, 30 epoch=0.01, 60 epoch=0.001 
