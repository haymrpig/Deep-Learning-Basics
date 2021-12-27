# 다양한 MODEL

- **목차**
  1. [VGGNet](#1-vggnet)
  2. [ResNet](#2-resnet)

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

  <img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211223174511518.png" alt="image-20211223174511518" style="zoom:67%;" />

- **결과**

  - single scale image

  <img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211223174736075.png" alt="image-20211223174736075" style="zoom:67%;" />

  - multi scale image

    <img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211223175342184.png" alt="image-20211223175342184" style="zoom:67%;" />

  layer가 깊어질수록 더 나은 성능을 보였으며, C와 D의 경우, hidden layer에서 conv1, conv3를 사용했냐의 차이인데, conv1을 사용했을 때가 conv3를 사용한 경우보다 성능이 좋지 못함

  -> conv을 이용한 spatial context( 공간적 맥락 )를 잘 파악하는 것이 중요하다.  



# 2. ResNet

- **목적**

  layer를 무작정 깊게 쌓는다고 해서 성능이 좋아지는 것이 아니란 것을 실험적으로 도출하게 되었고, 층을 깊게 쌓으면서 좋은 성능을 내기 위해 다른 방법이 필요했다. ResNet은 이 필요성에 맞춰 새로운 방식을 도입하였다. 그것은 Residual Block의 출현이다. residual block은 입력값을 출력값에 더해줄 수 있도록 지름길 (shortcut)을 만들어준 것이다. 

  <img src="../../../../AppData/Roaming/Typora/typora-user-images/image-20211227233916174.png" alt="image-20211227233916174" style="zoom:67%;" />

  기존의 신경망은 x를 target y로 매핑하는 함수 H(x)를 얻는 것이 목적이었지만, ResNet은 F(x)+x를 최소화하는 것을 목적으로 한다. x는 현시점에서 변하지 않는 값으로 F(x)를 최소화하는 것을 목적으로 한다. 즉, H(x)-x를 최소화하는 것으로 이를 잔차 (residual)이라고 한다.

- **Architecture**

  - input image
    - input image size 224*224 RGB image
  - VGG-19 base
    - VGG-19를 기본 베이스 뼈대로 하여 convolution layer와 shortcut을 추가한 것
    - shortcut에서 channel 수를 맞춰서 합을 구함

  <img src="../../../../AppData/Roaming/Typora/typora-user-images/image-20211228002543411.png" alt="image-20211228002543411" style="zoom:67%;" /><img src="../../../../AppData/Roaming/Typora/typora-user-images/image-20211228002703523.png" alt="image-20211228002703523" style="zoom:67%;" />

  

- **결과**

  ![image-20211228002807505](../../../../AppData/Roaming/Typora/typora-user-images/image-20211228002807505.png)

  plain구조에서 18층의 error가 34층의 error보다 낮은 것을 알 수 있다. 하지만 ResNet에서는 34층의 결과가 더 좋은 것으로 미루어 보아 residual 방식의 효과를 알 수 있다. 
