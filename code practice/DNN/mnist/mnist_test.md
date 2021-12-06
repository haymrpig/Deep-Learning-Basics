- 구글 드라이브 연동하기

```python
from google.colab import drive
drive.mount('/content/gdrive/')
```

- 필요한 모듈 import

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, datasets, utils
from ipywidgets import interact
import matplotlib.pyplot as plt
import pandas as pd
```

- 출력 옵션 설정

```python
np.set_printoptions(linewidth=200, precision=2)
```

- keras에서 제공하는 기본 데이터셋 load하기

```python
(train_datas, train_labels), (test_datas, test_labels) = datasets.mnist.load_data()
```

- train data 확인하기

```python
# unique method는 딕셔너리처럼 생성한다. 
# 현재 train_labes를 key로 return_count=True는 각각의 key에 해당하는 개수를 반환한다. 
unique, counts = np.unique(train_labels, return_counts=True)
num_labels = len(unique)
print("Train labels: {}, labels : {}".format(dict(zip(unique, counts)), num_labels))
```

![image-20211207033837759](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211207033837759.png)

```python
# decorator를 통해 train data를 모두 살펴볼 수 있다. 
# interact는 사용자의 입력을 넣어줄 수 있게 만든 메소드이다.
@interact(idx=(0, train_datas.shape[0]-1))
def showImage(idx):
    plt.imshow(train_datas[idx], cmap="gray")
    plt.grid(False)
    plt.title("LABEL : {}".format(train_labels[idx]))
    plt.show()
```

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211207033946743.png" alt="image-20211207033946743" style="zoom:67%;" />

```python
@interact(idx=(0, train_datas.shape[0]-1))
def showImage(idx):
    print(train_datas[idx])
```

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211207034013861.png" alt="image-20211207034013861" style="zoom:67%;" />

- data 크기 변경하기

```python
# 이미지가 현재 28*28사이즈인데, 이 사이즈를 한 줄로 바꾸는 과정이다. 
train_datas = train_datas.reshape(60000, 28*28).astype("float32")
test_datas = test_datas.reshape(10000, 28*28).astype("float32")
```

- model 구성하기

```python
inputs = tf.keras.Input(shape=(28*28,))
hidden = layers.Dense(64, activation="sigmoid")(inputs)
hidden = layers.Dense(64, activation="sigmoid")(hidden)
outputs = layers.Dense(10)(hidden)

model = models.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()
```

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211207034311325.png" alt="image-20211207034311325" style="zoom:67%;" />

```python
utils.plot_model(model, "model.png", True)
```

```python
# 사실 위에서 마지막 Dense에서 activation함수를 적용하여 어떤 레이블이 1이 나오는지 정해주어야 한다.
# ex) outputs = layers.Dense(10, activation="softmax")
# 하지만 아래에서 Sparse, from_logits=True로 함으로써 자동으로 활성함수가 붙은 것과 같은 동작을 하게 된다. 
# 결과가 Scalar로 해당 레이블만 1인 결과가 나와도 이 과정을 거치면 10개의 레이블이 0000100000 이런식으로 one-hot
# encoding이 된다. 
# 보통 loss값만 확인하지만, accuracy까지 확인하기 위해 metrics=["accuracy"]를 추가한다. 
model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizers.RMSprop(),
    metrics=["accuracy"],
)
```

- train 단계

```python
# validation_split은 cross validation개념으로 학습데이터 중 20%를 validation data로 보겠다는 의미이다. 
history = model.fit(train_datas, train_labels, batch_size=64, epochs=5, validation_split=0.2)
```

![image-20211207034415320](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211207034415320.png)

- 평가 단계

```python
# verbose : 학습 과정에서 진행 내용을 출력하기 위한 모드 설정
# 0 : 출력안함
# 1 : Progress Bar(진행바)로 출력
# 2 : 각 epoch마다 한줄 씩 출력
test_scores = model.evaluate(test_datas, test_labels, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
```

![image-20211207034443992](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211207034443992.png)

- train한 결과 확인하기

```python
history_df = pd.DataFrame(history.history)
history_df[["loss", "val_loss"]].plot()
```

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211207034527348.png" alt="image-20211207034527348" style="zoom:67%;" />

```python
history_df
```

![image-20211207034544774](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211207034544774.png)

```python
history_df[["accuracy", "val_accuracy"]].plot()
```

<img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211207034608599.png" alt="image-20211207034608599" style="zoom:67%;" />

- 모델 저장하기

```python
model.save("mnist_hong.h5")
del model
```

- 모델 load

```python
model = models.load_model("mnist_hong.h5")
```