# PYTORCH

목차

- **Tensor**
- **PyTorch 문법**
- **Linear Regression**
- **학습데이터 불러오기**













# 1. Tensor

- pytorch에서의 tensor 표현

  ![image](https://user-images.githubusercontent.com/71866756/146793425-faab3035-ab14-41fa-902e-3e348ff5fa91.png)


  

# 2. PyTorch 문법

- **1D Array with PyTorch**

  ```python
  t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
  
  print( t.dim() )    # rank
  print( t.shape )    # shape
  print( t.size() )   # shape
  print( t[0] )
  print( t[2:5] )
  ```

  

- **Broadcasting**

  ```python
  # 같은 위치의 원소끼리 연산 수행
  m1 = torch.FloatTensor([3, 3]) 
  m2 = torch.FloatTensor([1, 1])
  print(m1 + m2)
  
  # shape이 다른 경우, 같은 shape을 가지도록 확장시킨 후 연산 수행
  m1 = torch.FloatTensor([3, 3]) 
  m2 = torch.FloatTensor([1])
  print(m1 + m2) # [4, 4]
  
  m1 = torch.FloatTensor([[1, 2]]) 	# [[1,2],[1,2]]로 확장
  m2 = torch.FloatTensor([[3], [4]]) 	# [[3,3],[4,4]]로 확장
  print(m1 + m2) #[[4, 5], [5, 6]]
  ```

  

- **Multiplication v.s. Matrix Multiplication**

  ```python
  m1 = torch.FloatTensor([1,2],[3,4])
  m2 = torch.FloatTensor([[1],[2]])
  print(m1*m2)			# 단순 곱셈
  print(m1.mul(m2))		# 단순 곱셈
  print(m1.matmul(m2))	# 행렬곱
  ```

  

- **Mean**

  ```python
  t = torch.FloatTensor([1,2])
  print( t.mean() )	# 1.5, 평균 구하는 method
  
  t = torch.FloatTensor([[1,2],[3,4]])
  print( t.mean() )		# 2.5
  print( t.mean(dim=0) )	# [2., 3.]
  print( t.mean(dim=1) )	# [1.5, 3.5]
  print( t.mean(dim=-1) )	# [1.5, 3.5]
  
  # integer의 경우 mean method 사용 불가
  t = torch.LongTensor([1,2])
  print( t.mean() )	# error!! 
  
  
  ```

  

- **Sum**

  ```python
  t = torch.FloatTensor([[1,2],[3,4]])
  print( t.sum() )		# 10
  print( t.sum(dim=0) )	# [4,6]
  print( t.sum(dim=1) )	# [3,7]
  print( t.sum(dim=-1) )	# [3,7]
  
  ```

  

- **Max v.s. Argmax**

  ```python
  t = torch.FloatTensor([[1,2],[3,4]])
  print( t.max() )			# 4, 가장 큰 원소 출력
  print( t.max(dim=0) )		# [3,4], [1,1] -> max원소 출력, max원소의 index 출력
  print( t.max(dim=0)[0] )	# max 원소 출력
  print( t.max(dim=0)[1] )	# Argmax, 즉 max원소 index 출력
  ```

  

- **View( Reshape )**

  ```python
  t = np.array([[[0,1,2],
  			   [3,4,5]],
  				
                [[6,7,8],
  			   [9,10,11]]])
  ft = torch.FloatTensor(t)	# (2,2,3)의 shape을 가진다.
  print( ft.view([-1, 3]) )	# (2,2,3) -> (?,3)으로 바꾼다.
  							# 즉 (2,2,3) -> (4,3)
     							# [[0,1,2],
          					#  [3,4,5],
              				#  [6,7,8],
                  			#  [9,10,11]]
  ```

  

- **Squeeze** **& Unsqueeze**

  ```python
  ft = tensor.FloatTensor([[0],[1],[2]])	# shape : (3,1)
  print( ft.squeeze() )		# [0,1,2], shape에서 1인 차원을 없앤다.
  print( ft.squeeze(dim=0) )	# shape : (3,1)에서 dim=0는 3이므로 차원을 축소 X
  print( ft.squeeze(dim=1) )	# shape : (3,1)에서 dim=1은 1이므로 차원을 없앤다. 
  
  ft = torch.FloatTensor([0,1,2])	# shape : (3, )
  print( ft.unsqueeze(0) ) 	# [[0,1,2]], dim=0 추가 -> shape : (1,3)
  print( ft.unsqueeze(1) )	# [[0],[1],[2]], dim=1 추가 -> shape : (3,1)
  print( ft.view(1,-1) )		# [[0,1,2]], dim=0은 1로 reshape
  ```

  

- **Type Casting**

  ```python
  lt = tensor.LongTensor([1,2,3,4])
  print( lt.float() )
  
  bt = torch.ByteTensor([True, False, False, True])
  print( bt.long() )
  print( bt.float() )
  ```

  

- **Concatenate( cat & stack )**

  ```PYTHON
  x = torch.FloatTensor([[1,2],[3,4]])
  y = torch.FloatTensor([[5,6],[7,8]])
  print( torch.cat([x,y], dim=0) )	# [[1,2],[3,4],[5,6],[7,8]]
  print( torch.cat([x,y], dim=1) )	# [[1,2,5,6],[3,4,7,8]]
  
  x = torch.FloatTensor([1,4])
  y = torch.FloatTensor([2,5])
  z = torch.FloatTensor([3,6])
  print( torch.stack([x,y,z]) )			# [[1,4],[2,5],[3,6]]
  print( torch.stack([x,y,z], dim=1) )	# [[1,2,3],[4,5,6]]
  print( torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0) )
  # [[1,4],[2,5],[3,6]]
  ```

  

- **Ones and Zeors**

  ```python
  x = torch.FloatTensor([[0,1,2],[2,1,0]])
  print( torch.ones_like(x) )
  print( torch.zeros_like(x) )
  ```

  

- **In-place Operation**

  ```python
  x = torch.FloatTensor([[1,2],[3,4]])
  print( x.mul(2.) )		# x에 결과값이 저장되지 않는다.
  print( x.mul_(2.) )		# x에 결과값이 저장된다.
  ```

  

- 

# 3. Linear Regression

- **단일 변수 선형회귀**

```python
from torch import optim
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

# weight와 bias는 0으로 초기화
# requires_grad=True, 학습할 것이라고 명시
w = torch.zeros(1, requires_grad=True)  
b = torch.zeros(1, requires_grad=True)

optimizer = optim.Adam([w,b], lr=0.1)

nb_epochs = 1000

for epoch in range(1, nb_epochs + 1):
    hypothesis = w*x_train + b
    cost = torch.mean((hypothesis-y_train)**2)
    
    optimizer.zero_grad()   # zero_grad()로 gradient 초기화
    cost.backward()         # backward()로 gradient 계산
    optimizer.step()        # step()으로 개선
    if( epoch % 100 == 0 ):
        print( ">>>>>>> weight : {}, bias : {}, cost : {}".format(w.item(), b.item(), cost.item()) )
```



- **다변수 선형회귀**

```python
from torch import optim
x_train = torch.FloatTensor([[73,80, 75],[93,88, 93],[89,91, 90],[96,98, 100],[73,66, 70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

# weight와 bias는 0으로 초기화
# requires_grad=True, 학습할 것이라고 명시
w = torch.zeros((3,1), requires_grad=True)  
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([w,b], lr=0.00001)

nb_epochs = 20

for epoch in range(1, nb_epochs + 1):
    hypothesis = x_train.matmul(w) + b
    cost = torch.mean((hypothesis-y_train)**2)
    
    optimizer.zero_grad()   # zero_grad()로 gradient 초기화
    cost.backward()         # backward()로 gradient 계산
    optimizer.step()        # step()으로 개선

    print( ">>>>>>> weight : {}, bias : {}, cost : {}".format(w.squeeze(), b.squeeze(), cost.item()) )
```



- **torch.nn을 이용한 간편한 구현**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)

    def forward(self, x):
        return self.linear(x)

x_train = torch.FloatTensor([[73,80, 75],[93,88, 93],[89,91, 90],[96,98, 100],[73,66, 70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

model = MyModel()
optimizer = optim.SGD(model.parameters(), lr = 1e-5)

for epoch in range(10001):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if( epoch % 1000 == 0):
        print("Epoch   {}  cost {} ".format(epoch, cost) )

test = model(x_train)
print( "test_output :", test )
```



# 4. 학습 데이터 불러오기

- **PyTorch 제공 데이터셋 CIFAR10 이용하기**

  ```python
  import torch
  import torchvision
  import torchvision.transforms as tr # 전처리를 위한 라이브러리
  from torch.utils.data import DataLoader, Dataset
  import numpy as np
  
  transf = tr.Compose([tr.Resize(8), tr.ToTensor()]) # data 전처리
  
  trainset = torchvision.datasets.CIFAR10(root='/data',
                                          train=True,
                                          download=True,
                                          transform=transf)
  testset = torchvision.datasets.CIFAR10(root='/data',
                                         train=False,
                                         download=True,
                                         transform=transf)
  
  trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
  testloader = DataLoader(testset, batch_size=8, shuffle=True, num_workers=2)
  
  dataiter = iter(trainloader)	# iter를 이용하여 이미지를 확인할 수 있다.
  images, labels = dataiter.next()	# 현재 images에는 배치사이즈만큼의 image가 저장되어 있음
  ```



- **개인 데이터셋 이용하기**

  ```python
  import torch
  import torchvision
  import torchvision.transforms as tr # 전처리를 위한 라이브러리
  from torch.utils.data import DataLoader, Dataset
  import numpy as np
  
  train_images = np.random.randint(256, size=(20,32,32,3))
  train_labels = np.random.randint(2, size=(20,1))
  
  class TensorData(Dataset):
      def __init__(self, x_data, y_data):
          self.x_data = torch.FloatTensor(x_data)
          self.x_data = self.x_data.permute(0,3,1,2) # torch의 경우 (batch, channel, width, height)순이므로 permute메소드를 이용하여 입력 데이터의 순서를 바꿔준다.
          self.y_data = torch.LongTensor(y_data)
          self.len = self.y_data.shape[0]
  
      def __getitem__(self, index):
          return self.x_data[index], self.y_data[index]
  
      def __len__(self):
          return self.len
      
  train_data = TensorData(train_images, train_labels)
  train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
  
  dataiter = iter(train_loader)
  images, labels = dataiter.next()
  
  
  ```



- **개인 데이터셋 이용하기 2**

  ```python
  class MyDataset(Dataset):
      def __init__(self, x_data, y_data, transform=None):
          self.x_data = x_data
          self.y_data = y_data
          self.transform = transform
          self.len = len(y_data)
  
      def __getitem__(self, index):
          sample = self.x_data[index], self.y_data[index]
  
          if self.transform:
              sample = self.transform(sample)
  
          return sample
  
      def __len__(self):
          return self.len
  
  class ToTensor:
      def __call__(self, sample):     # 인스턴스가 호출될 때 실행된다. 이런식으로 전처리 클래스를 만들어줄 수 있다. 
          inputs, labels = sample
          inputs = torch.FloatTensor(inputs)
          inputs = inputs.permute(2,0,1)
          return inputs, torch.LongTensor(labels)
  
  
  trans = tr.Compose([ToTensor()])
  dest1 = MyDataset(train_images, train_labels, transform=trans)
  train_loader = DataLoader(dest1, batch_size=10, shuffle=True)
  ```

  

```python
class MyDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        self.len = len(y_data)

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.len

class MyTransform:
    def __call__(self, sample):     # 인스턴스가 호출될 때 실행된다. 이런식으로 전처리 클래스를 만들어줄 수 있다. 
        inputs, labels = sample
        inputs = torch.FloatTensor(inputs)
        inputs = inputs.permute(2,0,1)
        labels = torch.LongTensor(labels)

        transf = tr.Compose([tr.ToPILImage(), tr.Resize(128)])
        final_output = transf(inputs)
        return final_output, labels


#trans = tr.Compose([ToTensor()])
dest1 = MyDataset(train_images, train_labels, transform=MyTransform)
train_loader = DataLoader(dest1, batch_size=10, shuffle=True)
```









