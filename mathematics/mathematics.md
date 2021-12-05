# 목차

1. 미분

2. 조건부 확률
   - 베이즈 정리
   - 인과관계

# 1. 미분(differentiation)

- **정의**

  변수의 움직임에 따른 함수값의 변화를 측정하기 위한 도구

  ( 접선의 기울기를 나타낸다. )

![image](https://user-images.githubusercontent.com/71866756/144754284-27e41f3b-336e-459c-97d4-23aabbcdc07e.png)

- **미분 코드**

  - 단일변수

  ```python
  import sympy as sym
  from sympy.abc import x
  
  sym.diff(sym.poly(x**2+2*x+3), x)
  ```

  - 다변수 (편미분)

  ```python
  import sympy as sym
  from sympy.abc import x,y
  
  sym.diff(sym.poly(x**2+2*x*y+3), x)
  ```

  

- **경사하강법/상승법**에서의 미분

  - 경사하강법 (극소값) : 현재 값 - 미분값

  $$
  -(df/dx, df/dy)
  $$

  <img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211206001749978.png" alt="image-20211206001749978" style="zoom:67%;" />![image-20211206001643670](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211206001643670.png)

  - 경사상승법 (극대값) : 현재 값 + 미분값

  $$
  (df/dx, df/dy)
  $$

  <img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211206001536841.png" alt="image-20211206001536841" style="zoom: 67%;" />![image-20211206001643670](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211206001643670.png)



# 2. 조건부 확률

- **정의**

  사건 A/B가 일어난 상황에서 사건 A/B가 발생할 확률

  

  <img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211206002245517.png" alt="image-20211206002245517" style="zoom:50%;" />

  

- **베이즈 정리**

  조건부 확률을 이용하여 정보를 갱신하는 방법으로 새로운 데이터가 들어왔을 때 앞서 계산한 사후확률을 사전확률로 사용하여 갱신된 사후확률을 계산할 수 있다. 

  

  

  <img src="C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211206002351336.png" alt="image-20211206002351336" style="zoom:50%;" />

  

- **인과관계**

  데이터 분포의 변화에 강건한 예측모델을 만들 때 사용한다. 

  -  주의점

    1. 중첩요인의 효과를 제거하고 원인에 해당하는 변수만의 인과관계를 계산해야 한다. 
    2. 조건부 확률을 함부로 적용하면 안된다. 

  - 인과관계의 추론 예제

    - 조건부 확률로 구한 결과

      ![image-20211206003550478](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211206003550478.png)

    - Z의 개입을 제거(조정)한 결과

      ![image-20211206003719507](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211206003719507.png)

      ![image-20211206003812015](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20211206003812015.png)

      

      **!!! 정반대의 결과가 나오는 것을 확인할 수 있다 !!!!**

      



















