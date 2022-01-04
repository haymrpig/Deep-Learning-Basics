# 목차

1. [행렬](#1-행렬)
2. [선형결합](#2-선형결합)
   - Span
   - Sum of outer product
   - Subspace
   - Basis( 기저 )
3. [선형독립/종속 (Linearly independent/dependent)](#3-선형독립/종속)
4. [선형변환](#4-선형변환)
   - 용어정리
   - ONTO( 전사 ) / ONE-TO-ONE( 일대일 )
5. [Least Squares](#5-least-squares)
   - 용어정리
   - 정규방정식
6. [Projection](#6-projection)
   - Orthogonal basis
   - Gram-Schmidt Orthogonalization
7. [고유벡터 (Eigenvectors) & 고유값 (Eigenvalues)](#7-eigenvectors&eigenvalues)
8. [영공간 (Null Space)](#8-영공간-null-space)

# 1. 행렬

- **역행렬( Inverse matrix )**

  자기자신과 자신의 역행렬을 곱하면 항등행렬이 나오는 행렬. 역행렬이 존재하는 경우 해는 단일해가 나온다. 

  NxN 정방행렬은 역행렬을 가지지만, 정방행렬이 아닌 경우 역행렬이 존재하지 않는다. 

  - 역행렬 판별식 ( determinant 이용)

    2x2행렬에선 ad-bc를 determinant라 부르며, 이 값이 0일 경우 역행렬이 존재하지 않는다.

  - 역행렬이 존재하지 않는 경우

    - 해가 없는 경우( over-determined system )

      데이터의 개수가 feature의 개수보다 많은 경우

    - 해가 무수히 많은 경우( under-determined system )

      데이터의 개수가 feature의 개수보다 적은 경우

      -> regularization을 통해 해결할 수 있다. 

  ```python
  from numpy.linalg import inv
  # 역행렬 구하는 함수
  A_inv = inv(A)
  ```

  ```python
  # 이 둘은 같은 식으로 보통 직접 inverse를 구하진 않는다.( 분수로 인한 오차 문제 때문 )
  x = A_inv.dot(b)
  x = solve(A,b)
  ```

- **항등행렬(  Identity matrix ) **

  어떤 행렬을 곱해도 결과가 그 행렬이 나오는 행렬

  

# 2. 선형결합

벡터의 일정값을 곱해주어 결합하는 것을 선형결합이라고 한다. 

![image](https://user-images.githubusercontent.com/71866756/145265004-c357fde6-0411-4d0e-82f9-f0326bd3dbef.png)

- **Span**

  가능한 모든 선형 결합의 집합을 의미한다. 아래 그림처럼 coefficient를 변경하여 만들어진 모든 집합을 Span이라고 한다.

  ![image](https://user-images.githubusercontent.com/71866756/145265083-2a7036bd-baa1-46ec-8f6f-c4fa90f2f5bc.png)

- **Sum of (Rank-1) outer product**

  ![image](https://user-images.githubusercontent.com/71866756/145265135-f33952ed-43d2-4407-9b33-db19a6d9b106.png)


  복잡한 행렬곱도 행렬의 분해를 통해 간단하게 계산을 할 수 있다. 

  - 공분산, style transfer 등 다양하게 쓰인다. 

- **Subspace**

  span이랑 비슷하지만 닫혀있다는 개념이 추가된 것이 subspace이다. 즉,  모든 부분집합끼리 또는 부분집합의 연산이 부분집합이 되어야 닫혀있다고 할 수 있다. 여기서는 선형결합에 닫혀있다고 할 수 있다. 

  예를 들어, S={2,4,6, ... }으로 2의 배수를 원소로 갖는 S의 경우는 곱셈에 닫혀있다고 할 수 있다( 어떤 부분집합의 곱이 S의 부분집합에 속하므로 ).

  - Dimension of Subspace

    basis( 기저 벡터 )의 개수가 dimension( Rank )이다.

    여기서 Rank가 중요한 이유는 학습에 있어서 여러가지의 feature가 존재하고 각각의 feature를 하나의 벡터라고 해보자. 만약 v1이 3*v2라고 한다면 이 두 벡터는 겹친다고 할 수 있어 feature로서의 역할을 하지 못한다. 이 둘 중 하나의 벡터는 쓸모가 없어지는 벡터( 선형 종속 )가 되기 때문에 이 때 basis 즉, rank를 파악하는 것이 중요하다.  

- **Basis of Subspace**

  기저벡터라고 하며, 기저벡터는 주어진 subspace를 fully span하며 선형 독립이여야 한다. 즉, full span이라는 의미는 2차원을 예로 들면 2개의 벡터를 사용든, 3개의 벡터를 사용하든 한 평면을 모두 표현가능해야 한다는 의미이다. 선형독립은 아래 설명되어 있다. 

  - 기저 벡터는 unique하지 않다!

    즉, 2차원을 예로 들어 한 평면을 표현하기 위한 벡터는 다양한 조합이 가능하다. 이 조합들이 모두 기저벡터가 될 수 있어 unique하지 않다.( 단, 이 벡터의 조합은 선형독립이며 fully span해야 한다. )

  - Standard basis vector

    가장 간단한 기저 벡터로 3차원의 경우, [1 0 0], [0 1 0], [0 0 1]이라고 할 수 있다. 

  - Column space

    열벡터가 subspace인 경우

# 3. 선형독립/종속

- **선형독립 (Linearly independent)**

  Ax=b에서 b가 0벡터인 경우, x도 0벡터로 무조건 해를 가지게 된다.이 때의 해를 trivial solution이라 하고, 만약 x=0인 벡터인 것을 제외하고 해가 존재하지 않는 경우 선형독립이라 하고, 또 **다른 해( non trivial solution )가 존재**하는 경우 **선형종속**이라고 한다. 

- **선형종속 (Linearly dependent)**

  Span을 생각해보았을 때, 벡터가 추가될 때 기존 Span에 포함이 된다면 선형종속이라고 한다. 

  즉, 기존의 벡터들로 충분히 해를 구할 수 있었을 때, 새로운 벡터가 들어와도 그 벡터가 해를 구하는데 영향을 미치지 않아도 되는 경우 선형종속이다.  새로운 벡터는 기존의 벡터들로 표현이 된다. 

  **!!! 3차원 공간에서 4개의 벡터가 주어지면 무조건 선형종속이 될 수밖에 없다 !!!**



# 4. 선형변환

선형변환이란 함수에 넣기 전에 두 input의 결합을 넣어 얻은 결과와 각각의 input을 따로 넣어 결과를 결합한 것이 일치할 때 선형변환이라고 한다. 

![image](https://user-images.githubusercontent.com/71866756/145265239-967807df-ea6d-4185-9a18-5a1db19028aa.png)

- **용어 정리**

  - Function( 함수 )

    모든 정의역은 치역을 가지고 있어야 한다. 그렇지 않은 경우 함수라고 부르지 못한다. 

  - Domain( 정의역 )

    input

  - Co-domain( 공역 )

    치역이 될 수 있는 집합

  - Image( 상 또는 함수의 output )

    input에 따른 함수의 output을 의미한다.

  - Range( 치역 )

    정의역에 매핑되는 정답

    ![image](https://user-images.githubusercontent.com/71866756/145265299-1e719be0-e6ad-4c2c-8105-ff95fc1b9b05.png)

- 선형변환을 만족할 경우

  ![image](https://user-images.githubusercontent.com/71866756/145265351-47029257-6b67-4907-8c86-70b52cd3a4ce.png)

  항상 행렬과 입력벡터의 곱으로 나타낼 수 있다. 

  따라서 기저벡터를 입력으로 나온 출력을 결합한 것이 T라는 matrix가 된다. 

- **ONTO( 전사 )**

  치역과 공역이 같은 경우, 정의역의 개수가 공역의 개수보다 많거나 같아야 한다. 

  함수의 경우 정의역 하나당 치역 하나를 가지기 때문에, fully connected layers에서 decoding단을 보면 ONTO가 될 수 없다.

- **ONE-TO-ONE( 일대일 ) **

  정의역과 공역이 1:1로 매핑이 되는 경우, 즉 치역에 매핑되는 정의역이 하나일 경우이다. 치역에 여러개의 정의역이 매핑 되면 ONE-TO-ONE이 아니다. 

  fully connected layers의 경우 encoding단에서 ONE-TO-ONE이 될 수 없기 때문에 의도적으로 정보의 손실을 일으켜 중요한 정보만을 넘긴다고 볼 수 있다. 

  

# 5. Least Squares

![image](https://user-images.githubusercontent.com/71866756/145265433-41a5eaf5-8866-4293-9bd1-58e777bdb157.png)

- **용어 정리**

  - 내적 ( inner product )

  - 벡터의 길이 ( norm )

    - L1

      단순히 벡터의 각 요소에 대한 절대값의 합으로 구한 거리

    - L2

      실제 벡터의 직선 거리를 의미한다. 

    - 두 벡터 사이 끼인각

      ![image](https://user-images.githubusercontent.com/71866756/145265460-b038e3b0-4372-46da-ba6a-beb5f3fee37c.png)

    - 직교하는 두 벡터

      내적은 0이 나온다. 

- **정규방정식( Normal Equation )**

  ![image](https://user-images.githubusercontent.com/71866756/145265508-0b7d44d4-5a70-4e44-b196-56f8fad53eeb.png)

  - 역행렬이 존재할 경우

    역행렬을 곱해서 구할 수 있다. 

  - 역행렬이 존재하지 않을 경우( 해가 무수히 많은 경우 )

    유사역행렬을 통해 구한다. 

    ![image](https://user-images.githubusercontent.com/71866756/145265542-664bb369-2a15-44be-8ab2-f8b092c97bbf.png)



# 6. Projection

b를 projection시켜 b 햇을 구할 수 있다. 이 때 식은 위에서 구했던 식과 일치한다. 

![image](https://user-images.githubusercontent.com/71866756/145265578-90aae95e-b236-400c-b0e1-2abd7cd81a8b.png)

- Orthogonal basis

  모든 기저벡터끼리 수직인 벡터

- 그람-슈미트 직교화

  - 2차원

    기존 기저벡터와 orthogonal한 기저벡터를 구하는 방법이다. 

    ![image](https://user-images.githubusercontent.com/71866756/145265648-74afb586-c107-4d46-9f5d-58811f15281f.png)

    y의 orthogonal basis를 구하면 y-y_hat이 된다. 

    이 때 벡터 u와 벡터 y의 끼인각을 세타라고 하면 y_hat = y*cos(세타)가 되고 cos(세타)는 끼인각 공식을 이용해서 구할 수 있다.

    그 결과 아래와 같은 식을 유도할 수 있다.  

    ![image](https://user-images.githubusercontent.com/71866756/145265674-cf58a5c4-54f8-447b-8a70-6bedf563bab8.png)

    만약 u가 길이가 1인 단위벡터라면 아래와 같은 식이 된다. 

    ![image](https://user-images.githubusercontent.com/71866756/145265707-8060b3f5-ea2a-4119-88d3-bfbe22492ba4.png)

  - 다차원

    2차원에서 확장된 것이라 보면 된다. 각각의 u에 대해 같은 방식으로 진행한뒤 더하면 다차원에서의 orthogonal basis가 된다. 

    ![image](https://user-images.githubusercontent.com/71866756/145265747-d595e1d4-70ef-416d-8b09-2be192463990.png)

  - 위 식을 선형변환으로 표현할 수 있다. 

    ![image](https://user-images.githubusercontent.com/71866756/145265779-bb04d262-fc26-4028-a042-e80523a0ae56.png)

    위에서 구한 식에 의해서 만약 u1과 u2를 단위벡터라고 가정하면 결국 마지막에 b_hat은 UU^Tb가 된다!!

    

    **!!!! orthogonal basis가 중요한 이유 !!!!**

    만약 두개의 feature가 있다고 할 때, 두 벡터가 비슷한 경향을 보인다고 하자. 그럴 경우, 한 벡터는 다른 벡터의 방향으로 projection된다고 할 수 있고, 이 두 벡터는 직교에서 멀어질 것이다. 그러면 이러한 조그만 비슷한 경향성에 의해 weight값이 크게 변할 수 있고, 이러한 데이터의 오류 때문에 결과가 안 좋아질 수 있다. ( 아래 그림 )

    ![image](https://user-images.githubusercontent.com/71866756/145265807-500104d9-153f-4bf9-aed6-5dc4168d5ca5.png)

    이 때 regularization기법을 사용한다. 

    ![image](https://user-images.githubusercontent.com/71866756/145265838-5346f35e-4472-404e-98d0-8dbbc1ed7a94.png)

    

- **Gram-Schmidt Orthogonalization**

  orthogonal basis를 만드는 방법이다. 

  1. 첫번째 벡터를 먼저 normalization을 한다. 
  2. 두번째 벡터를 첫번째 벡터에 projection한 벡터에서 뺀다.
  3. 벡터의 개수만큼 2번을 반복한다. 




# 7. Eigenvectors & Eigenvalues

- **고유벡터 (Eigenvectors) & 고유값 (Eigenvalues)**
  $$
  고유벡터란\space 0이\space 아닌\space R^{n*n}인 \space 벡터\space A가 \space있을\space 때,\\
  \space Ax = \lambda x를\\
  \space 만족하는\space 벡터\space x를 의미한다.(\lambda은\space 상수로 \space 고유값, eigenvalue라고\space 불린다.)
  $$
  즉, 입력이 x라고 할 때, 고유벡터는 **방향은 바뀌지 않고 크기만 바뀌는 벡터**를 의미한다.

  고유벡터는 계산이 효율적이라는 장점이 있다. 

  

  **EX)** 
  $$
  \begin{bmatrix}
  2&6\\5&3
  \end{bmatrix}
  \begin{bmatrix}
  1\\1
  \end{bmatrix}
  =8
  \begin{bmatrix}
  1\\1
  \end{bmatrix}
  $$
  좌변의 경우에는 2 * 1 + 6 * 1, 5 * 1 + 3 * 1로 총 6번의 연산이 필요하지만, Eigenvector와 Eigenvalues를 알고 있는 경우에는 2번의 연산만 하면 값을 구할 수 있다.  

  

- **고유벡터 (Eigenvectors) & 고유값 (Eigenvalues) 구하기**
  $$
  (A-\lambda I)x=0
  $$
  위 경우에서 Ax=0일 경우, x=0으로 무조건 해가 하나 존재하는 trivial solution, 즉 linearly independent하게 된다. 이때, A에서 람다*I를 빼주었을 때, 또 다른 해가 존재하는 경우 **선형종속**(Linearly dependent)하게 만들어 주는 것이 고유벡터, 고유값을 구하는 과정이다. 

  

# 8. 영공간 (Null Space) 

Ax=0을 만족시키는 x를 Null Space of A, Nul A라고 부른다. 

Ax=0을 달리 말하면, A의 각각의 row vector에 대해서 x는 모두와 직교 (orthogonal)하다는 것을 의미한다. 

**EX)**
$$
A = \begin{bmatrix}
1&2\\3&4\\5&6
\end{bmatrix}
일\space 때, x는 \begin{bmatrix}1&2\end{bmatrix},\begin{bmatrix}3&4\end{bmatrix},\begin{bmatrix}5&6\end{bmatrix} 모두와\space 직교한다고\space 볼\space 수\space 있다.
$$
 **EX1)**
$$
R^3인\space 공간에서\space A=\begin{bmatrix}1&2&3\end{bmatrix}으로\space 주어졌을\space 때,\\ NulA는\space 그람슈미츠를\space 이용하여\space 두개를\space 더\space 구할\space 수\space 있다.\space (3-1=2)
$$
<img src="../../../../AppData/Roaming/Typora/typora-user-images/image-20220104231002690.png" alt="image-20220104231002690" style="zoom:67%;" />

즉, Nul A는 두개를 더 구할 수 있으므로, **하나의 평면**을 나타내게 된다. 

**Rank theorem = 위의 예제에서 Rank 3 (Row of A + Nul of A)**



























