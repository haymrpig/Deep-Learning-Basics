# 목차

1. [행렬](#1-행렬)
2. [선형결합](#2-선형결합)
   - Span
   - Sum of outer product
   - Subspace
   - Basis( 기저 )
3. [선형독립/종속 (Linearly independent/dependent)](#3-선형독립종속)
4. [선형변환](#4-선형변환)
   - 용어정리
   - ONTO( 전사 ) / ONE-TO-ONE( 일대일 )
5. [Least Squares](#5-least-squares)
   - 용어정리
   - 정규방정식
6. [Projection](#6-projection)
   - Orthogonal basis
   - Gram-Schmidt Orthogonalization
7. [고유벡터 (Eigenvectors) & 고유값 (Eigenvalues)](#7-eigenvectors--eigenvalues)
8. [영공간 (Null Space)](#8-영공간-null-space)
9. [대각화 (Diagonalization)](#9-대각화-diagonalization)
10. [고유값 분해 (Eigendecomposition)](#10-고유값-분해-eigendecomposition)
11. [특이값 분해 (SVD, Singular Value Decomposition)](#11-특이값-분해-svd-singular-value-decomposition)

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

`선형결합` : 벡터의 상수값을 곱해 결합하는 것  
![image](https://user-images.githubusercontent.com/71866756/151376960-59583ff1-d92c-49d5-8b5c-59e58fa7a5e8.png)  

![image](https://user-images.githubusercontent.com/71866756/145265004-c357fde6-0411-4d0e-82f9-f0326bd3dbef.png)

- **Span**

  `span` : 가능한 모든 선형 결합의 집합 (공간 자체를 의미한다고 생각하면 쉽다)

  ![image](https://user-images.githubusercontent.com/71866756/145265083-2a7036bd-baa1-46ec-8f6f-c4fa90f2f5bc.png)

  

- **Sum of (Rank-1) outer product**

  ![image](https://user-images.githubusercontent.com/71866756/145265135-f33952ed-43d2-4407-9b33-db19a6d9b106.png)


  복잡한 행렬곱도 행렬의 분해를 통해 간단하게 계산을 할 수 있다. 

  - 공분산, style transfer 등 다양하게 쓰인다. 

- **Subspace**

  `닫혀있다`의 개념은 모든 부분집합끼리 또는 부분집합의 연산이 부분집합이 되는 것을 의미한다.

  >예를 들어, S={2,4,6, ... }으로 2의 배수를 원소로 갖는 S의 경우는 곱셈에 닫혀있다고 할 수 있다( 어떤 부분집합의 곱이 S의 부분집합에 속하므로 ).

  `subspace`란 **span이랑 비슷하지만 닫혀있다는 개념이 추가**된 것으로, 선형 결합에 닫혀있는 S의 부분집합을 의미한다. 

  선형결합에 닫혀있다라는 것은 어떠한 space S에서 임의의 두 부분집합을 뽑아내었을 때, 이 두 벡터의 선형결합이 S의 부분집합에 속해있을 때를 의미한다.

  **Span은 subspace(선형결합)에 대해서 닫혀있다고 할 수 있다.**

  - **Dimension of Subspace**

    기저 벡터의 개수가 dimension (Rank)이다.

    > 여기서 Rank가 중요한 이유는 학습에 있어서 여러가지의 feature가 존재하고 각각의 feature를 하나의 벡터라고 해보자. 
    >
    > 만약 v1이 3*v2라고 한다면 이 두 벡터는 겹친다고 할 수 있어 feature로서의 역할을 하지 못한다. 
    >
    > 이 둘 중 하나의 벡터는 쓸모가 없어지는 벡터 (선형 종속)가 되기 때문에 이 때 basis 즉, rank를 파악하는 것이 중요하다.  

- **Basis of Subspace**

  기저벡터라고 하며, 기저벡터는 주어진 subspace를 fully span하며 선형 독립이여야 한다.

   즉, full span이라는 의미는 2차원을 예로 들면 2개의 벡터를 사용든, 3개의 벡터를 사용하든 한 평면을 모두 표현가능해야 한다는 의미이다.   
  ![image](https://user-images.githubusercontent.com/71866756/152662994-bbcb488a-0f38-4994-8837-3c859fb68dc5.png)  
  
  
  - **기저 벡터는 unique하지 않다!**
  
    즉, 2차원을 예로 들어 한 평면을 표현하기 위한 벡터는 다양한 조합이 가능하다. 이 조합들이 모두 기저벡터가 될 수 있어 unique하지 않다.
  
    ( 단, 이 벡터의 조합은 선형독립이며 fully span해야 한다. )
  
    ![image](https://user-images.githubusercontent.com/71866756/152663007-0520fae2-eb40-4f92-a365-b0fa68db4d06.png)
  
    >위 그림처럼 어떠한 공간이 주어지고, 특정한 점이 주어졌을 때, 우리는 그 공간을 fully span하는 기저벡터를 여러개 찾을 수 있다. 
    >
    >그리고 이러한 기저벡터들로 목표하는 점에 도달하기 위해서는 기저벡터의 선형결합으로 표현할 수 있으며, coefficient는 기저벡터의 값에 따라 달라지게 된다. 
    >
    >따라서, 기저벡터를 바꿈으로 coefficient의 값도 달라지며, 이러한 아이디어를 single value decompsition 등에 사용할 수 있게 된다. 
  
  - **Standard basis vector**
  
    가장 간단한 기저 벡터로 3차원의 경우, [1 0 0], [0 1 0], [0 0 1]이라고 할 수 있다. 
  
  - **Column space**
  
    열벡터가 subspace인 경우  
    ![image](https://user-images.githubusercontent.com/71866756/152663014-08cacad9-3139-414c-bdc5-47e20cc88ac3.png)  
    

# 3. 선형독립/종속

- **선형독립 (Linearly independent)**

  Ax=b에서 b가 0벡터일 때, 만약 x=0이 유일한 해라면 벡터들의 집합 A는 **선형독립**이다.

  (즉, A의 임의의 벡터는 A에 포함된 다른 벡터들의 선형결합으로는 표현할 수 없다)

  (**다른 해 (non trivial solution)가 존재**하는 경우 **선형종속**이다)

  

- **선형종속 (Linearly dependent)**

  Span을 생각해보았을 때, 벡터가 추가될 때 기존 Span에 포함이 된다면 선형종속이라고 한다. 

  즉, 기존의 벡터들로 충분히 해를 구할 수 있었을 때, 새로운 벡터가 들어와도 그 벡터가 해를 구하는데 영향을 미치지 않아도 되는 경우 선형종속이다.  새로운 벡터는 기존의 벡터들로 표현이 된다. 

  **!!! 3차원 공간에서 4개의 벡터가 주어지면 무조건 선형종속이 될 수밖에 없다 !!!**
  
  ![image](https://user-images.githubusercontent.com/71866756/151377055-b7e24127-7e33-46cf-a846-d1b84da4c844.png)
  
  쉽게 말해서, 하나의 span을 구성하는 서로 다른 벡터들 중에서, 임의의 하나의 벡터가 다른 여러 벡터의 선형결합으로 나타낼 수 없는 경우를 의미한다. 
  
  예를 들어, 세 벡터가 있다고 했을 때, 두 벡터로 하나의 벡터를 표현할 수 있다고 하자. 
  
  그러면 기존의 세 벡터로 표현할 수 있는 공간은 사실상 두 벡터로 표현할 수 있다는 의미가 된다. 
  
  (어차피 세 번째 벡터도 두 벡터의 선형결합이기 때문에)
  
  위 그림처럼 벡터는 세개지만, 2차원 공간밖에 표현을 하지 못하는 것이다. 
  
  이 같은 경우를 **선형 종속**이라고 한다. 

- **선형종속과 선형독립 구분방법**  
  ![image](https://user-images.githubusercontent.com/71866756/151377124-b6a601f5-a39b-42a7-b8a8-8a9fae27e874.png)  
  

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

    여기서 a1과 a2를 학습할 데이터의 key값들이라고 생각하고, dest는 출력 y값이라고 생각하자.

    만약 a1과 a2가 비슷한 경향을 보이게 된다면 a1과 a2에 특정 값을 곱하여 dest값이 되게 하기 위해서는 a1에는 음수방향으로 꽤 큰 값을 곱해줘야 하며, a2역시 양수방향으로 큰 값을 곱해줘야 한다. (dest에서 긋는 선분과 a2는 평행해야 한다.)

    이 때, weight의 절댓값이 매우 커질 수가 있기 때문에 좋지 않은 결과가 나올 가능성이 높다는 것이다. 

    ![image](https://user-images.githubusercontent.com/71866756/145265807-500104d9-153f-4bf9-aed6-5dc4168d5ca5.png)
    
    이 때 regularization기법을 사용한다. 
    
    특정 값을 더해줌으로써, 두 벡터의 경향을 다르게 만들어 주는 것이다. 
    
    ![image](https://user-images.githubusercontent.com/71866756/145265838-5346f35e-4472-404e-98d0-8dbbc1ed7a94.png)
    
    

- **Gram-Schmidt Orthogonalization**

  orthogonal basis를 만드는 방법이다. 

  1. 첫번째 벡터를 먼저 normalization을 한다. 
  2. 두번째 벡터를 첫번째 벡터에 projection한 벡터에서 뺀다.
  3. 벡터의 개수만큼 2번을 반복한다. 




# 7. Eigenvectors & Eigenvalues

- **고유벡터 (Eigenvectors) & 고유값 (Eigenvalues)**  
  ![image](https://user-images.githubusercontent.com/71866756/148072376-9704329d-6728-43f3-a3d7-37b8ea524bf4.png)  
  즉, 입력이 x라고 할 때, 고유벡터는 **방향은 바뀌지 않고 크기만 바뀌는 벡터**를 의미한다.

  고유벡터는 계산이 효율적이라는 장점이 있다. 

  

  **EX)**   
  ![image](https://user-images.githubusercontent.com/71866756/148072452-27d9bf79-758e-4644-9ce4-61259c70a166.png)  
  좌변의 경우에는 2 * 1 + 6 * 1, 5 * 1 + 3 * 1로 총 6번의 연산이 필요하지만, Eigenvector와 Eigenvalues를 알고 있는 경우에는 2번의 연산만 하면 값을 구할 수 있다.  

  

- **고유벡터 (Eigenvectors) & 고유값 (Eigenvalues) 구하기**  
  ![image](https://user-images.githubusercontent.com/71866756/148072530-ff96d9e8-b821-41b6-a400-26e291ef7966.png)  
  위 경우에서 Ax=0일 경우, x=0으로 무조건 해가 하나 존재하는 trivial solution, 즉 linearly independent하게 된다. 이때, A에서 람다*I를 빼주었을 때, 또 다른 해가 존재하는 경우 **선형종속**(Linearly dependent)하게 만들어 주는 것이 고유벡터, 고유값을 구하는 과정이다. 

  
  
- **고유값 구하는 식**

  우선 역행렬에 대해서 생각해보자. 역행렬의 경우 정방행렬에서만 정의될 수 있으며 만약 정방행렬의 역행렬이 존재하지 않는 경우  
  ![image](https://user-images.githubusercontent.com/71866756/148206190-c5cfd6ff-e800-4845-810d-76ab567b03ce.png)  

  이런 식으로 나타낼 수 있으며, 역행렬이 존재하지 않으면 정방행렬의 column은 linearly dependent하다고 할 수 있다. 즉, 고유값을 구하기 위해서는 dependent한 경우를 찾는 것이고, 그것은 위의 식과 일치한다고 할 수 있다. 

  따라서 위의 식을 풀면 고유값 (lambda)의 값을 구할 수 있다. 

  **EX)**  
  ![image](https://user-images.githubusercontent.com/71866756/148206248-02331c9f-becd-4490-be0a-5b3b611c8f0d.png)  
  
- **Eigenspace**

  eigenvalue값에 대응되는 eigenvector이 존재하는 공간을 eigenspace라고 한다.   
  ![image](https://user-images.githubusercontent.com/71866756/148206317-8fa2f5bc-d399-4d06-abb6-d80245ce9dd1.png)  
  위 식을 만족한다고 할 수 있는데, 이 때, eigenspace상의 어떠한 벡터를 입력 x로 넣어도 eigenspace를 벗어나는 vector는 나오지 않는다.

# 8. 영공간 (Null Space) 

Ax=0을 만족시키는 x를 Null Space of A, Nul A라고 부른다. 

Ax=0을 달리 말하면, A의 각각의 row vector에 대해서 x는 모두와 직교 (orthogonal)하다는 것을 의미한다. 

**EX)**  
![image](https://user-images.githubusercontent.com/71866756/148072568-ad394202-7367-4657-83f4-8d5861bb2281.png)  
 **EX1)**  
![image](https://user-images.githubusercontent.com/71866756/148072616-858eab9c-03f2-4ac6-8e55-e0d815037525.png)  

즉, Nul A는 두개를 더 구할 수 있으므로, **하나의 평면**을 나타내게 된다. 

**Rank theorem = 위의 예제에서 Rank 3 (Row of A + Nul of A)**



# 9. 대각화 (Diagonalization)

대각행렬이란 대각성분을 제외한 모든 성분이 0인 행렬을 의미한다. 일반 행렬에서 대각행렬로 만드는 것이 대각화 (diagonalization)이다. 

- **대각화 방법과 의미**

  행렬 A를 대각행렬 D로 만드는 방법으로 아래 식과 같다. 

  ( 하지만 모든 행렬이 대각화가 가능한 것은 아니다. 아래 식을 만드는 V행렬을 찾을 수 있는 경우, V행렬이 역행렬이 존재할 경우 가능하다. )  
  ![image](https://user-images.githubusercontent.com/71866756/148224480-4769f5f0-9e35-4eb8-9ce5-bb9165d89278.png)  
  이러한 대각화가 가능한 행렬 A를 diagonalizable한 행렬이라 한다. 

  'diagonalizable'은 결국,  V가 3개의 linearly independent eigenvectors를 가져야 한다와 동일한 의미로 해석될 수 있다. ( 아래 식으로 증명 )  
  ![image](https://user-images.githubusercontent.com/71866756/148224662-23646444-dc1d-4cd5-94da-334ac88d1d43.png)  
  => 여기서 V가 역행렬을 가지려면 V의 column들이 linearly independent해야 하기 때문이다. 



# 10. 고유값 분해 (Eigendecomposition)

eigendecomposition이란 대각화된 행렬을 다시 원래대로 복원하는 분해하는 과정이라고 생각할 수 있다.   
![image](https://user-images.githubusercontent.com/71866756/148224726-c0a97759-4633-497a-bc8c-46d382fdcdfe.png)  
여기서 우측에 있는 식이 고유값 분해이다. diagonalizable 한 A행렬은 eigendecomposition을 가지고 있다고 할 수 있다. 

V와 V^-1는 역관계이며, D의 경우는 diagonal matrix여야 한다. 

- **고유값 분해의 필요성**

  이전에 말했듯이 고유값을 이용하면 계산을 효율적으로 할 수 있다는 장점이 있었다. 하지만 만약 어떤 행렬이 고유벡터가 아니라면 복잡한 계산을 해야한다. 이 때 고유값 분해를 이용할 수 있다. 고유값을 가지지 않는 행렬을 아래처럼 선형결합 형태로 만들 수가 있다.   
  ![image](https://user-images.githubusercontent.com/71866756/148224768-78b6f99e-7b04-4119-a70b-9ea26ffca3e6.png)  
   선형결합인 경우, 합친 후 matrix곱을 하나, matrix곱을 먼저하고 합치나 동일하다. 

  즉, 쉽게 예로 들면  
  ![image](https://user-images.githubusercontent.com/71866756/148224809-1560fb05-02a4-4288-b3ae-ea8fd5ad81f1.png)  
  이런식으로 만약 x의 고유값이 존재하지 않을 경우 계산이 복잡해지지만, x, 즉 [[3],[4]]를 두개의 선형결합으로 나타낸 후, 각각의 eigenvalue를 구하면 계산을 쉽게 할 수 있다는 의미이다.  

  간단하게 그림으로 표현하면 아래와 같다.   

  ![image](https://user-images.githubusercontent.com/71866756/148224869-8faca6b1-f63c-4f48-9612-281664412215.png)  
  - **A^k 구하기**  

  ![image](https://user-images.githubusercontent.com/71866756/148224918-194f9c6a-b3d6-4550-a273-65761fb42587.png)  

  

# 11. 특이값 분해 (SVD, Singular Value Decomposition)

특이값 분해는 rectangular matrix, 즉,   
![image](https://user-images.githubusercontent.com/71866756/148512534-a79020ba-8474-495e-8ed9-af9460e4a242.png)  
의 경우에 사용된다.   
![image](https://user-images.githubusercontent.com/71866756/148512561-6a92b99b-edb2-4246-8fbd-f224b53f2ff2.png)  
위 식으로 분해하는 것을 특이값 분해라고 한다. 

여기서 중요한 것은, U와 V는 모두 orthogonal한 vector를 가지며,  시그마는 diagonal matrix여야 한다.

( U의 column vector들이 orthonomal, V^T의 row vector들이 orthonomal )   

![image](https://user-images.githubusercontent.com/71866756/148512597-8ae540be-5a93-4acc-ae8d-f3f4e9902ff3.png)  

=> U : mxm, V^T : nxn

위 그림에서 아래 빨간색 사각형은 SVD의 reduce form이라고 하고, 위의 식과 정확히 일치한다. 



- **orthonomal 한 특정 (U, V)찾기**

  orthonomal한 U와 V는 무수히 많다. 그람슈미츠 방법을 사용할 때를 생각해보면 첫번째 벡터와, 그에 orthogonal한게 만든 두번째 벡터를 구할 수 있고, 이 둘의 순서를 바꾼다고 해도 orthonomal한 벡터를 찾을 수 있다. 즉, 영역에 대해 span하는 orthonomal한 벡터의 조합은 여러개가 나올 수 있다는 뜻이다. U,V도 마찬가지로 여러개가 나올 수 있지만, 여기서 특정한 U,V의 조합을 찾으려고 한다. 

  이 때 아래 수식을 이용한다.   
  ![image](https://user-images.githubusercontent.com/71866756/148512648-40158b75-e256-4247-af4f-c0107c2bd5b4.png)  
  이렇게 u와 v를 jointly해서 특정한 orthonomal한 vector들을 찾을 수가 있다. 

  즉, 아래 식과 같다고 할 수 있다.   
  ![image](https://user-images.githubusercontent.com/71866756/148512670-cb09cee4-e5c6-423d-bba7-c06ebd1d3d1a.png)  
  또한, V는 orthonormal한 vector들로 이루어져 있고, nxn이므로 V^-1=V^T이다.  
  ![image](https://user-images.githubusercontent.com/71866756/148512699-4dded537-4cbc-495c-8b1a-2adaef7d51aa.png)  

- **SVD 구하기**

  SVD는 구하는 알고리즘이 딱히 개별적으로 존재하지 않기 때문에, eigendecomposition과 같은 방식으로 진행한다.   
  ![image](https://user-images.githubusercontent.com/71866756/148532837-84e48fb6-5104-4c80-8165-0b8267118177.png)  

  - 위 조건에 앞서 우선적으로   
    ![image](https://user-images.githubusercontent.com/71866756/148532891-7adef1ea-f45e-4a4c-9636-6dfd65175020.png)  

  

- **Symmetric matrices의 Spectral Theorem**  
  ![image](https://user-images.githubusercontent.com/71866756/148532945-00b46625-f158-4d03-89c4-a2a41f67ce09.png)  

  - S has n real eigenvalues, counting multiplicities
  - The dimension of the eigenspace for each eigenvalue equals the multiplicity of lambda as a root of the characteristic equation
  - the eigenspaces are mutually orthogonal. that is, eigenvectors corresponding to different eigenvalues are orthogonal
  - to sum up, A is orthogonally diagonalizable



- **Positive Definite Matrices**

  직관적으로 생각해보면, 2차원 포물선이 항상 0보다 큰 포물선이 있을 것이다. 그러한 것처럼 matrix중에도 어떤 값을 넣어도 양수인 matrix가 존재한다. 이를 positive definite matrix라고 한다. 

  positive definite matrix는 정방행렬이다. 

  

  - positive definite matrix의 조건  

  ![image](https://user-images.githubusercontent.com/71866756/148532980-5e6584bd-adff-4da5-ab46-d6670d87facb.png)  

  - positive semi-definite matrix의 조건  

  ![image](https://user-images.githubusercontent.com/71866756/148533009-03f74e70-9937-4793-90ee-cda5d5b9ffd4.png)  

  - Theroem  
    ![image](https://user-images.githubusercontent.com/71866756/148533031-2f5918ba-4d95-4816-a7fe-102c4cd545de.png)  



- **Symmetric + positive definite matrix**

  위 두가지 개념을 합친 matrix의 경우 아래와 같다.   

  ![image](https://user-images.githubusercontent.com/71866756/148533073-3a16fe7a-b568-4f11-b300-b6a1845a3b64.png)  

  - 위 개념에 대해 생각해보면 아래 식이 성립한다.  
    ![image](https://user-images.githubusercontent.com/71866756/148533104-aece762b-e203-4e15-906c-0b658febbf4a.png)  
    위 식에 따라서 SVD 구하기의 1,2번 조건이 성립하는 것을 알 수 있다. 



- **최종 정리**
  - R^mxn의 A가 주어졌을 때, SVD는 항상 존재한다.
  - R^nxn의 A가 주어졌을 때, eigendecomposition은 존재하지 않을 수도 있지만, SVD는 항상 존재한다. 
  - 정사각, 대칭 positive (semi-)definite 행렬 S이 주어지면, eigendecomposition은 항상 존재하며, 이는 SVD와 동일하다. 

