# Visdom module

- **설치**

  pip install visdom

- **visdom 서버 켜기**

  python -m visdom.server로 나오는 http://localhost:----을 새로운 창으로 띄워서 사용한다.

- **visdom 사용도**

  visdom을 이용해 loss를 그래프로 실시간으로 plot이 가능하며, 원하는 이미지 확인 또한 가능하다. 이 외에도 다양한 기능을 제공하고 있다. 

- **visdom 사용법**

  - visdom 모듈 이용하기

    ```python
    import visdom
    vis = visdom.Visdom()   
    # visdom 서버를 먼저 키고 진행
    ```

  - text 띄우기

    ```python
    vis.text("example_text", env="main")
    # env는 없어도 되지만, 모든 창을 한꺼번에 끌 때 유용하다. 
    ```

  - image 띄우기

    ```python
    vis.images(torch.Tensor(3,28,28))	# 이미지 하나 띄우기
    vis.images(torch.Tensor(3,3,28,28))	# 이미지 여러개 띄우기
    ```

  - Line Plot

    ```python
    Y_data = torch.randn(5)
    plt = vis.line(Y=Y_data)			
    # x축이 없는 경우 0~1사이로 지정됨
    
    X_data = torh.Tensor([1,2,3,4,5])
    plt = vis.line(Y=Y_data, X=X_data)	
    # 주어진 x축에 맞게 plot
    
    Y_append = torch.randn(1)
    X_append = torch.Tensor([6])
    vis.line(Y=Y_append, X=X_append, win=plt, update='append')
    # 기존 그래프에 값 추가하는 방법
    
    num = torch.Tensor(list(range(0,10)))
    num = num.view(-1,1)
    num = torch.cat((num, num), dim=1)
    plt = vis.line(Y=torch.randn(10,2), X=num)
    # 두개의 그래프를 하나의 figure에 plot하기
    # 이 때, Y와 X의 shape은 같아야 한다. 
    # Y의 1열, 2열 두개의 그래프가 생성되며, X의 1열, 2열이 각각의 x축을 나타낸다.
    ```

  - Line info

    ```python
    plt = vis.line(Y=Y_data, X=X_data, opts=dict(title='Test', showlegend=True))
    # title은 figure의 제목, showlegend는 범례이다. 
    
    plt = vis.line(Y=Y_data, X=X_data, opts=dict(title='Test', legend=['1번'], showlegend=True))
    # 범례의 이름을 legend로 붙여줄 수 있다. 
    
    plt = vis.line(Y=torch.randn(10,2), X=num, opts=dict(title='Test', legend=['1번','2번'], showlegend=True))
    # 그래프가 두개인 경우 범례도 두개를 넣어줄 수 있다. 
    ```

  - close window

    ```python
    vis.close(env="main")
    ```

    

    

    