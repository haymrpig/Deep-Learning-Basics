# XML 파일 다루기

```python
import xml.etree.ElementTree as et
```



- **xml파일 parsing하기**

  ```python
  tree = et.parse('Path to xml file')
  ```

- tree 구조의 파일에서 원하는 값 읽어오기

  ![image](https://user-images.githubusercontent.com/71866756/152807659-44ccac13-b03c-4387-bc3b-f7ab9e8eefc6.png)

  ```python
  # bbox 예제이다. 
  # 트리 구조는 위 그림과 같다. 
  # 여기서 xmin, ymin, xmax, ymax를 읽어온다. 
  xmin = float(tree.find('./object/bndbox/xmin').text)
  ymin = float(tree.find('./object/bndbox/ymin').text)
  xmax = float(tree.find('./object/bndbox/xmax').text)
  ymax = float(tree.find('./object/bndbox/ymax').text)
  ```

  
