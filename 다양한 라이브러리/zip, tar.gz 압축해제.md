# zip, tar.gz 압축해제

```python
import tarfile
import zipfile
```



- **zip 압축해제**

  ```python
  ap = zipfile.open('filename')
  ap.extractall('dest path')
  ap.close()
  
  ##################
  
  zipfile.ZipFile("filename").extractall('dest path')
  ```

- **tar.gz 압축해제**

  ```python
  ap = tarfile.open('filename')
  ap.extractall('dest path')
  ap.close()
  ```

  