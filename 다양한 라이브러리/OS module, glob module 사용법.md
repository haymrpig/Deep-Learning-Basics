# OS module / glob moduel 사용법

- **glob module**

  ```python
  from glob import glob
  glob('*.exe')			# 현재 경로상의 모든 .exe파일을 list로 반환
  glob(r'C:/U*')			# 현재 경로가 아닌 다른 경로도 탐색 가능, U로 시작하는
  						# 모든 파일 검색
  						
  						# r은 raw string으로 일반적인 string에서는 '/'를 escape
  						# 문자와 구분하기 위해서 /를 앞에 붙여야 하지만, raw 문자열
  						# 에서는 사용하지 않아도 된다.
  
  ```

1. **os.getcwd()**

   : 현재 경로를 구하기

2. **os.chdir(*path*)**

   : *path*를 현재 경로로 바꾸기

3. **os.listdir(*path*)**

   : *path*상에 파일과 디렉토리를 리스트로 반환

   ```python
   # 'exe'로 끝나는 파일 출력
   for x in os.listdir('c://Anaconda'):
       if x.endswith('exe'):
           print(x)
   ```

4. **os.rename(*origin*, *target*)**

   : *origin*명을 *target*명으로 변경

5. **os.path.join(*path1*, *path2*)**

   : *path1*,*path2* 경로 합치기

   ```python
   os.path.join("/Anaconda", "file.py")	# /Anaconda/file.py
   ```

   ```python
   list_path = ['C:','Users','file.py']
   folder_path = os.path.join(*list_path)	# C:/Users/file.py
   ```
   
6. **os.path.basename**

   : 경로명 *path*의 기본 이름을 반환한다. 

   ```python
   list_path = ['C:','Users','file.py']
   path = os.path.join(*list_path)
   base_path = os.path.basename(path)		# file.py
   ```

7. **os.mkdir / os.makedirs**

   : 전자는 하나의 폴더만을 생성할 수 있지만, 후자는 폴더 내부 폴더까지도 생성할 수 있다.

   ```python
   os.mkdir('./new_folder')
   os.makedirs('./new_folder/new_folder1')
   ```

9. **os.path.exists()**

   ```python
   os.path.exists('Users')		# 현재 실행되는 경로에 Users 파일이 존재하는지 확인
   							# True/False
   ```

9. **os.path.abspath()**

   : 해당 파일의 절대경로를 가져올 수 있다. 

   ```python
   os.path.abspath("./hong.txt")
   # C:/content/hong.txt
   ```

10. **os.path.isdir()**

    : dirtory이면 True 반환

11. **os.path.splitext()**

    : 확장자와 파일명 구분

    ```python
    fname = "catimage.jpg"
    splited_name = os.path.splitext(fname)
    print(splited_name[0], splited_name[1])	# "catimage", ".jpg"
    ```

12. **os.remove()**

    : file삭제



   

   