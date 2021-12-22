# OS module 사용법

1. os.getcwd()

   : 현재 경로를 구하기

2. os.chdir(*path*)

   : *path*를 현재 경로로 바꾸기

3. os.listdir(*path*)

   : *path*상에 파일과 디렉토리를 리스트로 반환

   ```python
   # 'exe'로 끝나는 파일 출력
   for x in os.listdir('c://Anaconda'):
       if x.endswith('exe'):
           print(x)
   ```

4. os.rename(*origin*, *target*)

   : *origin*명을 *target*명으로 변경

5. os.path.join(*path1*, *path2*)

   : *path1*,*path2* 경로 합치기

   ```python
   os.path.join("/Anaconda", "file.py")	# /Anaconda/file.py
   ```

   ```python
   list_path = ['C:','Users','file.py']
   folder_path = os.path.join(*list_path)	# C:/Users/file.py
   ```