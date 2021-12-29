# Pillow (PIL) module 사용법

- **주요기능**

  PIL 이미지 작업을 위한 표준 절차를 제공한다.

  - 픽셀 단위의 조작
  - 마스킹 및 투명도 제어
  - 흐림, 윤곽 보정, 윤곽 검출 등의 이미지 필터
  - 선명하게, 밝기 보정, 명암 보정, 색 보정 등의 화상 조정
  - 이미지에 텍스트 추가
  - 기타 

1. 모듈 불러오기

   ```python
   from PIL import Image
   ```

2. Image.open

   이미지 불러오기 (이미지 출력)

   ```python
   img = Image.open("C:/Users/image.jpg")
   img.show()
   ```

3. 이미지 속성 정보 확인하기

   - img.filename : 이미지 파일 이름 (경로포함)
   - img.format : 이미지 포멧 (ex. JPEG)
   - img.size : 이미지 사이즈
   - img.mode : 색상 모드 (ex. RGB)
   - img.width : 이미지 너비
   - img.height : 이미지 높이

4. img.resize()

   ```python
   resize_img = img.resize((w,h))
   ```

5. img.crop()

   ```python
   crop_img = img.crop((100,100,200,200))	# 좌상단, 우하단좌표
   ```

6. img.rotate()

   ```python
   rotate_img = img.rotate(45)		# 좌로 45도 만큼 이미지 회전
   ```

7. img.transpose()

   ```python
   flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
   flip_img1 = img.transpose(Image.FLIP_TOP_BOTTOM)
   ```







