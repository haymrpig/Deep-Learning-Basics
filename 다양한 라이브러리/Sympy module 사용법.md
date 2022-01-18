# Sympy module

1. sympy module

   ```python
   import sympy
   ```

2. sympy.symbols()

   ```python
   import sympy
   
   x = sympy.symbols("x")
   # 대수 기호를 사용하기 위한 symbols 메소드
   # 일반적으로 변수명과 대수기호는 같은 것을 사용한다. 
   
   x, y = sympy.symbols("x y")
   # 두 개 이상의 대수기호 사용법
   
   x = sympy.symbols('x1:11')
   # x1, x2, ..., x10을 튜플로 생성
   
   x = sympy.symbols('x', integet = True)
   # 대수 기호의 종류를 지정할 수도 있다. 
   # integer(정수), real(실수), complex(복소수), positive(양수)
   f = sympy.symbols('f', cls=Function)
   # 함수 기호로 정의
   ```

3. sympy.poly()

   ```python
   import sympy
   
   x, y = sympy.symbols('x y')
   func = sympy.poly(x**2 + 2*y +3)
   # 다항식
   print( func.subs({x:2, y:3}) )
   # x=2, y=3을 넣어 방정식 풀이
   ```

4. sympy.solve()

   ```python
   import sympy
   
   x = sympy.symbols('x')
   func = sympy.poly(x**2 + 2*x +3)
   sympy.solve(func, x)
   # func=0으로 x의 해를 구함
   ```

   