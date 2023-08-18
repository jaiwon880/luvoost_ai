
# 파이썬 베이스 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 7777 포트 열기
EXPOSE 32000

# 애플리케이션 실행
CMD gunicorn --bind 0.0.0.0:32000 app:app
