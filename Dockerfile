
# 파이썬 베이스 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 인증서 복사
# 인증서와 개인 키 복사 (레포지토리 내 certs 폴더로부터)
COPY certs/fullchain.pem /etc/letsencrypt/live/luvoost.co.kr/fullchain.pem
COPY certs/privkey.pem /etc/letsencrypt/live/luvoost.co.kr/privkey.pem

# 7777 포트 열기
EXPOSE 7777

# 애플리케이션 실행 (SSL 옵션 추가)
CMD gunicorn --certfile=/etc/letsencrypt/live/luvoost.co.kr/fullchain.pem --keyfile=/etc/letsencrypt/live/luvoost.co.kr/privkey.pem --bind 0.0.0.0:32000 app:app