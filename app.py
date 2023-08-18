# flask
from flask import Flask, request, jsonify
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import pandas as pd
import json
from flask import Response
# from reviewline import preprocess_review_data
import numpy as np
from flask_sslify import SSLify
from recommend import recommend_restaurants, recommend_optimal, recommend_optimal_cafe, recommend_optimal_movie
from recommend import recommend_cafe, recommend_optimal_park, recommend_theme


app = Flask(__name__)
sslify = SSLify(app)

# 모델 로드
model = tf.keras.models.load_model('halp.h5')

df = pd.read_csv('./레스토랑.csv', encoding='utf-8')
cafe_df = pd.read_csv('./카페.csv', encoding='utf-8')
movie_df = pd.read_csv('./영화관.csv', encoding='utf-8')

@app.route("/")
def hello():
    return "flask test page"

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    # 1. 클라이언트로부터 데이터 추출
    data = request.json  # 클라이언트로부터 받은 JSON 데이터를 추출
    print(data)  # 추출한 데이터 출력 (디버깅 용도)

    # 사용자의 위치 및 선호 음식 추출
    latitude = float(data['user_latitude'])
    longitude = float(data['user_longitude'])
    preferred_food = data['food']

    # 2. 사용자의 음식점 선호도 추출
    user_preferences = {
        '맛': data['taste'],
        '서비스': data['service'],
        '분위기': data['ambiance'],
        '매장상태': data['storeCondition'],
        '양': data['quantity'],
        '친절도': data['kindness']
    }

    # 3. 음식점 및 카페 추천 및 결과 구성
    recommendations = []

    # 음식점 추천 결과를 받아서 세부 추천 수행
    results = recommend_restaurants(user_preferences, preferred_food, (latitude, longitude), n_recommendations=3)
    results_json = results.to_json(orient='records')
    results_data = json.loads(results_json)

    for location in results_data:
        cafe_latitude = location['latitude']
        cafe_longitude = location['longitude']

        # 카페 추천 함수 호출
        cafe_results = recommend_cafe(user_preferences, "카페", (cafe_latitude, cafe_longitude), n_recommendations=1)
        cafe_results_json = cafe_results.to_json(orient='records')

        #테마
        theme_results = recommend_theme((cafe_results['latitude'].iloc[0], cafe_results['longitude'].iloc[0]), n_recommendations=1)
        theme_results_json = theme_results.to_json(orient='records')

        # 영화관 추천 함수 호출
        movie_results = recommend_optimal_movie((theme_results['latitude'].iloc[0], theme_results['longitude'].iloc[0]), n_recommendations=1)
        movie_results_json = movie_results.to_json(orient='records')

        # 영화관 좌표를 사용하여 공원 추천 함수 호출
        park_results = recommend_optimal_park((movie_results['latitude'].iloc[0], movie_results['longitude'].iloc[0]), n_recommendations=1)
        park_results_json = park_results.to_json(orient='records')

        # 추천 결과를 구성하고 리스트에 추가
        recommendation = {
            'latitude': str(latitude),
            'longitude': str(longitude),
            'restaurant_prediction': [location] + json.loads(cafe_results_json) + json.loads(theme_results_json)
              + json.loads(movie_results_json) + json.loads(park_results_json)
    
        }
        # print(cafe_results['latitude'])
        # print(cafe_results['longitude'])
        # print(theme_results['latitude'])
        # print(theme_results['longitude'])
        # print(movie_results['latitude'])
        # print(movie_results['longitude'])
        # print("=============================")

        recommendations.append(recommendation)

    # 4. 최종 결과 JSON 구성
    result = {
        'recommendations': recommendations
    }
    # 5. 결과 응답 반환
    return Response(response=json.dumps(result), status=200, mimetype="application/json")

# 경로 /predict/random 에서 GET 요청이 들어왔을 때 처리
@app.route('/predict/optimal', methods=['GET', 'POST'])
def predict_optimal():
    # 1. 클라이언트로부터 데이터 추출
    data = request.json  # 데이터가 리스트의 첫 번째 요소로 전달됨
    print(data)

    latitude = float(data['user_latitude'])
    longitude = float(data['user_longitude'])

    # 음식점 추천 함수 호출
    results = recommend_optimal((latitude, longitude), n_recommendations=1)
    results_json = results.to_json(orient='records')

    # 선택된 음식점의 위치 정보 추출
    recommended_location = json.loads(results_json)[0]
    cafe_latitude = recommended_location['latitude']
    cafe_longitude = recommended_location['longitude']
    
    # 카페 추천 함수 호출
    cafe_results = recommend_optimal_cafe((cafe_latitude, cafe_longitude), n_recommendations=1)
    cafe_results_json = cafe_results.to_json(orient='records')

    #영화관 추천
    movie_location = json.loads(cafe_results_json)[0]
    movie_latitude = movie_location['latitude']
    movie_longitude = movie_location['longitude']

    movie_results = recommend_optimal_movie((movie_latitude, movie_longitude), n_recommendations=1)
    movie_results_json = movie_results.to_json(orient='records')
    
    # 결과 데이터 생성 및 반환
    result = {
        'latitude': str(latitude),
        'longitude': str(longitude),
        'combined': json.loads(results_json) + json.loads(cafe_results_json) + json.loads(movie_results_json)
    }
    
    return jsonify(result)
# @app.route('/predict/random', methods=['GET'])
# def predict_random():

if __name__ == '__main__':
    app.run(debug=False)
