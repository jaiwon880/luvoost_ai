# flask
from flask import Flask, request, jsonify
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import pandas as pd
import json
from flask import Response
# from reviewline import preprocess_review_data
import random
import numpy as np
from recommend import recommend_restaurants, recommend_optimal, recommend_optimal_cafe, recommend_optimal_movie
from recommend import recommend_cafe, recommend_optimal_park, recommend_theme, recommend_random_places


app = Flask(__name__)

# 모델 로드
model = tf.keras.models.load_model('halp.h5')

df = pd.read_csv('./레스토랑.csv', encoding='utf-8')
cafe_df = pd.read_csv('./카페.csv', encoding='utf-8')
movie_df = pd.read_csv('./영화관.csv', encoding='utf-8')
park_df = pd.read_csv('./산책.csv', encoding='utf-8')
theme_df = pd.read_csv('./테마.csv', encoding='utf-8')

# 이미 처리한 데이터를 기록할 세트
processed_data = []  # 리스트로 초기화
processed_data2 = []  # 리스트로 초기화

@app.route("/api/v1")
def hello():
    return "flask test!"

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    # 1. 클라이언트로부터 데이터 추출
    data = request.json  # 클라이언트로부터 받은 JSON 데이터를 추출
    print(data)  # 추출한 데이터 출력 (디버깅 용도)

    # 사용자의 위치 및 선호 음식 추출
    latitude = float(data['user_latitude'])
    longitude = float(data['user_longitude'])
    preferred_food = data['food']
    budget = int(data['budget']) if 'budget' in data else None  # 예산이 설정되지 않았을 경우 None

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
    results = recommend_restaurants(user_preferences, preferred_food, (latitude, longitude), budget, n_recommendations=3)
    results_json = results.to_json(orient='records')
    results_data = json.loads(results_json)

    for location in results_data:
        total_expected_cost = 0  # 예상 소비 금액을 저장할 변수 초기화
        total_expected_cost += location['mean_price']
        if budget:  # budget이 None이 아닐 경우만 차감
            budget -= location['mean_price'] # 예산 비용 차감

        cafe_latitude = location['latitude']
        cafe_longitude = location['longitude']

        # 카페 추천 함수 호출
        cafe_results = recommend_cafe(user_preferences, "카페", (cafe_latitude, cafe_longitude), budget, n_recommendations=1)
        cafe_results_json = cafe_results.to_json(orient='records')
        total_expected_cost += cafe_results['mean_price'].iloc[0]
        if budget:  # budget이 None이 아닐 경우만 차감
            budget -= cafe_results['mean_price'].iloc[0]

        #테마
        theme_results = recommend_theme((cafe_results['latitude'].iloc[0], cafe_results['longitude'].iloc[0]), n_recommendations=3)
        theme_results_json = theme_results.to_json(orient='records')
        total_expected_cost += float(theme_results['mean_price'].iloc[0])

        if budget:  # budget이 None이 아닐 경우만 차감
            budget -= float(theme_results['mean_price'].iloc[0])

        # 영화관 추천 함수 호출
        movie_results = recommend_optimal_movie((theme_results['latitude'].iloc[0], theme_results['longitude'].iloc[0]), n_recommendations=1)
        movie_results_json = movie_results.to_json(orient='records')
        total_expected_cost += movie_results['mean_price'].iloc[0]

        if budget:  # budget이 None이 아닐 경우만 차감
            budget -= movie_results['mean_price'].iloc[0]

        # 영화관 좌표를 사용하여 공원 추천 함수 호출
        park_results = recommend_optimal_park((movie_results['latitude'].iloc[0], movie_results['longitude'].iloc[0]), n_recommendations=3)
        park_results_json = park_results.to_json(orient='records')

        # JSON 형태의 문자열을 JSON 객체 리스트로 변환
        theme_results = json.loads(theme_results_json)
        park_results = json.loads(park_results_json)

            # 첫 번째 값을 출력
        for theme in theme_results:
            if theme not in processed_data:
                processed_data.append(theme)
                break

                    # 첫 번째 값을 출력
        for park in park_results:
            if park not in processed_data2:
                processed_data2.append(park)
                break       

        # 추천 결과를 구성하고 리스트에 추가
        recommendation = {
            'restaurant_prediction': [location] + json.loads(cafe_results_json) + [theme]
              + json.loads(movie_results_json) + [park],
            'expected_total_cost': total_expected_cost  # 예상 소비 금액 추가
        }     
        recommendations.append(recommendation)

    # 4. 최종 결과 JSON 구성
    # result = {
    #     'recommendations': recommendations
    # }
    # 5. 결과 응답 반환
    return Response(response=json.dumps(recommendations), status=200, mimetype="application/json")

@app.route('/api/v1/optimal', methods=['GET', 'POST'])
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

@app.route('/api/v1/random', methods=['GET', 'POST'])
def predict_random():
    data = request.json
    selected_region = str(data['selected_region'])  # 클라이언트가 선택한 지역 받아오기

    num_restaurant_recommendations = 2
    recommended_restaurants = recommend_random_places(df,
        num_restaurant_recommendations,
        '맛', 2.8,  # 친절도 2.4 이상
        '분위기', 2.5,  # 분위기 2.5 이상
        selected_region
        )
    restaurants_results_json = recommended_restaurants.to_json(orient='records')

    # 랜덤으로 카페 두 개 추천 (분위기 2.5 이상, 선택한 지역 포함)
    num_cafe_recommendations = 2
    recommended_cafes = recommend_random_places(cafe_df,
        num_cafe_recommendations,
        '친절도', 2.4,  # 친절도 2.4 이상
        '분위기', 2.5,  # 분위기 2.5 이상
        selected_region
        )
    cafes_results_json = recommended_cafes.to_json(orient='records')

    #영화관 추천
    movie_location = json.loads(cafes_results_json)[0]
    movie_latitude = movie_location['latitude']
    movie_longitude = movie_location['longitude']

    movie_results = recommend_optimal_movie((movie_latitude, movie_longitude), n_recommendations=1)
    movie_results_json = movie_results.to_json(orient='records')

    # 랜덤으로 테마 두 개 추천 (분위기 2.5 이상, 선택한 지역 포함)
    num_theme_recommendations = 2
    recommended_theme = recommend_random_places(theme_df,
        num_theme_recommendations,
        '재미', 2.3,  # 친절도 2.4 이상
        '추천', 2.6,  # 분위기 2.5 이상
        selected_region
        )
    theme_results_json = recommended_theme.to_json(orient='records')


    #산책로 추천
    park_location = json.loads(theme_results_json)[0]
    park_latitude = park_location['latitude']
    park_longitude = park_location['longitude']

    # n_recommendations에 원하는 값(2)을 넣으세요
    park_results = recommend_optimal_park((park_latitude, park_longitude), n_recommendations=2)
    park_results_json = park_results.to_json(orient='records')

    # 추천 결과
    recommendation_result = {
        "random": json.loads(restaurants_results_json) + json.loads(cafes_results_json) + json.loads(movie_results_json)
          + json.loads(theme_results_json) + json.loads(park_results_json)
    }

    return jsonify(recommendation_result)

if __name__ == '__main__':
    app.run(debug=False)
