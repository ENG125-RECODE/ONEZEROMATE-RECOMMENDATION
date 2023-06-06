# 라이브러리 임포트
from flask import Flask, request, jsonify
import requests
import pandas as pd
from urllib import parse
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import pymysql
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# 추천을 위한 사전 변수들 선언
# 서버에서 credentials 수정

client_credentials_manager = SpotifyClientCredentials(client_id='', client_secret='')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# mysql 연결 - 현재 로컬 서버
# 또한 서버에서 수정
db = pymysql.connect(
    host='localhost',
    user='root',
    password='1234',
    database='FOOD_DATA'
)
cursor = db.cursor()

# DB 연결
db_connection_str = 'mysql+pymysql://root:1234@127.0.0.1/FOOD_DATA'
db_connection = create_engine(db_connection_str)
conn = db_connection.connect()

# seed 장르로 넣을 수 있는 것의 교집합 구하기 위한 변수 미리 선언
seeds = sp.recommendation_genre_seeds()['genres']

app = Flask(__name__)

@app.route('/')
def running():
    return 'server is running.'

@app.route('/api/v1/foods/user-profile', methods=['POST'])
def user_check():
    args = request.json
    uuid = args['uuid']

    # foodName -> index number로 변환할 딕셔너리 로드 및 선언
    foodProfile = pd.read_sql_table('food', conn)
    foodName = list(foodProfile['name'])
    idx = list(foodProfile['id'])
    foodName_idx_dict = dict(zip(foodName, idx))

    # change foodName into foodId
    foodId = foodName_idx_dict[args['foodName']]
    moodId = args['moodId']

    # 여기서 uuid 입력값 검증 - DB에 존재하는지 확인 필요
    sql = "SELECT id FROM user WHERE uuid = %s"
    cursor.execute(sql, (uuid))
    result = cursor.fetchall()
    userId = result[0][0]

    # 이때도 food_id, mood_id 쌍의 중복이 있는지 확인해야 함
    sql = 'INSERT INTO userfoodinteraction (user_id, food_id, mood_id) VALUES (%s, %s, %s)'
    cursor.execute(sql, (userId, foodId, moodId))
    db.commit()
    return 'DB updated', 200


@app.route('/api/v1/foods/recommendation', methods=['POST'])
def food_recommend():
    # uuid, 감정 인덱스 받아 옴
    args = request.json
    user_id = args['uuid']  # 인코딩 필요할수도
    mood_index = args['mood']

    # food_length를 위해 food 테이블 불러오기
    foodProfile = pd.read_sql_table('food', conn)
    foodName = list(foodProfile['name'])

    # user_food_table 형태를 위해 다음과 같이 선언
    array = np.zeros([len(foodName), 7])  # row: food, col: mood
    user_food_table = pd.DataFrame(array)
    user_food_table.astype('int64')

    # UserFoodInteraction에서 불러와서 user_food_table 채우기
    sql = "SELECT food_id, mood_id FROM userfoodinteraction WHERE user_id = %s"
    cursor.execute(sql, (1))
    result = cursor.fetchall()
    for t in result:
        user_food_table.loc[t[0], t[1]] = 1

    # DB table -> dataframe
    foodProfile = pd.read_sql_table('food', conn)
    foods = list(foodProfile['name'])

    # food_index, index_food 딕셔너리 선언
    index_food = foodProfile[['name']]
    index_food = index_food.to_dict()
    index_food = index_food['name']
    index_food_swap = dict(zip(index_food.values(), index_food.keys()))
    # 새로운 음식과 기존 음식 비교위해 size 변수 선언
    # -> 우선 DB에 저장된 음식을 기준으로 할 것이므로 생략
    #     user_food_table = pd.read_sql_table('user_food_table', conn)
    #     food_indexes = list(user_food_table['food_index'])
    #     size = len(food_indexes)
    #     del(food_indexes)

    # 지금부터 추천 시작

    # 데이터 로드
    foodProfile = pd.read_sql_table('food', conn)
    # 1. foodProfile
    # 데이터 전처리
    foodProfile.drop(columns=['name', 'id'], inplace=True)
    foodProfile = foodProfile.astype(float)
    # 정규화
    scaler = MinMaxScaler()
    scaler.fit(foodProfile)
    foodProfile_scaled = scaler.transform(foodProfile)
    # 2. user_food_table 전처리
    user_food_table = user_food_table.transpose()
    user_food_table_np = user_food_table.to_numpy()
    # 3. dotProudct
    dotProduct_np = user_food_table_np.dot(foodProfile_scaled)
    dotProduct = pd.DataFrame(dotProduct_np)
    # 4. dotProduct 한 것을 기반으로 추천 아이템 return
    # dot product한 것의 감정 인덱스에 해당하는 프로필 불러옴
    userProfile = dotProduct.iloc[mood_index]
    # 음식 프로필과 감정 인덱스에 해당하는 예측값을 구함
    userPrediction = foodProfile_scaled.dot(userProfile)
    # 정렬을 위해 시리즈화
    userPrediction = pd.Series(userPrediction)
    userPrediction = userPrediction.sort_values(ascending=False)
    # output 변수
    recommend_index = userPrediction.index
    recommend_foods = []
    for i in range(len(recommend_index)):
        recommend_foods.append(index_food[recommend_index[i]].replace('_', ' '))
    data = {
        "food": recommend_foods[:10]
    }
    return jsonify(data), 200


@app.route('/api/v1/musics/user-profile', methods=['POST'])
def data_updated():
    # 입력값 처리
    args = request.json
    uuid = args['uuid']
    moodId = args['moodId']
    track = args['track']
    artist = args['artist']  # optional

    # 아티스트 + 제목으로 검색해야 정확함 -> 완
    # 제목만으로 검색해도 됨 !
    full = 'track:' + track + ' ' + 'artist:' + artist
    result = sp.search(q=full, type='track')
    musicId = result['tracks']['items'][0]['id']

    # 여기서 uuid 입력값 검증 - DB에 존재하는지 확인 필요
    sql = "SELECT id FROM user WHERE uuid = %s"
    cursor.execute(sql, (uuid))
    result = cursor.fetchall()
    userId = result[0][0]

    # 이때도 food_id, mood_id 쌍의 중복이 있는지 확인해야 함
    sql = 'INSERT INTO usermusicinteraction (user_id, music_id, mood_id) VALUES (%s, %s, %s)'
    cursor.execute(sql, (userId, musicId, moodId))
    db.commit()

    return 'DB updated', 200


@app.route('/api/v1/musics/recommendation', methods=['POST'])
def music_recommend():
    # 입력값 처리
    args = request.json
    uuid = args['uuid']
    moodId = args['mood']

    # 여기서 uuid 입력값 검증 - DB에 존재하는지 확인 필요
    sql = "SELECT id FROM user WHERE uuid = %s"
    cursor.execute(sql, (uuid))
    result = cursor.fetchall()
    userId = result[0][0]

    # UserMusicInteraction에서 music, mood 불러와서 데이터 프레임화
    sql = "SELECT music_id, mood_id FROM usermusicinteraction WHERE user_id = %s"
    cursor.execute(sql, (userId))
    result = cursor.fetchall()
    df = pd.DataFrame(result, columns=['music_id', 'mood_id'])

    # moodIndex 로 music id 분리
    tracks_id = list(df[df['mood_id'] == moodId]['music_id'])

    # artists id 분리
    result = sp.tracks(tracks_id)
    artists = []
    for r in result['tracks']:
        artists.append(r['album']['artists'][0]['id'])
    artists_id = set(artists)

    # artists_id 로 장르 뽑기
    result = sp.artists(artists_id)
    seed_genres = []
    for r in result['artists']:
        for item in r['genres']:
            seed_genres.append(item)
    seed_genres = set(seed_genres)
    # seed 장르로 넣을 수 있는 값들의 교집합 구하기
    seed_genres = seed_genres.intersection(seeds)
    # 장르, 아티스트 , 트랙 seed들 합쳐서 랜덤으로 합이 5개 경우의 수 다음과 같이 지정
    # 랜덤으로 선택 진행
    seed_ratio = [[2, 2, 1], [2, 1, 2]]
    seed_ratio = random.choice(seed_ratio)
    seed_g = random.sample(list(seed_genres), seed_ratio[0])
    seed_t = random.sample(tracks_id, seed_ratio[1])
    seed_a = random.sample(list(artists_id), seed_ratio[2])

    # recommend api 호출
    result = sp.recommendations(seed_artists=seed_a, seed_tracks=seed_t, seed_genres=seed_g, limit=10)

    # 데이터 파싱
    music = []
    for r in result['tracks']:
        music.append({"artist": r['artists'][0]['name'], "track": r['name'], "url": r['album']['images'][0]['url']})
    data = {
        "music": music
    }
    return jsonify(data), 200

if __name__ == '__main__':
    app.run()