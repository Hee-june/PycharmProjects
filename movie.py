import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('C:/Users/June/PycharmProjects/movies_metadata.csv')
#data.head(2)
#data.shape
data = data[['id','genres','vote_average', 'vote_count', 'popularity', 'title', 'overview']]

tmp_m = data['vote_count'].quantile(0.89)
#tmp_m

tmp_data = data.copy().loc[data['vote_count'] >= tmp_m]
#tmp_data.shape

del tmp_data

m = data['vote_count'].quantile(0.9)
data = data.loc[data['vote_count'] >= m]
#data.head()

C = data['vote_average'].mean()
#print(C)
#print(m)

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']

    return ( v / (v+m) * R) + (m / (m + v) * C)
data['score'] = data.apply(weighted_rating, axis = 1)
#data.head(5)

#data.shape

#data[['genres', 'keywords']].head(2)

data['genres'] = data['genres'].apply(literal_eval)
#data['keywords'] = data['keywords'].apply(literal_eval)
#data[['genres', 'keywords']].head(2)

data['genres'] = data['genres'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))
#data['keywords'] = data['keywords'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))

#data.head(2)
data.to_csv('C:/Users/June/PycharmProjects/pre_movies_metadata.csv', index = False)

#이상 데이터 전처리

data.genres.head(2)
count_vector = CountVectorizer(ngram_range=(1, 3))
c_vector_genres = count_vector.fit_transform(data['genres'])
c_vector_genres.shape
#단어를 벡터화 시켜서 저장

#영화의 유사도 측정을 위해  cosine similarity를 사용
#1. 코사인 유사도를 이용해 장르가 비슷한 영화를 추천
#2. vote_count를 이용하여 vote_count가 높은 것을 기반으로 최종 추천

gerne_c_sim = cosine_similarity(c_vector_genres, c_vector_genres).argsort()[:, ::-1]
#코사인 유사도를 구한 벡터를 미리 저장
#genre_c_sim.shape

def get_recommend_movie_list(df, movie_title, top=30):
    target_movie_index = df[df['title'] == movie_title].index.values

    sim_index = gerne_c_sim[target_movie_index, :top].reshape(-1)
    sim_index = sim_index[sim_index != target_movie_index]

    result = df.iloc[sim_index].sort_values('score', ascending=False)[:10]
    return result

get_recommend_movie_list(data, movie_title='The Dark Knight Rises')
data[data['title'] == 'The Dark Knight Rises']