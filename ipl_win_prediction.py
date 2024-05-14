import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')

total_score_df = deliveries.groupby(['match_id','inning']).sum()['total_runs'].reset_index()

total_score_df = total_score_df[total_score_df['inning']==1]

match_df = matches.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]

match_df= match_df[['match_id','city','winner','total_runs']]

deliveries_df = match_df.merge(deliveries,on='match_id')

deliveries_df = deliveries_df[deliveries_df['inning']==2]
deliveries_df['total_runs_y'] = pd.to_numeric(deliveries_df['total_runs_y'], errors='coerce')
deliveries_df['current_score'] = deliveries_df.groupby('match_id')['total_runs_y'].cumsum()

deliveries_df['runs_left'] = deliveries_df['total_runs_x'] - deliveries_df['current_score']
deliveries_df['balls_left'] = 126-(deliveries_df['over']*6+deliveries_df['ball'])

deliveries_df['player_dismissed'] = deliveries_df['player_dismissed'].fillna("0")
deliveries_df['player_dismissed'] = deliveries_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")
deliveries_df['player_dismissed'] = deliveries_df['player_dismissed'].astype('int')
wickets = deliveries_df.groupby('match_id')['player_dismissed'].cumsum().values
deliveries_df['wickets'] = 10 - wickets

deliveries_df['crr'] = deliveries_df['current_score']*6/(120-deliveries_df['balls_left'])

deliveries_df['rrr'] = deliveries_df['runs_left']*6/(120-deliveries_df['balls_left'])

def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0

deliveries_df['result'] = deliveries_df.apply(result,axis=1)

final_df = deliveries_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]
final_df.dropna(inplace=True)
final_df = final_df[final_df['balls_left'] != 0]

x=final_df.iloc[:,:-1]
y=final_df.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.compose import ColumnTransformer

trf = ColumnTransformer(
    [(
        'trf', OneHotEncoder(drop='first', sparse_output=False), ['batting_team', 'bowling_team', 'city']
    )],
    remainder='passthrough'
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])

pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
pipe.predict_proba(x_test)[10]

def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))

import pickle

pickle.dump(pipe,open('pipe.pkl','wb'))

