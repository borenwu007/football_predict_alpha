import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.optimizers import RMSprop
from keras.utils import np_utils
import urllib.request

def download(url,local):
    response=urllib.request.urlopen(url)
    html = response.read()
    with open(local,'wb') as output:
        output.write(html)


def model():
    global dict
    model = Sequential()
    model.add(Dense(200, input_shape=(2,len(dict),)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.7))
    model.add(Dense(60))
    model.add(Activation('relu'))
    model.add(Dropout(0.9))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model

def one_to_hot(team):   
    global dict
    ar=np.zeros((len(dict)), dtype=np.uint)
    teamnr=dict.get(team)
    ar[teamnr]=1
    return ar


def one_hot_res(result):
    ar = np.zeros(3, dtype=np.uint)
    if(result=='H'):
        ar[0]=1
    if (result == 'D'):
        ar[1] = 1
    if (result == 'A'):
        ar[2] = 1
    return ar

# url="http://www.football-data.co.uk/mmz4281/1617/N1.csv"
# csv="data/N1.csv"
# download(url,csv)
# download('http://www.football-data.co.uk/fixtures.csv','data/fixtures.csv')

# files = [
#             'data/E0_2019.csv','data/E0_2018.csv',
#             'data/E0_2017.csv','data/E0_2016.csv',
#             'data/E0_2015.csv','data/E1_2019.csv',
#             'data/E1_2018.csv','data/E1_2017.csv',
#             'data/E1_2016.csv','data/E1_2015.csv'
#         ]
files = ['data/E0_2018.csv',]

fixture_file = 'data/fixtures_new.csv'

dict={}
dict_rev={}

def get_df(files):
    frames = []
    cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    
    for file in files:
        df = pd.read_csv(file)
        df = df[cols]
        frames.append(df)
    
    result = pd.concat(frames)
    return result

def get_dx(file):
    cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    dx = pd.read_csv(file)
    dx = dx[cols]
    return dx

df = get_df(files)
dx = get_dx(fixture_file)


index = 0
for item in enumerate(df['HomeTeam']):
    team_name = item[1]
    if not (dict.__contains__(team_name)):
        dict[team_name]=index
        dict_rev[index]=team_name
        index += 1

# print(dict)
x=np.zeros((len(df),2,len(dict)), dtype=np.uint)
y=np.zeros((len(df),3),dtype=np.float32)
x_pred=np.zeros((len(dx),2,len(dict)),dtype=np.uint)

    

for a,b in df.iterrows():
    home=b[0]
    away=b[1]

    x[a,0,:]=one_to_hot(home)
    x[a,1, :] = one_to_hot(away)
    result=b[4]
    y[a,:]=one_hot_res(result)
fix=[]
count=0
for a, b in dx.iterrows():
    home = b[0]
    away = b[1]
    if home in dict:
        count+=1
x_pred=np.zeros((count,2,len(dict)),dtype=np.uint)
count=0
for a, b in dx.iterrows():
    home = b[0]
    away = b[1]

    if home in dict:
        fix.append(home + '-' + away)

        x_pred[count,0,:]=one_to_hot(home)
        x_pred[count, 1, :] = one_to_hot(away)
        count+=1

model=model()
print(model)


model.fit(x, y, batch_size=1, nb_epoch=50,
          verbose=1)
pred=model.predict(x_pred)
print(pred.shape,len(fix))
print("match","home","draw","away")
for a,p in enumerate(pred):
    home=round(p[0]*100)
    draw=round(p[1]*100)
    away=round(p[2]*100)
    print(fix[a],home,draw,away)