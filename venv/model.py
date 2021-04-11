import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.datasets import load_boston
boston = load_boston()

data = pd.DataFrame(boston.data)
data.head()

df = pd.DataFrame(boston.data)

df.columns = boston.feature_names
df.head()

df.describe()
df['PRICE'] = boston.target
bins = [0,200,400,600,800]
gr_name=[200,400,600,800]
df['NEW_TAX']=pd.cut(df['TAX'],bins,labels=gr_name)

bins = [0,2,4,6,8,10]
gr_name=[2,4,6,8,10]
df['NEW_RM']=pd.cut(df['RM'],bins,labels=gr_name)

df = df[~(df['PRICE'] >= 40.0)] # removed the outliers

dr_x=['PRICE','RM','TAX']
dr_y=['RM','TAX']
x=df.drop(dr_x,axis=1)
y=df.drop(dr_y,axis=1)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
#predict the price
y_pred=reg.predict(x_test)

import pickle

pickle.dump(reg,open('boston.pkl','wb'))
model=pickle.load(open('boston.pkl','rb'))