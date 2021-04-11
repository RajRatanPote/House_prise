import pandas as pd
import cv2
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston
boston = load_boston()
print(boston)

# data = indepandent values or X
# target = depandent a=variable
df_x = pd.DataFrame(boston.data, columns=boston.feature_names)
df_y = pd.DataFrame(boston.target)

x_train,x_test,y_train,y_test = train_test_split(df_x,df_y, test_size = 0.33)


reg= linear_model.LinearRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
#print(y_pred)

import pickle

pickle.dump(reg,open('B_new.pkl','wb'))
model=pickle.load(open('B_new.pkl','rb'))