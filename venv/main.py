from flask import Flask, render_template, request
import pickle
import numpy as np
model=pickle.load(open('B_new.pkl','rb'))
app = Flask(__name__)


@app.route('/')
def main():
   return render_template('home.html')



@app.route('/result',methods = ['POST'])
def result():
   #if request.method =='POST':
   print("hii")
   print(request.form)
   data1 = request.form['CRIM']
   data2 = request.form['ZN']
   data3 = request.form['INDUS']
   data4 = request.form['CHAS']
   data5 = request.form['NOX']
   data6 = request.form['AGE']
   data7 = request.form['DIS']
   data8 = request.form['RAD']
   data9 = request.form['PTRATIO']
   data10 = request.form['B']
   data11 = request.form['LSTAT']
   data12 = request.form['TAX']
   data13 = request.form['RM']
   arr = [np.array([data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13])]
   #arr = arr.astype(float)
   #arr = arr.reshape(1, -1)
   pred=model.predict(arr)
   #pred = pred.astype(float)
   #pred = round(pred,2)
   print(pred[0,0])

   return render_template('result.html',data='house prise is $ {}'.format(pred[0,0]))




if __name__ == '__main__':
   app.run(debug = True)