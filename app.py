import csv
import pandas as pd 
import numpy as np 
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from datetime import datetime

def convert_date(date_obj):
    year = str(date_obj.year)
    month = str(date_obj.month).zfill(2)
    day = str(date_obj.day).zfill(2)
    return year + month + day

def get_data(filename):
    df = pd.read_excel('GPdata.xls', engine = 'xlrd', usecols = [0, 1])
    data = {}
   
    for index, row in df.iterrows():
        date = convert_date(row[0])
        price = row[1]
        
        data[date] = price
        print(f"Date: {date}, Price: {price}")
    return data
        
def predict_prices(data_dict, date):
    dates = []
    prices = []

    for d, price in data_dict.items():
        dates.append(d)
        prices.append(price)

    dates = np.array(dates).reshape(-1, 1)
    prices = np.array(prices)

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates.ravel(), prices.tolist(), color='black', label='Data')
    plt.plot(dates.ravel(), svr_rbf.predict(dates), color='red', label='RBF model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    date = np.array([date]).reshape(1, -1)  

    predicted_price = svr_rbf.predict(date)[0]
    return predicted_price

data_dict = get_data('GPdata.xls')

future_date = '20201109'  
predicted_price = predict_prices(data_dict, future_date)

print(f"Predicted Gas Price for {future_date}: {predicted_price}")
    



    
    
        
            
    
   