import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = {
    "StudyHours": [5, 3, 6, 2, 4, 7, 5],        
    "Attendance": [90, 75, 95, 60, 80, 100, 85], 
    "PreviousGrade": [45,35,27,48,14,50,26],
    "SleepHours": [7, 6, 8, 5, 7, 8, 7],         
    "Marks": [48,45,27,48,44,50,46]    
}

df=pd.DataFrame(data)
x=df[["PreviousGrade","StudyHours","Attendance","SleepHours"]]
y=df[["Marks"]]

model=LinearRegression()
model.fit(x,y)
pickle.dump(model,open("model.pkl",'wb'))
print("Model trained and saved")