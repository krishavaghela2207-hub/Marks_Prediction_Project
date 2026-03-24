from flask import Flask,request,jsonify
import pickle

app=Flask(__name__)

model=pickle.load(open("model.pkl",'rb'))

@app.route('/')
def home():
    return "API is Running!"

@app.route("/predict",methods=["POST"])
def predict():
    data= request.json
    
    stu=data["StudyHours"]
    a=data["Attendance"]
    p=data["PreviousGrade"]
    sl=data["SleepHours"]
    
    
    result=model.predict([[stu,a,p,sl]])
    
    return jsonify({
        "salary":int(result[0])
    })
    
if __name__ == '__main__':
    app.run(debug=True)