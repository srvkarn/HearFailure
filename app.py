#!/usr/bin/env python
# coding: utf-8

# In[1]:



from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd 



from sklearn.ensemble import GradientBoostingClassifier


# In[2]:


heart_data = pd.read_csv('heart_failure_clinical_record.csv')


# In[3]:


Features = ['time','ejection_fraction','serum_creatinine','age','sex']
x = heart_data[Features]
y = heart_data["DEATH_EVENT"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)


# In[4]:


# GradientBoostingClassifier

gradientboost_clf = GradientBoostingClassifier(max_depth=2, random_state=1)
gradientboost_clf.fit(x_train,y_train)
gradientboost_pred = gradientboost_clf.predict(x_test)
gradientboost_acc = accuracy_score(y_test, gradientboost_pred)


# In[5]:


from flask import Flask, render_template,request

app = Flask(__name__)
 

@app.route("/")
def home():
    return render_template("HeartFail.html")

@app.route('/details', methods=['POST', 'GET'])
def api_response():
    if request.method == 'POST':
        Time = int(request.form["T1"])
        EjectionFriction = int(request.form["ef"])
        SerumC = int(request.form["sc"])
        Age = int(request.form["A1"])
        Gen = int(request.form["G"])
        if Gen =="MALE":
            Gen = 1
        else :
            Gen =0
        result = gradientboost_clf.predict([[Time,EjectionFriction,SerumC,Age,Gen]])
        if 1 in result:
            result ="Yes"
        else:
            result ="No"
        return render_template("HeartFail.html",Answer =result)

if __name__ == "__main__" :
    app.run()


# In[ ]:




