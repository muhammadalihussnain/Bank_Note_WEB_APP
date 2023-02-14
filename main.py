from fastapi import FastAPI
from pydantic import BaseModel
from BankNote import BankNote
import sklearn
import numpy as np 
import pandas as pd
import uvicorn
import pickle


app=FastAPI()
pickle_in=open('Classifier.pkl','rb')
model=pickle.load(pickle_in)

class Blog(BaseModel):
    title:str
    body:str

@app.post('/blog')
def prediction_mdoel(blog:BankNote):
    data=blog.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy =data['entropy']
    prediction= model.predict([[variance,skewness,curtosis,entropy]])
    if (prediction[0]>0.5):
        prediction='Fake Note'
    else:
        prediction='Bank Note'
        
    return prediction     

    return blog
