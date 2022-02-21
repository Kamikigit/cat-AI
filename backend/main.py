from fastapi import FastAPI
from fastapi import File, UploadFile
from fastapi.responses import HTMLResponse
import os
import uuid
import model
import shutil
import pandas as pd
import base64
import io
app = FastAPI()
@app.get('/')
def root():
    return {"text": "Hello World."}

@app.post("/upload")
async def uploaded(in_file:bytes = File(...)):
    # 推論中
    # print(type(in_file))
    data = io.BytesIO(in_file)
    result = predict(data)
    return {"text": result}

def predict(in_file):
    pred = model.serving(in_file)
    print('finished prediciton')
    return pred
