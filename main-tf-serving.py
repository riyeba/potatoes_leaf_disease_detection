from fastapi import FastAPI
import uvicorn
from fastapi import FastAPI, File, UploadFile
import logging
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()


origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# write the end point aafter ts serving set up#"
# endpoint="http://localhost:8501/v1/models/potatoes_model/version/2:predict"
endpoint="http://localhost:8501/v1/models/potato-disease/1:predict"






# MODEL = tf.keras.models.load_model("C:/POTOTATO_D/saved_models/potatoes_disease/1")

CLASS_NAMES=["Early Blight", "Late Blight", "Healthy"]

# @app.get("/ping")
# async def ping():
#     return "hello Taiwo, you wrote your first fast api code. Congratulations!"
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

def read_file_as_image(data)-> np.ndarray:
   image= np.array(Image.open(BytesIO(data)))
   return image

@app.post("/predict/")
async def predict(file: UploadFile=File(...)):
    image = read_file_as_image(await file.read())
    image_batch=np.expand_dims(image,0)
    
    
    json_data={
        "instances" :image_batch.tolist()
    }
    
    response= requests.post(endpoint, json=json_data)
    
    # prediction=np.array(response.json()["predictions"][0])
    prediction=response.json()["predictions"]
    print(prediction)
    predicted_class= CLASS_NAMES[np.argmax(prediction)]
    confidence_level=np.max(prediction)
    

    return {
        "class": predicted_class,
        "Confidence" : float(confidence_level)
    }
    

if __name__ =="__main__":
    uvicorn.run(app,host='localhost',port=8000)
    


