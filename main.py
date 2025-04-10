from fastapi import FastAPI
import uvicorn
from fastapi import FastAPI, File, UploadFile
import logging
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware


app=FastAPI()

origins = [
    "http://localhost:5173",
    "https://potatoes-leaf-disease-detection.onrender.com/predict"
    "https://potatoleafdiseasedetectionapp.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://potatoes-leaf-disease-detection.onrender.com/predict","http://localhost:5173", "https://potatoleafdiseasedetectionapp.vercel.app/"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# write the end point aafter ts serving set up#"
# endpoint="http://localhost:8501/v1/models/potatoes_disease:predict"
endpoint = "https://potatoes-leaf-disease-detection.onrender.com/predict"





MODEL = tf.keras.models.load_model("potato.h5")    
CLASS_NAMES=["Early Blight", "Late Blight", "Healthy"]  

# @app.get("/ping")
# async def ping():
#     return "hello Taiwo, you wrote your first fast api code. Congratulations!"
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

def read_file_as_image(data)-> np.ndarray:
   image= np.array(Image.open(BytesIO(data))) 
   return image

@app.post("/predict")
async def predict(file: UploadFile):
    imagearray = read_file_as_image(await file.read())
    image_batch=np.expand_dims(imagearray,0)
    predictions= MODEL.predict(image_batch)
    predicted_class= CLASS_NAMES[np.argmax(predictions[0])]
    confidence_level=np.max(predictions[0])
    # logger.debug(predicted_class,confidence_level,image_batch)
    return {"class": predicted_class,"Confidence" : float(confidence_level)}
    

if __name__ =="__main__":
    uvicorn.run(app,host='localhost',port=8000)
    
#convert to numpy array#

