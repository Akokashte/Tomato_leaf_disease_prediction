from flask import Flask, request, render_template, request, url_for
from pymongo import MongoClient
import datetime

from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import urllib.request

app = Flask(__name__)
client = MongoClient("")
db = client.leaf_disease
leaf_collection = db.leaf_collection

@app.route('/',methods=['POST','GET'])
def home():
    if request.method != 'POST':
         return render_template('index.html')

@app.route('/prediction',methods=['POST','GET'])
def prediction():
    class_dict ={
                'Bacterial_spot': 0,
                'Early_blight': 1,
                'Late_blight': 2,
                'Leaf_Mold': 3,
                'Septoria_leaf_spot': 4,
                'Spider_mites Two-spotted_spider_mite': 5,
                'Target_Spot': 6,
                'Tomato_Yellow_Leaf_Curl_Virus': 7,
                'Tomato_mosaic_virus': 8,
                'healthy': 9
                }
    model = load_model('D:/BDA project/tomato_leaf_disease_prediction/tomato_disease_model.h5')
    my_url = "https://www.tutorialspoint.com/images/logo.png"
    # Download the image using urllib
    img_path = request.form['image_link']
    urllib.request.urlretrieve(img_path, "logo.png")
    print(img_path)

    # Open and resize the image
    img = Image.open("logo.png").resize((150, 150))

    # Preprocessing the image
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255

    # Make predictions using the model
    prediction = model.predict(img)
    pred = np.argmax(prediction,axis = 1)
    pred_cat = [k for k, v in class_dict.items() if v == pred[0]][0]

    # return pred_cat
    predicted_Disease = pred_cat

    if request.method == 'POST':
        name = request.form["name"]
        email = request.form["email"]
        image_link = request.form["image_link"]

        data = {"name":name,"email":email,
                 "image":image_link,
                 "disease_predicted":predicted_Disease, 
                 "date": datetime.datetime.now(tz=datetime.timezone.utc),}
        
        leaf_collection.insert_one(data)
    
        return render_template('alldata.html',data=[data])

@app.route('/alldata',methods=['POST','GET'])
def allData():
    data = [x for x in leaf_collection.find()]
    return render_template('alldata.html',data=data)

if __name__ == '__main__':
    app.run(debug=True)
