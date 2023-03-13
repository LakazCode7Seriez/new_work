from PIL import Image
# import streamlit as st
import cv2
import tensorflow as tf 
import numpy as np
from keras.models import load_model
from PIL import Image
import PIL
from distutils.log import debug
from fileinput import filename
from flask import *  


app = Flask(__name__)  

#Loading the Inception model
model= load_model('frames.h5',compile=(False))

#Functions
def splitting(name):
    vidcap = cv2.VideoCapture(name)
    success,frame = vidcap.read()
    count = 0
    frame_skip =1
    while success:
        success, frame = vidcap.read() # get next frame from video
        cv2.imwrite(r"img/frame%d.jpg" % count, frame) 
        if count % frame_skip == 0: # only analyze every n=300 frames
            #print('frame: {}'.format(count)) 
            pil_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) into PIL Image
            #st.image(pil_img)
        if count > 20 :
            break
        count += 1
    preprocessing()

def preprocessing():
    x = tf.io.read_file('img/frame2.jpg')
    x = tf.io.decode_image(x,channels=3) 
    x = tf.image.resize(x,[299,299])
    x = tf.expand_dims(x, axis=0)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    return x
    
def predict(x):
    P = tf.keras.applications.inception_v3.decode_predictions(model.predict(x), top=1)
    return P


@app.route('/')
def main():  
    return render_template("index.html")



@app.route('/success', methods = ['GET,POST'])  
def success():
    if request.method == 'POST':  
        file = request.files['file']
        if file is not None: # run only when user uploads video
            vid = file.filename
            with open(vid, mode='wb') as f:
                f.write(file.read()) # save video to disk

        # st.markdown(f"""
        # ### Files
        # - {vid}
        # """,
        # unsafe_allow_html=True) # display file name

        vidcap = cv2.VideoCapture(vid) # load video from disk
        cur_frame = 0
        success = True
    
    def generatesearchitems():
        check = ""
        for i in range(20):
            filename = (r"img/frame%d.jpg" % i)
            x = tf.io.read_file(filename)
            x = tf.io.decode_image(x,channels=3) 
            x = tf.image.resize(x,[299,299])
            x = tf.expand_dims(x, axis=0)
            x = tf.keras.applications.inception_v3.preprocess_input(x)
            P = tf.keras.applications.inception_v3.decode_predictions(model.predict(x), top=1)
            if (P[0][0][1]) == selected :
                # st.success("Item Found")
                pic =  Image.open(filename)
                check = "Item Found"

                return pic,P
                # st.image(pic)
                # st.text(P)
                # return 0
            else:
                check = "Item was not found"
        
    # if request.method == 'GET':  
        output1 = splitting(vid)
        output2 = preprocessing()
        output = predict(output2)
        #st.success('The Output is {}'.format(output))
        print("Successfuly detected all the objects!")
        picture,name =  generatesearchitems()

        return render_template('success.html' , name = name, picture = picture, check = check)
    
if __name__=='__main__':
    app.run(debug=True)
