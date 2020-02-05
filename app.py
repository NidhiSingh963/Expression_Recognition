import base64
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.reset_default_graph()
import FaceToolKit as ftk
import DetectionToolKit as dtk
import matplotlib.pyplot as plt
from flask import Flask, flash, request, redirect, render_template,Response, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import PIL
import pickle
import pafy
import re
import urllib
import requests
import io
import base64
verification_threshhold = 1.175
image_size = 160
v = ftk.Verification()
# Pre-load model for Verification
v.load_model("./models/20180204-160909/")
v.initial_input_output_tensors()
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','mp4'])
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
EMOTIONS = ["sad","surprise","happy","disgust","contempt","fear","angry"]
app = Flask(__name__)
app.secret_key = "nidhipriyasingh"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

d = dtk.Detection()

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image
#buid

def counting_emotions(emotion_lists):
    present_emtion = []

    for emotion in EMOTIONS:
        co_emotion  = re.findall(emotion,str(emotion_lists).lower())
        present_emtion.append({emotion:len(co_emotion)})
    return present_emtion


def build_graph_single(image):
    img = io.BytesIO()
    plt.imshow(image)
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

def build_graph(image,pred):
    with tf.Graph().as_default():
        sess = tf.Session()
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    img = io.BytesIO()
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 #
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    default_color = (0, 255, 0) #BGR
    default_thickness = 2
    # Line thickness of 2 px
    thickness = 2

    bounding_boxes, points = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    i=0
    for bounding_box in bounding_boxes:
            pts = bounding_box[:4].astype(np.int32)
            print(pts)
            pt1 = (pts[0], pts[1])
            pt2 = (pts[2], pts[3])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(image, pt1, pt2, color=default_color, thickness=default_thickness)
            cv2.putText(image,str(pred[i][0]),(pts[0],pts[1]), font, 1,color,1, cv2.LINE_AA)
            i+=1
    img = io.BytesIO()
    plt.imshow(image)
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)



print("loading model....")
file = open("clf_model.sav",'rb')
model = pickle.load(file)
print("model loaded sucessfully")


from detection.mtcnn import detect_face
default_color = (0, 255, 0) #BGR
default_thickness = 2
color = (255, 0, 0)

with tf.Graph().as_default():
    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness=2
#
def test_model_for_url(image):
    encodings=[]
    encodings_list=[]
    prediction_list=[]
    pred=[]
    try:
        faces = d.align(image, True)
        print("aligned",faces)
        for face in faces:
            #print(face)
            encodings.append(np.array(v.img_to_encoding(face,image_size),dtype="float32").reshape(1,-1))
        for encod in encodings:
            pred.append([model.predict(encod)[0]])
        if  len(encodings)> 1:
            prediction_list.append([build_graph(image,pred),pred,counting_emotions(pred)])
        else:
            prediction_list.append([build_graph_single(image),pred,counting_emotions(pred)])
        return  prediction_list
    except Exception as e:
        print(e)
        return "invalid image format"

def show_video(video):
    cap = cv2.VideoCapture(video)
    while True:
        ret, frame = cap.read()
        #print(frame)
        if ret:
            faces = d.align(frame, True)
            encodings=[]
            pred=[]
            #print("working")
            for face in faces:
                encodings.append(np.array(v.img_to_encoding(face,image_size),dtype="float32").reshape(1,-1))
                #print(face)
            for encod in encodings:
                pred.append([model.predict(encod),model.predict_proba(encod)])
                print(pred)
            pred_val=[]
            pred_class=[]
            for i in pred:
                pred_class.append(i[1])
                pred_val.append(i[0][0])
                print(pred_val)
            bounding_boxes, points = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            for bounding_box in bounding_boxes:
                pts = bounding_box[:4].astype(np.int32)
                #print(pts)
                pt1 = (pts[0], pts[1])
                pt2 = (pts[2], pts[3])
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(frame, pt1, pt2, color=default_color, thickness=default_thickness)
                cv2.putText(frame,i[0][0],(pts[0],pts[1]), font, 1.0,color,1, cv2.LINE_AA)

            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        else:
            break


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#sadsadasdsadsad
@app.route('/')
def upload_form():
    return render_template('file_upload.html')

@app.route('/video_feed', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        #print("sadsadasdsadsad")
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        vid_url  = request.form.get('url_img')
        print("vid_url",vid_url)
        print(file.filename)
        ALLOWED_Image =["jpg","jpeg","png","gif"]
        check_youtube = re.findall("youtube",str(vid_url).lower())
        if file.filename == '':
            #flash('No file selected for uploading')
            #print("entered here")
            if str(vid_url)[:-3].lower() == "mp4":
                return Response(show_video(vid_url),mimetype='multipart/x-mixed-replace; boundary=frame')

            elif len(check_youtube)>0:
                vPafy = pafy.new(vid_url)
                play = vPafy.getbest(preftype="webm")
                #cap = cv2.VideoCapture(play.url)
                return Response(show_video(play.url),mimetype='multipart/x-mixed-replace; boundary=frame')

            else:

                image=url_to_image(vid_url)
                print("url_image read",image)
                return render_template("pic.html",results=[graph for graph in test_model_for_url(image)])

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')
            #show_video(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            return Response(show_video(os.path.join(app.config['UPLOAD_FOLDER'],filename)),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif,mp4')
            return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True,host='localhost', port=8080)
