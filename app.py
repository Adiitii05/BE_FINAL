from flask import *
import sqlite3, hashlib, os
from glob import glob
#Import necessary libraries
from werkzeug.utils import secure_filename
#from flask import Flask, render_template, request
#from werkzeug import secure_filename
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
#import os
import cv2
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# pre processing input image for prediction
from PIL import Image
import matplotlib.pyplot as plt
from utils.poincare import calculate_singularities
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import calculate_minutiaes
from tqdm import tqdm
from utils.skeletonize import skeletonize
import matplotlib.image as img







# Create flask instance
app = Flask(__name__)
app.secret_key = 'random string'
#Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = tf.keras.models.load_model('fing.h5')
# Function to load and prepare the image in right shape
def read_image(filename):
	test_image = cv2.imread(filename)
	test_image = cv2.resize(test_image,(128,128))
	test_image = np.array(test_image)
	test_image = test_image.astype('float32')
	test_image /= 255
	test_image= np.expand_dims(test_image, axis=0)
	return test_image



def getLoginDetails():
	with sqlite3.connect('database.db') as conn:
		cur = conn.cursor()
		if 'email' not in session:
			loggedIn = False
			userId = ''
		else:
			loggedIn = True
			cur.execute("SELECT userId FROM users WHERE email = ?", (session['email'], ))
			userId = cur.fetchone()
	conn.close()
	return (loggedIn, userId)



def open_images(file_path):
    return np.array(cv2.imread(file_path,0) )

def fingerprint_pipline(input_img):
    block_size = 16

    # normalization - removes the effects of sensor noise and finger pressure differences.
    normalized_img = normalize(input_img.copy(), float(100), float(100))

    # ROI and normalisation
    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)

    # orientations
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)

    # find the overall frequency of ridges in Wavelet Domain
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)

    # create gabor filter and do the actual filtering
    gabor_img = gabor_filter(normim,angles,freq)
    # thinning oor skeletonize
    thin_image = skeletonize(gabor_img)
    # minutias
    minutias = calculate_minutiaes(thin_image)
    # singularities
    singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask)

    results = singularities_img

    return results



def ridge_count(img):
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    lines = 0
    for c in cnts:
        cv2.drawContours(img, [c], -1, (36,255,12), 3)
        lines += 1
        results = lines
                  
    return results

def ridge_per(a, total):
    per = ((100*a)/total)
    per = "{:.2f}".format(per)
    return per


def lobes(a,b,total):
    per = (((a+b)/total)*100)
    per = round(int(per),2)
    return per


@app.route("/", methods=['GET', 'POST'])
def home():


    return render_template('index.html')


@app.route("/new")
def new():
	if 'email' not in session:
		return redirect(url_for('loginForm'))

	else:
		with sqlite3.connect('database.db') as conn:
			cur = conn.cursor()
			cur.execute("SELECT * FROM users WHERE email=?", (session['email'], ))
			data = cur.fetchone()
		conn.close()
		return render_template('Upload.html',data=data)


@app.route("/load", methods=["GET", "POST"])
def load():
	loggedIn , userId = getLoginDetails()
	if 'email' not in session:
		return redirect(url_for('loginForm'))
	else:
		userId = request.args.get('userId')
		with sqlite3.connect('database.db') as conn:
			cur = conn.cursor()
			cur.execute("SELECT * FROM users WHERE userId = ?", (userId, ))
			user = cur.fetchone()
			cur.execute("SELECT * FROM percentage WHERE userId = ?", (userId, ))
			per = cur.fetchone()
			cur.execute("SELECT * FROM pattern WHERE userId = ?", (userId, ))
			pat = cur.fetchone()
			cur.execute("SELECT * FROM functions WHERE userId = ?", (userId, ))
			fun = cur.fetchone()
		conn.close()
		return render_template("load.html", user=user,per=per,pat=pat,fun=fun)





@app.route("/uploader", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        a,l,w = 0,0,0

        file = request.files['L1']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/uploads', filename)
            file.save(file_path)
            img = read_image(file_path)
            im = open_images(file_path)
            res = fingerprint_pipline(im)
            r1 = ridge_count(res)
            
            
            # Predict the class of an image
            class_prediction = model.predict_classes(img)
            print(class_prediction)
            #Map apparel category with the numerical class
            if class_prediction == 0:
                p1 = "Arches"
                r1=0
                a+=1
            elif class_prediction == 1:
                p1 = "loops"
                l+=1
            else:
                p1 = "whorls"
                w+=1

        file = request.files['L2']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/uploads', filename)
            file.save(file_path)
            img = read_image(file_path)
            im = open_images(file_path)
            res = fingerprint_pipline(im)
            r2 = ridge_count(res)
            # Predict the class of an image
            class_prediction = model.predict_classes(img)
            print(class_prediction)
            #Map apparel category with the numerical class
            if class_prediction == 0:
                p2 = "Arches"
                r2=0
                a+=1
            elif class_prediction == 1:
                p2 = "loops"
                l+=1
            else:
                p2= "whorls"
                w+=1


        file = request.files['L3']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/uploads', filename)
            file.save(file_path)
            img = read_image(file_path)
            im = open_images(file_path)
            res = fingerprint_pipline(im)
            r3 = ridge_count(res)

            # Predict the class of an image
            class_prediction = model.predict_classes(img)
            print(class_prediction)
            #Map apparel category with the numerical class
            if class_prediction == 0:
                p3 = "Arches"
                r3=0
                a+=1
            elif class_prediction == 1:
                p3 = "loops"
                l+=1
            else:
                p3 = "whorls"
                w+=1

        file = request.files['L4']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/uploads', filename)
            file.save(file_path)
            img = read_image(file_path)
            im = open_images(file_path)
            res = fingerprint_pipline(im)
            r4 = ridge_count(res)
            # Predict the class of an image
            class_prediction = model.predict_classes(img)
            print(class_prediction)
            #Map apparel category with the numerical class
            if class_prediction == 0:
                p4 = "Arches"
                r4=0
                a+=1
            elif class_prediction == 1:
                p4 = "loops"
                l+=1
            else:
                p4 = "whorls"
                w+=1


        file = request.files['L5']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/uploads', filename)
            file.save(file_path)
            img = read_image(file_path)
            im = open_images(file_path)
            res = fingerprint_pipline(im)
            r5 = ridge_count(res)
            # Predict the class of an image
            class_prediction = model.predict_classes(img)
            print(class_prediction)
            #Map apparel category with the numerical class
            if class_prediction == 0:
                p5 = "Arches"
                r5=0
                a+=1
            elif class_prediction == 1:
                p5 = "loops"
                l+=1
            else:
                p5 = "whorls"
                w+=1


        file = request.files['R1']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/uploads', filename)
            file.save(file_path)
            img = read_image(file_path)
            im = open_images(file_path)
            res = fingerprint_pipline(im)
            r6 = ridge_count(res)
            # Predict the class of an image
            class_prediction = model.predict_classes(img)
            print(class_prediction)
            #Map apparel category with the numerical class
            if class_prediction == 0:
                p6 = "Arches"
                r6=0
                a+=1
            elif class_prediction == 1:
                p6 = "loops"
                l+=1
            else:
                p6 = "whorls"
                w+=1


        file = request.files['R2']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/uploads', filename)
            file.save(file_path)
            img = read_image(file_path)
            im = open_images(file_path)
            res = fingerprint_pipline(im)
            r7 = ridge_count(res)
            # Predict the class of an image
            class_prediction = model.predict_classes(img)
            print(class_prediction)
            #Map apparel category with the numerical class
            if class_prediction == 0:
                p7= "Arches"
                r7=0
                a+=1
            elif class_prediction == 1:
                p7 = "loops"
                l+=1
            else:
                p7 = "whorls"
                w+=1

        file = request.files['R3']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/uploads', filename)
            file.save(file_path)
            img = read_image(file_path)
            im = open_images(file_path)
            res = fingerprint_pipline(im)
            r8 = ridge_count(res)
            # Predict the class of an image
            class_prediction = model.predict_classes(img)
            print(class_prediction)
            #Map apparel category with the numerical class
            if class_prediction == 0:
                p8 = "Arches"
                r8=0
                a+=1
            elif class_prediction == 1:
                p8 = "loops"
                l+=1
            else:
                p8 = "whorls"
                w+=1

        file = request.files['R4']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/uploads', filename)
            file.save(file_path)
            img = read_image(file_path)
            im = open_images(file_path)
            res = fingerprint_pipline(im)
            r9 = ridge_count(res)
            # Predict the class of an image
            class_prediction = model.predict_classes(img)
            print(class_prediction)
            #Map apparel category with the numerical class
            if class_prediction == 0:
                p9 = "Arches"
                r9=0
                a+=1
            elif class_prediction == 1:
                p9 = "loops"
                l+=1
            else:
                p9 = "whorls"
                w+=1

        file = request.files['R5']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/uploads', filename)
            file.save(file_path)
            img = read_image(file_path)
            im = open_images(file_path)
            res = fingerprint_pipline(im)
            r0 = ridge_count(res)
            # Predict the class of an image
            class_prediction = model.predict_classes(img)
            print(class_prediction)
            #Map apparel category with the numerical class
            if class_prediction == 0:
                p0 = "Arches"
                r0=0
                a+=1
            elif class_prediction == 1:
                p0 = "loops"
                l+=1
            else:
                p0 = "whorls"
                w+=1

        if ((a>l) and (a>w)):
            dominant = 'ARCHES'

        elif((l>a) and (l>w)):
            dominant = 'LOOPS'

        else:
            dominant = 'WHORLS'



        lrc = (r1+r2+r3+r4+r5)

        rrc = (r6+r7+r8+r9+r0)

        total = (r1+r2+r3+r4+r5+r6+r7+r8+r9+r0)



        left = round(int(((lrc)/total)*100),2)

        right = (100-left)

        P1 = ridge_per(r1,total)

        P2 = ridge_per(r2,total) 

        P3 = ridge_per(r3,total)

        P4 = ridge_per(r4,total)

        P5 = ridge_per(r5,total)

        P6 = ridge_per(r6,total)

        P7 = ridge_per(r7,total)

        P8 = ridge_per(r8,total)

        P9 = ridge_per(r9,total)

        P0 = ridge_per(r0,total)

        pref = lobes(r1,r6,total)

        frontal = lobes(r2,r7,total)

        parietal = lobes(r3,r8,total)

        temporal = lobes(r4,r9,total)

        occipital = lobes(r5,r0,total)

        loggedIn , userId = getLoginDetails()
        user = int(request.args.get('userId'))
        with sqlite3.connect('database.db') as conn:
        	cur = conn.cursor()
        	cur.execute("SELECT * FROM users WHERE email = ?", (session['email'], ))
        	userdata = cur.fetchone()
        	cur.execute('INSERT INTO percentage (a1 , a2 ,a3 ,a4 ,a5 ,a6 , a7 ,a8 , a9 , a10,userId) VALUES (?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?)',( P1,P2,P3,P4,P5,P6,P7,P8,P9,P0,user))
        	cur.execute('INSERT INTO pattern (q1 , q2 ,q3 ,q4 ,q5 ,q6 , q7 ,q8 , q9 , q10,userId) VALUES (?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?)',( p1,p2,p3,p4,p5,p6,p7,p8,p9,p0,user))
        	cur.execute('INSERT INTO functions (lrc,left,rrc,right,pref,frontal,parietal,temporal,occipital,dominant,userId) VALUES (?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?)' ,(lrc,left,rrc,right,pref,frontal,parietal,temporal,occipital,dominant,user))
        	conn.commit()
        conn.close()
        return render_template('predict.html',data = userdata, p1=p1 ,p2=p2,p3=p3,p4=p4,p5=p5,p6=p6,p7=p7,p8=p8,p9=p9,p0=p0, user_image = file_path , r1=r1,r2=r2,r3=r3,r4=r4,r5=r5,r6=r6,r7=r7,r8=r8,r9=r9,r0=r0 ,P1=P1,P2=P2,P3=P3,P4=P4,P5=P5,P6=P6,P7=P7,P8=P8,P9=P9,P0=P0,left=left,right=right,pref=pref,frontal=frontal,occipital=occipital,temporal=temporal,parietal=parietal,dominant=dominant,lrc=lrc,rrc=rrc)






'''

@app.route("/new/create", methods=["GET", "POST"])
def create():
    if 'email' not in session:
        return redirect(url_for('loginForm'))
    else:
        if request.method == 'POST':
             if request.files:
                LT1 = request.files["L1"].read()
                file_name = form.name.data
                l1 = ImageFile(image_name=file_name, image=LT1)
                LT2 = request.files['L2'].read()
                l2 = ImageFile(image_name=file_name, image=LT2)
                LT3 = request.files['L3'].read()
                l3 = ImageFile(image_name=file_name, image=LT3)
                LT4 = request.files['L4'].read()
                l4 = ImageFile(image_name=file_name, image=LT4)
                LT5 = request.files['L5'].read()
                l5 = ImageFile(image_name=file_name, image=LT5)
                RT1 = request.files['R1'].read()
                l6 = ImageFile(image_name=file_name, image=RT1)
                RT2 = request.files['R2'].read()
                l7 = ImageFile(image_name=file_name, image=RT2)
                RT3 = request.files['R3'].read()
                l8 = ImageFile(image_name=file_name, image=RT3)
                RT4 = request.files['R4'].read()
                l9 = ImageFile(image_name=file_name, image=RT4)
                RT5 = request.files['R5'].read()
                l0= ImageFile(image_name=file_name, image=RT5)
                with sqlite3.connect('database.db') as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT firstName FROM users WHERE email = ?", (session['email'], ))
                    firstName = cur.fetchone()[0]
                    cur.execute('INSERT INTO prints (LT , LI ,LM ,LR ,LS ,RT , RI ,RM , RR , RS) VALUES (?, ?, ?, ?, ?,?, ?, ?, ?, ?)',( l1,l2,l3,l4,l5,l6,l7,l8,l9,l0,))
                    conn.commit()
                    conn.close()
        return render_template("pred.html")






'''


















@app.route("/logout")
def logout():
    session.pop('email', None)
    return redirect(url_for('home'))

def is_valid(email, password):
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    cur.execute('SELECT email, password FROM users')
    data = cur.fetchall()
    for row in data:
        if row[0] == email and row[1] == hashlib.md5(password.encode()).hexdigest():
            return True
    return False

def is_valid_admin(email, password):
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    cur.execute('SELECT email, password FROM admin')
    data = cur.fetchall()
    for row in data:
        if row[0] == email and row[1] == hashlib.md5(password.encode()).hexdigest():
            return True
    return False


    
@app.route("/loginForm")
def loginForm():
    if 'email' in session:
        return redirect(url_for('new'))
    else:
        return render_template('index.html', error='')


@app.route("/login", methods = ['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if is_valid(email, password):
            session['email'] = email
            return redirect(url_for('new'))
        else:
            error = 'Invalid UserId / Password'
            return render_template('index.html', error=error)


@app.route("/register", methods = ['GET', 'POST'])
def register():
    if request.method == 'POST':
        #Parse form data    
        password = request.form['password']
        email = request.form['email']
        firstName = request.form['firstName']
        lastName = request.form['lastName']
        age = request.form['Age']
        gender = request.form['Gender']
        mobile = request.form['mobile']
        father = request.form['father']
        mother = request.form['mother']
        add = request.form['address']
        dob = request.form['dob']


        with sqlite3.connect('database.db') as con:
            try:
                cur = con.cursor()
                cur.execute('INSERT INTO users (password, email, firstName, lastName,age,gender,mobile, mother , father,address,dob) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (hashlib.md5(password.encode()).hexdigest(), email, firstName, lastName,age,gender,mobile,mother,father,add,dob))
                con.commit()
                msg = "Registered Successfully"
            except:
                con.rollback()
                msg = "Error occured"
        con.close()
        return render_template("index.html", error=msg)

@app.route("/registerationForm")
def registrationForm():
    return render_template("register.html")







if __name__ == '__main__':
    app.run(debug=True)

