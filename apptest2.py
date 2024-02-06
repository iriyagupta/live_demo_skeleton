import flask
#import matplotlib.pyplot as plt
import os
import cv2
import re
import numpy as np
import glob
import random
import testfunction
import feather
from PIL import Image
from socket import gethostname
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from flask import Flask, render_template , url_for ,request
from werkzeug.utils import secure_filename
#import sys
#import cocoflask
from testfile2 import perform_test
import torch 
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as torchvar 
import torch.optim as optim



#initializing app
app = Flask(__name__, static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
#creating dictionary for basename
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/Inpainting-Simple')
#UPLOAD_FOLDER = os.path.basename("static/Inpainting-Simple")
app.config["UPLOAD_FOLDER"]= UPLOAD_FOLDER
#DISPLAY_FOLDER = os.path.join(APP_ROOT, 'static/tampered')
#UPLOAD_FOLDER = os.path.basename("static/Inpainting-Simple")
#app.config["DISPLAY_FOLDER"]= DISPLAY_FOLDER
#def function of inpainting
@app.route("/",methods=["GET","POST"])
def home_page():
	print("hey")
	return render_template("home.html")

#for the first upload page
@app.route("/upload",methods=["GET","POST"])
def upload_file():
	print("hey par2")
	return render_template("index.html")

#for 2nd augmentation link
@app.route("/demo",methods=["GET","POST"])
def upload_file1():
	print("hey par2")
	return render_template("index2.html")

#for 3rd augmentation link
@app.route("/featherup",methods=["GET","POST"])
def upload_file2():
	print("hey par2")
	return render_template("index3.html")

#for the final mask page
@app.route("/getFace",methods=["GET","POST"])
def post_output():
	print("Hey posting the image")
	file = request.files["im"]
	file.filename = "input.jpg"
	file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
	#a = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
	#print(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
	#print(inpaint(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))))
	dst = testfunction.inpaint(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
	return render_template("getFace.html", image_name="static/tampered/output.jpg")


#for the final mask page
@app.route("/getFeathering",methods=["GET","POST"])
def post_feathering():
	print("Hey posting the feathered image")
	file = request.files["im"]
	file.filename = "input.jpg"
	file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
	#a = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
	#print(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
	#print(inpaint(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))))
	dst = feather.feathering(app.config["UPLOAD_FOLDER"])
	return render_template("getFeathering.html", image_name="../static/merged/output.png")


@app.route("/getDemo",methods=["GET","POST"])
def post_demo():
	print("Hey posting the demo of output image")
	file = request.files["im"]
	file.filename = "input.jpg"
	file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
	#a = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
	#print(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
	#print(inpaint(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))))
	test=perform_test(transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], 
                                std = [0.225, 0.225, 0.225])]))
	output_number = test(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
	return render_template("getDemo.html", output = output_number)


@app.route("/getDemostatic",methods=["GET","POST"])
def post_demostatic():
	print("Hey posting the demo of output image")
	test=perform_test(transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], 
                                std = [0.225, 0.225, 0.225])]))
	output_number = test("static/sampleimages/1966_0.png")
	return render_template("getDemostatic.html", output = output_number)


@app.route("/getDemostatic1",methods=["GET","POST"])
def post_demostatic1():
	print("Hey posting the demo of output image")
	test=perform_test(transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], 
                                std = [0.225, 0.225, 0.225])]))
	output_number = test("static/sampleimages/110.jpg")
	return render_template("getDemostatic1.html", output = output_number)

@app.route("/getDemostatic2",methods=["GET","POST"])
def post_demostatic2():
	print("Hey posting the demo of output image")
	test=perform_test(transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], 
                                std = [0.225, 0.225, 0.225])]))
	output_number = test("static/sampleimages/1089_0.png")
	return render_template("getDemostatic2.html", output = output_number)



if __name__ == '__main__':
    app.debug = True
    app.run(host='preon.iiit.ac.in', port=5050)

