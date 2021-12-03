from os import close
from flask import Flask, render_template, Response, send_file, request
from flask.globals import request
from flask.helpers import url_for
from werkzeug.utils import redirect
from werkzeug.wrappers import response
from camera import Video 
import camera
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import io



app = Flask(__name__,static_url_path='')

@app.route('/')
def index():
   return render_template("index.html")

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield(b'--frame\r\n'
		b'Content-Type: image/jpeg\r\n\r\n' + frame + 
		b'\r\n'
		)

@app.route('/video')

def video():
	return Response(gen(Video()),
	mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/results')
def results():
	c= Video()
	c.close()
	fig,ax = plt.subplots(figsize=(12,6))
	ax = sns.set_style(style='darkgrid')
	sns.countplot(camera.emotions)
	plt.ylabel('Number of Frames')
	plt.xlabel('Emotions')
	fig.savefig('static/chart.jpg')
	return render_template('results.html')


app.run(debug = True)






