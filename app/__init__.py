from flask import Flask, render_template
from visual_features import features
from PIL import Image
import glob

app = Flask(__name__)

dati = glob.glob("/home/dario/wsgi/app/static/*")
data = []
images = []

# creo una lista di dizionari, dove ogni dizionario è composto dal percorso dell'immagine e dal suo relativo nome
for i in dati:
    dict = {}
    dict['path'] = i
    dict['name'] = i[28:]
    data.append(dict)

# creo una lista di dizionari images, dove i campi di ogni dizionario inzicano i valori dei descrittori per ogni singola immagine
for img in data:
    image = {}
    img_obj = Image.open(img['path'])
    image['name'] = img['name']
    image['coarseness'] = features.feature_coarseness(img_obj)
    image['contrast'] = features.feature_contrast(img_obj)
    image['directionality'] = features.feature_directionality(img_obj)
    image['line_likeliness'] = features.feature_line_likeliness(img_obj)
    image['roughness'] = features.feature_roughness(img_obj)
    images.append(image)


@app.route('/')
def index():
    return render_template('index.html', images=images)


@app.route('/coarseness')
def coarseness():
    return render_template('coarseness.html', images=images)


@app.route('/contrast')
def contrast():
    return render_template('contrast.html', images=images)


@app.route('/directionality')
def directionality():
    return render_template('directionality.html', images=images)


@app.route('/linelikeliness')
def linelikeliness():
    return render_template('line_likeliness.html', images=images)


@app.route('/roughness')
def roughness():
    return render_template('roughness.html', images=images)

