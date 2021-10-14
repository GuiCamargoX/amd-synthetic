from flask import Flask, render_template, request
from models import ResNet, EyeQ
import os
import glob
from math import floor

app = Flask(__name__)

IMAGE_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

model = ResNet()
model_quality = EyeQ()


@app.route('/')
def index():
    # delete all uploaded files
    files = glob.glob(app.config['UPLOAD_FOLDER']+r'/*')
    for f in files:
        os.remove(f)
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/infer', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        saveLocation = os.path.join(app.config['UPLOAD_FOLDER'], f.filename ) 
        f.save(saveLocation)
        print(saveLocation)
        quality_image= model_quality.infer(saveLocation)

        inference, confidence, prep_img, heat_img = model.infer(saveLocation)
        # make a percentage with 2 decimal points
        confidence = floor(confidence * 10000) / 100

        #save pil image prep_img
        imgLocation= os.path.join(app.config['UPLOAD_FOLDER'], 'prep'+f.filename )
        prep_img.save(imgLocation)
        heat_Location= os.path.join(app.config['UPLOAD_FOLDER'], 'heat'+f.filename )
        heat_img.save(heat_Location)

        # respond with the inference
        return render_template('inference.html', name=inference, confidence=confidence, eye = imgLocation, heat=heat_Location, quality_input=quality_image )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
