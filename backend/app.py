import os
from flask import Flask, request, redirect, url_for,flash,render_template
from werkzeug.utils import secure_filename
#from classification import neural
from imageProces import *
from neural import Neural

UPLOAD_FOLDER = '/home/adhan/Projek/sisdas-API/backend/temp'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#neural = neural()
neural = Neural()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            imageProcess(filename)
            akurasi,kelas=neural.klasifikasi()
            return render_template('index.html', berhasil="1",value=akurasi,value2=kelas)
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)