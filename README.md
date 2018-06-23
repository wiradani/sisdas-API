# sisdas-API
Intelligent systems to predict the maturity of tomatoes with artificial neural networks for classification of maturity and linear regression for heavy predictions. Using the flask framework to create web systems.

### Library :
Install using pip
* flask
* numpy
* pandas
* sklearn
* matplotlib
* csv
* Glob

### Run the web  :
1. change directory to folder backend
```
$ cd backend
```
2. export flask
```
$ export FLASK_APP=app.py
```

3. start the server
```
$ flask run
```

4. or start the server on different ip
```
$ flask run --host=0.0.0.0
```

5. open this url in your browser
```
$ http://127.0.0.1:5000/
```

### kode program  :
* app.py : fungsi main program untuk menjalankan seluruh server dan flask
* neural.py : berisikan class untuk klasifikasi kematangan menggunakan neural network backprop
* regresion.py : berisikan class untuk memprediksi berat menggunakan regresi linear
* imageProces.py : berisi class untuk menerima input gambar,melakukan praproses dan ekstrasi fitur
* ImageDataset2CSV.py : berisi fungsi untuk merubah folder image dataset menjadi file csv  

### Mengubah root folder temp  :
1. app.py 
2. neural.py
3. imageprocess.py
4. regresion.py
