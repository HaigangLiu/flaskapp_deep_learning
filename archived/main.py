import numpy as np
from flask import Flask, render_template, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import zipfile, torch, os, secrets
import process_single_image
from forms import OneImageForm, MultipleImagesForm, LoginForm, RegistrationForm
from helper_function import translate_result_to_English, generate_bar_chart
import zipfile
from process_multiple_images import pipeline_for_multiple_images

from werkzeug import secure_filename
app = Flask(__name__)
db = SQLAlchemy(app)

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
UPLOAD_FOLDER = '/Users/haigangliu/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def intropage():
    return render_template('intropage.html', title = 'Machine Learning Project')

@app.route('/result_panel')
def result_panel():
    return render_template('result_panel.html', title = 'A panel of multiple result')

@app.route('/login')
def login():
    form = LoginForm()
    return render_template('login_page.html', form = form)

@app.route('/registration')
def registration():
    form = RegistrationForm()
    return render_template('registration.html', form = form)

@app.route('/single_image_handler', methods=['POST', 'GET'])
def single_image_handler():
    #clean up the folder first
    image_folder = app.root_path + '/static/images/'
    image_folder_generated =  app.root_path + '/static/generated_images/'

    for folder in [image_folder, image_folder_generated]:
        filelist = [ f for f in os.listdir(folder) if f.endswith(('png', 'jpg', 'jpeg')) ]
        for f in filelist:
            os.remove(os.path.join(folder, f))

    form = OneImageForm()
    if form.validate_on_submit():
        file_name = secure_filename(form.image.data.filename)
        absolute_path = image_folder + file_name
        form.image.data.save(absolute_path)

        dir_to_image = os.path.join(app.root_path,  'static/images/')
        dir_to_model = os.path.join(app.root_path, 'static/models/multi_class_.pth.tar')

        absolute_path_new_image = image_folder_generated + file_name
        pred = process_single_image.generate_prediction(path_to_image_folder = dir_to_image,
            path_to_model = dir_to_model,
            new_image_name_abs = absolute_path_new_image)

        largest_three_cat = list(np.argsort(pred)[-3:])
        resulted_diseases = [translate_result_to_English(i) for i in largest_three_cat]
        destination_dir_bar =  'static/generated_images/' + 'bar_chart_' + file_name

        generate_bar_chart(pred, destination_dir_bar)
        # return file_name
        return render_template('resultpage_single.html',
            image_location_1 = 'static/generated_images/' + file_name,
            image_location_2 = destination_dir_bar,
            probs = pred,
            largest_three_cat = resulted_diseases)

    return render_template('single_image_handler.html', title = 'Machine Learning Project', form = form)


@app.route('/multiple_images_handler', methods=['POST', 'GET'])
def multiple_images_handler():
    form = MultipleImagesForm()
    if form.validate_on_submit():
        #cleaning up
        zip_and_txt = app.root_path + '/static/multiple/'
        unzipped_ =  app.root_path + '/static/multiple/unzipped'
        for folder in [zip_and_txt, unzipped_]:
            filelist = [ f for f in os.listdir(folder) if f.endswith(('png', 'jpg', 'jpeg', 'txt','zip')) ]
            for f in filelist:
                os.remove(os.path.join(folder, f))

        random_token = secrets.token_hex(8)
        txt_file = form.ground_truth
        txt_file_dir = app.root_path + '/static/multiple/' + secrets.token_hex(8) +'.txt'
        txt_file.data.save(txt_file_dir)

        zipped_image = form.image
        zip_dir = app.root_path + '/static/multiple/' + random_token +'.zip'
        zipped_image.data.save(zip_dir)

        with zipfile.ZipFile(zip_dir,"r") as zip_ref:
            zip_ref.extractall(app.root_path + '/static/multiple/unzipped')

        list_of_aurocs, mean_of_auroc = pipeline_for_multiple_images(
            image_folder_dir = '/Users/haigangliu/ImageData/ChestXrayData',
            groundtruth_dir = txt_file_dir,
            cached_model_dir = app.root_path + '/static/models/multi_class_.pth.tar')
        list_of_aurocs = [round(result,3) for result in list_of_aurocs]
        return render_template('resultpage_multiple.html', title = 'Result of multiple-image analysis', list_of_aurocs = list_of_aurocs)

    return render_template('multiple_images_handler.html', title = 'Machine Learning Project', form = form)

@app.route('/learn_more')
def learn_more():
    return render_template('learn_more.html')

@app.route('/tut1')
def tut1():
    return render_template('tut1.html', title = 'Implement a simple regression with PyTorch')
@app.route('/tut2')
def tut2():
    return render_template('tut2.html', title = 'CNN design basics and implementation')
@app.route('/tut3')
def tut3():
    return render_template('tut3.html', title = 'A practical way of data augmentation: Augmentor')
@app.route('/tut4')
def tut4():
    return render_template('tut4.html', title = "How ReLU works and what's Guided Backprop")
@app.route('/tut5')
def tut5():
    return render_template('tut5.html', title = 'Gradient CAM in PyTorch: Into the mind of machine')
if __name__ == '__main__':
    app.run(debug = True)
