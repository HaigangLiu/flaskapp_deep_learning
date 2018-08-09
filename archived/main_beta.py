import numpy as np
from flask import Flask, render_template, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import zipfile, torch, os, secrets
import process_single_image
from forms import OneImageForm, MultipleImagesForm, LoginForm, RegistrationForm, MultipleImagesForm2
from helper_function import translate_result_to_English, generate_bar_chart, class_dict
import zipfile
from process_multiple_images import pipeline_for_multiple_images
from flask import request
from werkzeug import secure_filename
import shutil
import pandas as pd
from cam_v3 import SingleImageHandler

app = Flask(__name__)
db = SQLAlchemy(app)

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
UPLOAD_FOLDER = '/Users/haigangliu/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class ImagesDB(db.Model):

    imageID = db.Column(db.String(100), primary_key = True)
    heatmapID = db.Column(db.String(100))
    folderID = db.Column(db.String(100))
    prob_list = db.Column(db.String(100))
    fileName = db.Column(db.String(100))
    relativePath = db.Column(db.String(100))
    First = db.Column(db.String(100))
    Second = db.Column(db.String(100))
    Third = db.Column(db.String(100))
    ProbFirst = db.Column(db.String(100))
    ProbSecond = db.Column(db.String(100))
    ProbThird = db.Column(db.String(100))

@app.route("/chart")
def chart():
    return render_template('chart.html', values= values, labels= labels)

@app.route('/')
def intropage():
    return render_template('intropage.html', title = 'Machine Learning Project')

@app.route('/result_panel')
def result_panel():
    return render_template('result_panel.html', title = 'A panel of multiple result')

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

@app.route('/multiple_images_handler_beta', methods=['POST', 'GET'])
def multiple_images_handler_beta():
    form = MultipleImagesForm2()

    if form.validate_on_submit():

        db.drop_all()
        db.create_all()

        file_name = secure_filename(form.image.data.filename)

        relative = '/static/multiple_images/'
        image_storage = app.root_path + relative

        try:
            shutil.rmtree(image_storage)
        except:
            pass
        os.mkdir(image_storage)

        form.image.data.save(image_storage + file_name)

        with zipfile.ZipFile(image_storage + file_name, 'r') as zip_ref:
            zip_ref.extractall(image_storage)

        disease_prev = []
        for file in os.listdir(image_storage):

            if file.endswith(('jpg', 'png', 'jpeg')):
                imageID =  image_storage + file
                heatmapID = image_storage + 'heatmap_'+ file
                folderID = image_storage + file.split('.')[0]

                os.mkdir(folderID)
                head, file = os.path.split(imageID)
                shutil.move(imageID, os.path.join(folderID, file))

                model_dir = os.path.join(app.root_path, 'static/models/multi_class_.pth.tar')
                pred = process_single_image.generate_prediction(path_to_image_folder = folderID, path_to_model = model_dir, new_image_name_abs = heatmapID)
                disease_prev.append(pred)

                largest_three_cat = list(np.argsort(pred)[-3:])
                resulted_diseases = [translate_result_to_English(i) for i in largest_three_cat]

                im = ImagesDB(imageID =  imageID,
                              heatmapID = heatmapID,
                              folderID = folderID,
                              prob_list = pred,
                              fileName = file,
                              relativePath = relative + 'heatmap_' + file,
                              First = resulted_diseases[2],
                              Second = resulted_diseases[1],
                              Third = resulted_diseases[0],
                              ProbFirst = str(round(sorted(pred)[-1],3) ),
                              ProbSecond = str(round(sorted(pred)[-2],3) ),
                              ProbThird  = str(round(sorted(pred)[-3],3) ))

                db.session.add(im)
                db.session.commit()

        disease_prev_df = pd.DataFrame(disease_prev, columns = class_dict.keys())
        mask = disease_prev_df > 0.25
        agg_stats = mask.sum(axis = 0).values

        return render_template('chart.html', agg_stats = agg_stats)

    return render_template('multiple_image_handler_beta.html', title = 'Machine Learning Project', form = form)

@app.route('/to_panel', methods=['POST', 'GET'])
def to_panel():
    page = request.args.get('page', 1, type = int)
    posts = ImagesDB.query.paginate(page = page, per_page = 4)
    return render_template('result_panel.html', posts = posts )

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
