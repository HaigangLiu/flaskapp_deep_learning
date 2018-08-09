import numpy as np
import pandas as pd
import shutil
import zipfile
import os
from flask import Flask, render_template, url_for, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug import secure_filename

from forms import OneImageForm, MultipleImagesForm
from helper_function import translate_result_to_English, generate_bar_chart, class_dict
from image_processing import SingleImageHandler

model_dict = {'DenseNet121': ['static/models/densenet121_multilabel.pth.tar',
                              'features.denseblock4.denselayer16.conv2'],
              'ResNet152': ['static/models/resnet152_multilabel.pth.tar',
                              'layer4.2.conv3']}

app = Flask(__name__)
db = SQLAlchemy(app)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
accepted_files = ('jpg', 'png', 'jpeg')
# UPLOAD_FOLDER = '/Users/haigangliu/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class ImagesDB(db.Model):

    fileName = db.Column(db.String(100), primary_key = True)
    relativePath = db.Column(db.String(100))
    #results
    First = db.Column(db.String(20))
    Second = db.Column(db.String(20))
    Third = db.Column(db.String(20))
    ProbFirst = db.Column(db.String(20))
    ProbSecond = db.Column(db.String(20))
    ProbThird = db.Column(db.String(20))

@app.route('/')
def intropage():
    return render_template('intropage.html', title = 'Machine Learning Project')

@app.route('/result_panel', methods = ['POST'])
def result_panel():
    return render_template('result_panel.html',
                           title = 'A panel of CXR reading result')

@app.route("/chart", methods = ['POST'])
def chart():
    return render_template('chart.html',
                            values= values,
                            labels= labels,
                            title = 'Aggregated statistics')

@app.route('/single_image_handler', methods=['POST', 'GET'])
def single_image_handler():
    #clean up the folder first
    image_folder = os.path.join(app.root_path,'static/single_image/images/')
    image_folder_generated =  os.path.join(app.root_path,'static/single_image/generated_images/')

    for folder in [image_folder, image_folder_generated]:
        filelist = [ f for f in os.listdir(folder) if f.endswith(accepted_files) ]
        for f in filelist:
            os.remove(os.path.join(folder, f))

    form = OneImageForm()
    if form.validate_on_submit():

        model_rel_path, layer_to_extract = model_dict[form.model_selection.data]
        model_dir = os.path.join(app.root_path, model_rel_path)
        processor = SingleImageHandler(model_dir, layer_to_extract)
        file_name = secure_filename(form.image.data.filename)

        original_img_dir = os.path.join(image_folder, file_name)
        generated_heatmap = os.path.join(image_folder_generated, file_name)
        form.image.data.save(original_img_dir)

        processor = SingleImageHandler(model_dir, layer_to_extract)
        pred = processor.run(original_img_dir, generated_heatmap)

        ordered_output_probs = list(np.argsort(pred)[-3:])
        disease_w_highest_prob = [translate_result_to_English(i) for i in ordered_output_probs]

        barchart_dir =  os.path.join('static/single_image/generated_images/', 'bar_chart_' + file_name)
        generate_bar_chart(pred, barchart_dir)

        return render_template('resultpage_single.html',
            title = 'Result of Single Image Analysis',
            disease_w_highest_prob = disease_w_highest_prob,
            heatmap_rel = os.path.relpath(generated_heatmap, app.root_path),
            barchart_rel = barchart_dir)
    return render_template('single_image_handler.html', title = 'Upload a Single Image', form = form)

@app.route('/multiple_images_handler_beta', methods=['POST', 'GET'])
def multiple_images_handler_beta():
    form = MultipleImagesForm()
    if form.validate_on_submit():
        file_name = secure_filename(form.image.data.filename)
        image_storage = app.root_path + '/static/multiple_images/'
        heatmap_storage = app.root_path + '/static/multiple_images/heatmaps/'
        try:
            shutil.rmtree(image_storage)
        except:
            pass
        os.mkdir(image_storage)
        os.mkdir(heatmap_storage)

        form.image.data.save(image_storage + file_name)
        with zipfile.ZipFile(image_storage + file_name, 'r') as zip_ref:
            zip_ref.extractall(image_storage)

        model_rel_path, layer_to_extract = model_dict[form.model_selection.data]
        model_dir = os.path.join(app.root_path, model_rel_path)
        processor = SingleImageHandler(model_dir, layer_to_extract)

        disease_prevalence = []
        db.drop_all()
        db.create_all()

        for file in os.listdir(image_storage):
            if file.endswith(accepted_files):
                imageID =  image_storage + file
                heatmapID = heatmap_storage + file

                pred = processor.run(imageID, heatmapID)
                disease_prevalence.append(pred)

                ordered_output_probs = list(np.argsort(pred)[-3:])
                disease_w_highest_prob = [translate_result_to_English(i) for i in ordered_output_probs]
                rec = ImagesDB(fileName = file,
                              relativePath = '/static/multiple_images/heatmaps/'+ file,
                              First = disease_w_highest_prob[2],
                              Second = disease_w_highest_prob[1],
                              Third = disease_w_highest_prob[0],
                              ProbFirst = str(round(sorted(pred)[-1], 3)),
                              ProbSecond = str(round(sorted(pred)[-2], 3)),
                              ProbThird  = str(round(sorted(pred)[-3], 3)))
                db.session.add(rec)
                db.session.commit()

        disease_prevalence_df = pd.DataFrame(disease_prevalence, columns = class_dict.keys())
        mask = disease_prevalence_df > float(form.threshold.data)
        agg_stats = mask.sum(axis = 0).values
        return render_template('chart.html', agg_stats = agg_stats)

    return render_template('multiple_image_handler_beta.html',
                            title ='Analyze multiple images', form = form)

@app.route('/result_report_comparison')
def result_report_comparison():
    return render_template('resultpage_multiple.html')

@app.route('/to_panel', methods=['POST', 'GET'])
def to_panel():
    page = request.args.get('page', 1, type = int)
    posts = ImagesDB.query.paginate(page = page, per_page = 4)
    return render_template('result_panel.html', posts = posts, title = 'A panel of CXR reading result')

@app.route('/learn_more')
def learn_more():
    return render_template('learn_more.html')


@app.route('/tut1')
def tut1():
    return render_template('tut1.html',
                            title = 'Implement a simple regression with PyTorch')
@app.route('/tut2')
def tut2():
    return render_template('tut2.html',
                            title = 'CNN design basics and implementation')
@app.route('/tut3')
def tut3():
    return render_template('tut3.html',
                            title = 'A practical way of data augmentation: Augmentor')
@app.route('/tut4')
def tut4():
    return render_template('tut4.html',
                            title = "How ReLU works and what's Guided Backprop")
@app.route('/tut5')
def tut5():
    return render_template('tut5.html',
                            title = 'Gradient CAM in PyTorch: Into the mind of machine')

if __name__ == '__main__':
    app.run(debug = True)
