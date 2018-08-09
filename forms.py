from flask_wtf import FlaskForm
from wtforms import  SubmitField, FileField, SelectField
from wtforms.validators import DataRequired
from flask_wtf.file import FileAllowed

class OneImageForm(FlaskForm):
    image = FileField('image',  validators = [DataRequired(), FileAllowed(['jpg', 'jpeg', 'png'])])
    submit = SubmitField('submit')

class MultipleImagesForm(FlaskForm):
    image = FileField('Zip file of images',  validators = [DataRequired(), FileAllowed(['zip'])])

    model_selection = SelectField('Model choice',
                                  choices = [('Model1', 'DenseNet'),
                                             ('Model2', 'ResNet'),
                                             ('Model3', 'InceptionV3')],
                                  validators = [DataRequired()])

    threshold = SelectField('Threshold',
                            choices = [('0.1', '0.1'),
                                       ('0.2', '0.2'),
                                       ('0.3', '0.3'),
                                       ('0.4', '0.4'),
                                       ('0.5', '0.5')],
                            validators = [DataRequired()])
    submit = SubmitField('submit')
