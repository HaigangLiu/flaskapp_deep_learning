from flask_wtf import FlaskForm
from wtforms import  SubmitField, FileField, SelectField
from wtforms.validators import DataRequired
from flask_wtf.file import FileAllowed

class OneImageForm(FlaskForm):
    image = FileField('image',  validators = [DataRequired(), FileAllowed(['jpg', 'jpeg', 'png'])])
    model_selection = SelectField('Model choice',
                                  choices = [('DenseNet121', 'DenseNet'),
                                             ('ResNet152', 'ResNet')],
                                  validators = [DataRequired()])
    submit = SubmitField('submit')

class MultipleImagesForm(FlaskForm):
    image = FileField('Zip file of images',  validators = [DataRequired(), FileAllowed(['zip'])])

    model_selection = SelectField('Model choice',
                                  choices = [('DenseNet121', 'DenseNet'),
                                             ('ResNet152', 'ResNet')],
                                  validators = [DataRequired()])

    threshold = SelectField('Threshold',
                            choices = [('0.1', '0.1'),
                                       ('0.2', '0.2'),
                                       ('0.3', '0.3'),
                                       ('0.4', '0.4'),
                                       ('0.5', '0.5'),
                                       ('0.6', '0.6'),
                                       ('0.7', '0.7'),
                                       ('0.8', '0.8'),
                                       ('0.9', '0.9')],
                            validators = [DataRequired()])
    submit = SubmitField('submit')
