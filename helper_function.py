import torch, os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

class_dict = { 'Atelectasis': 0,
                'Cardiomegaly': 1,
                'Effusion':2 ,
                'Pneumonia':3,
                'Mass':4,
                'Nodule':5,
                'Infiltrate':6,
                'Pneumothorax':7,
                'Consolidation':8,
                'Edema':9,
                'Emphysema':10,
                'Fibrosis':11,
                'Pleural_Thickening':12,
                'Hernia':13}

def translate_result_to_English(result, lookup_dict = None):
    '''
    Args: result(int): A category prediction.
          lookup_dict(dict): A lookup dictionary for category names.
          The chest x ray label dictionary is used as a default value.

    Return: The name of that corresponding category.
    '''
    if lookup_dict is None:
        class_dict_reverse = {k: v for (v, k) in class_dict.items() }
    return class_dict_reverse[result]

def generate_bar_chart(list_of_probs, image_path_out,  list_of_cats= None):
    '''
    Args: list_of_probs(list): a list of float numbers
          lookup_dict(dict): A lookup dictionary for category names.
          The chest x ray label dictionary is used as a default value.

    Return: None. The image will be saved into designated folder.
    '''
    if list_of_cats is None:
        category_name = list(class_dict.keys())
    ax = sns.barplot(list_of_probs, category_name)
    sns.despine(ax = ax, bottom=True, left=True)
    plt.savefig(image_path_out, bbox_inches="tight", pad_inches= 0.4)
    plt.close()
