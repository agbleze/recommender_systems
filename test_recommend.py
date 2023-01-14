from data_transform import DataTransformer
from recommender_model import AlternateLeastSquaresModel
import pytest
from helper import get_file_path


data_file_path = get_file_path(folder_name='data_store', file_name='movie_ratings_df.csv')


dt = DataTransformer(dataDirpath=data_file_path, sessionName='recommender')


def test_numeric_title_column_available():
    pass




def test_shape_spitData():
    pass



def test_predict():
    pass



def testing_convertColumnToNumeric():
    pass



def test_splitData():
    pass



def test_get_model():
    pass



def test_fit_model():
    pass


def test_predict_ratings():
    pass



def test_evaluate_model():
    pass
    
    













