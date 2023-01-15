from data_transform import DataTransformer
from recommender_model import AlternateLeastSquaresModel
import pytest
from helper import get_file_path
import pyspark


data_file_path = get_file_path(folder_name='data_store', file_name='movie_ratings_df.csv')


dt = DataTransformer(dataDirpath=data_file_path, sessionName='recommender')

data = dt.data


@pytest.fixture()
def create_model_instance():
    model = AlternateLeastSquaresModel()
    return model

@pytest.fixture()
def create_train_test_data():
    pass


@pytest.fixture()
def create_data_transformer_object():
    dt = DataTransformer(dataDirpath=data_file_path, sessionName='recommender')
    return dt




def test_title_column_is_availaible():
    assert 'title' in data.columns



def test_title_column_is_numeric():
    pass




def test_shape_spitData():
    dt.convertColumnToNumeric()
    assert len(dt.splitData()) == 2  # indicates both train and test data were returned successfully



def test_get_string_indexer_type():
    assert type(dt.getStringIndexer()) == pyspark.ml.feature.StringIndexerModel
    



def test_fit_model():
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
    
    













