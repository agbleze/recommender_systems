from data_transform import DataTransformer
from recommender_model import AlternateLeastSquaresModel
import pytest
from helper import get_file_path, recommendItem
import pyspark


data_file_path = get_file_path(folder_name='data_store', file_name='movie_ratings_df.csv')


dt = DataTransformer(dataDirpath=data_file_path, sessionName='recommender')

data = dt.data


@pytest.fixture()
def data_transformer():
    dt = DataTransformer(dataDirpath=data_file_path, sessionName='recommender')
    return dt


@pytest.fixture()
def train_test_data(data_transformer):
    data_transformer.convertColumnToNumeric()
    train_data, test_data = data_transformer.splitData()
    return train_data, test_data


@pytest.fixture()
def get_index_data(data_transformer):
    indexData = data_transformer.convertColumnToNumeric()
    return indexData

@pytest.fixture()
def ALS_model(train_test_data):
    train_data, test_data = train_test_data
    model = AlternateLeastSquaresModel(trainData=train_data, testData=test_data)
    return model


@pytest.fixture()
def prediction(ALS_model):
    ALS_model.fit()
    prediction = ALS_model.predict()
    return prediction


def test_title_column_is_availaible():
    assert 'title' in data.columns


def test_length_data_split(train_test_data):
    assert len(train_test_data) == 2  # indicates both train and test data were returned successfully


def test_get_string_indexer_type(data_transformer):
    data_transformer.convertColumnToNumeric()
    assert isinstance(data_transformer.getStringIndexer(), 
                      pyspark.ml.feature.StringIndexerModel
                      )
    

def test_fit_model(ALS_model):
    assert isinstance(ALS_model.fit(), pyspark.ml.recommendation.ALSModel)    

def test_prediction_object_type(prediction):
    assert isinstance(prediction, pyspark.sql.dataframe.DataFrame)

def test_prediction_available(prediction):
    assert 'prediction' in prediction.columns 


def test_get_model(ALS_model):
    loaded_model = ALS_model.getModel(modelStoreFolderName='model_store')
    assert isinstance(loaded_model, pyspark.ml.recommendation.ALSModel)


def test_recommend_items(get_index_data):
    userRecommendedItems = recommendItem(userId= 100, modelStoreFolderName='model_store',
                                    modelName='model.h5',
                                    numberOfItems=5, 
                                    getModel=AlternateLeastSquaresModel.getModel,
                                    readIndexedDataFromFile=False,
                                    indexedDataObject=get_index_data,
                                    stringIndexDir=get_file_path(folder_name='model_store', 
                                                                file_name='string-indexer-model'
                                                                )
                                    )
    assert userRecommendedItems.select('title').count() == 5
    



def test_evaluate_model(ALS_model):
    ALS_model.fit()
    ALS_model.predict()
    res = ALS_model.evaluateModel()
    assert isinstance(res, float)
    assert res > 1.0





