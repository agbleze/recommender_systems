
#%%
from data_transform import DataTransformer
from helper import get_file_path
from recommender_model import AlternateLeastSquaresModel
import seedir


#%%

seedir.seedir(style='emoji', exclude_folders=['.git', '__pycache__'], 
              exclude_files=['.DS_Store', 'README.md']
              )


#%%

data_path = get_file_path(folder_name='data_store', file_name='movie_ratings_df.csv')


#%%

transformerPipeline = DataTransformer(dataDirpath=data_path, sessionName='recommender')

#%%

data = transformerPipeline.data

data.show(n=5)

#%%
data.printSchema()
#%% the movie title is string and is converted to numberic

indexData = transformerPipeline.convertColumnToNumeric()

indexData.show(n=10)

stringIndexerModel = transformerPipeline.getStringIndexer()


stringIndexPath = get_file_path(folder_name='model_store', file_name='string-indexer-model')


#%%
stringIndexerModel.write().overwrite().save(path=stringIndexPath)

#%%


# indexData.write.format('csv')\
#     .mode('overwrite')\
#         .options(sep=',', header='true').save('indexed_data.csv')





#%% split data

trainData, testData = transformerPipeline.splitData()


# %%
alsModel = AlternateLeastSquaresModel(userCol='userId',
                                      itemCol='title_new', 
                                      ratingCol='rating', 
                                      trainData=trainData,
                                      testData=testData
                                      )

#%%

fittedData = alsModel.fit()


#%%
predictionData = alsModel.predict()


#%%
alsModel.evaluateModel()


#%%
alsModel.saveModel(modelStoreFolderName='model_store')

# %%
loadedModel = alsModel.getModel(modelStoreFolderName='model_store')


#%%

from helper import recommendItem

#%%

userRecommendedItems = recommendItem(userId= 100, modelStoreFolderName='model_store',
                                    modelName='model.h5',
                                    numberOfItems=5, 
                                    getModel=AlternateLeastSquaresModel.getModel,
                                    readIndexedDataFromFile=False,
                                    indexedDataObject=indexData,
                                    stringIndexDir=get_file_path(folder_name='model_store', 
                                                                file_name='string-indexer-model'
                                                                )
                                    )


#%%
userRecommendedItems.select('title').show()


#%%

type(userRecommendedItems.select('title'))

# %%
userRecommendedItems.select('title').count()
# %%
