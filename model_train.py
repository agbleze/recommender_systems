
#%%
from data_transform import DataTransformer
from helper import get_file_path
from recommender_model import AlternateLeastSquaresModel
import seedir


#%%




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

#%% the movie title is string and is converted to numberic

indexData = transformerPipeline.convertColumnToNumeric()

indexData.show(n=10)


#%% split data

trainData, testData = transformerPipeline.splitData()


# %%
alsModel = AlternateLeastSquaresModel(userCol='userId',
                                      itemCol='title_new', 
                                      ratingCol='rating_new', 
                                      trainData=trainData,
                                      testData=testData
                                      )

#%%

alsModel.fit()








# %%
