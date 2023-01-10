import os
from pyspark.ml.feature import StringIndexer, IndexToString, StringIndexerModel
from pyspark.sql.functions import rand, col, lit
from pyspark.sql import SparkSession 
from data_transform import DataTransformer

def get_file_path(folder_name: str = None, file_name: str = None):
    dirpath = os.getcwd()
    filepath = f"{folder_name}/{file_name}"
    datapath = dirpath + "/" + filepath
    return datapath



def recommendItem(userId, modelStoreFolderName, 
                  modelName,
                  #indexed,
                  numberOfItems,
                  getModel: callable, 
                  #data, 
                  itemCol: str = 'title_new',
                  readIndexedDataFromFile = True,
                  indexDataFileDirectory = None,
                  indexedDataObject = None,
                  stringIndexDir = None,
                  
                  ):
    
    if readIndexedDataFromFile:
        dt = DataTransformer(dataDirpath=indexDataFileDirectory,
                             sessionName='recommend'
                            )
        
        allItem = dt.data
        # spark = SparkSession.builder.appName('recommend')\
        #                     .getOrCreate()
        # allItem = spark.read.csv(indexDataFileDirectory, 
        #                          inferSchema=True, header=True
        #                          )
        
    loadedStringIndexerModel = StringIndexerModel.load(path=stringIndexDir)    
    allItem=indexedDataObject
    
    model = getModel(modelStoreFolderName=modelStoreFolderName, modelName=modelName)
    unique_movies = allItem.select(itemCol).distinct()
    a = unique_movies.alias('a')
    watched_movies = allItem.filter(allItem['userId']==userId)\
                            .select(itemCol).distinct()
    b = watched_movies.alias('b') 
    total_movies = a.join(b, a.title_new == b.title_new, how='left')

    remaining_movies = total_movies.where(col("b.title_new").isNull())\
                        .select(a.title_new).distinct()

    remaining_movies = remaining_movies.withColumn('userId', 
                                                    lit(int(userId))
                                                )

    recommendations = model.transform(remaining_movies)\
                                    .orderBy('prediction', 
                                            ascending=False
                                            )
                                    
    movie_title = IndexToString(inputCol=itemCol,
                                outputCol='title', 
                                labels=loadedStringIndexerModel.labels
                                )

    final_recommendations = movie_title.transform(recommendations)
    
    if numberOfItems > final_recommendations.count():
        print('The number of items specified to be recommended is more than \
                the actual recommendations so all recommendations are given.'
            )

    return final_recommendations.limit(num=numberOfItems)  #head(n=numberOfItems)  #.show(10, False)


    
