from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import joblib
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.sql.functions import rand, col, lit
import os

from helper import get_file_path
from pyspark.sql import SparkSession


class AlternateLeastSquaresModel(object):
    """_summary_

    Args:
        object (_type_): Class with various methods for fitting, evaluationg,
        saving and loading recommendation models using the ALS model.
    """
    def __init__(self, maxIter=10, regParam=0.01, 
                 userCol='userId', itemCol='title_new', 
                 ratingCol='rating', nonnegative=True, 
                 coldStartStrategy='drop', trainData = None,
                 testData = None
                ):
        self.maxIter = maxIter
        self.regParam = regParam
        self.userCol = userCol
        self.itemCol = itemCol
        self.ratingCol = ratingCol
        self.nonnegative = nonnegative
        
        self.coldStartStrategy = coldStartStrategy
        self.trainData = trainData
        self.testData = testData
        
        self.model = ALS(maxIter=self.maxIter, regParam=self.regParam, 
                         userCol=self.userCol, itemCol=self.itemCol, 
                         ratingCol=self.ratingCol, nonnegative=self.nonnegative, 
                         coldStartStrategy=self.coldStartStrategy
                         )
    
    def fit(self, trainData):
        """Method for fitting ALS model to training data

        Args:
            trainData (pyspark dataframe): Data for training the model

        Returns:
            pyspark dataframe: Dataframe with prediction column indicationg predictions on 
                    the training data
        """
        if self.trainData is not None:
            try:
                self.trainData = trainData
            except:
                print('trainData needs to be provided for this method since it was not \
                      initialized for the class'
                      )
        
        self.modelFitted = self.model.fit(self.trainData)
        return self.modelFitted
        
        
    def predict(self, testData):
        if self.testData is None:
            try:
                self.testData = testData
            except:
                print('testData needs to be provided for this method since it was \
                        not initialized for the class'
                    )
                
        self.predicted_ratings = self.modelFitted.transform(self.testData)
        return self.predicted_ratings
    
    def evaluateModel(self, metricName='rmse',
                      predictionCol='prediction', 
                     labelCol='rating'
                    ):
        self.metricName = metricName
        self.predictionCol = predictionCol
        self.labelCol = labelCol
        
        self.evaluator = RegressionEvaluator(metricName=self.metricName,
                                            predictionCol=self.predictionCol, 
                                            labelCol=self.labelCol
                                            )
        
        #self.metricEvaluated =  self.metricName + 'Evaluated_'
        self.metricEvaluated = self.evaluator.evaluate(self.predicted_ratings)
        
        return self.metricEvaluated
    
    def saveModel(self, modelStoreFolderName: str, modelName='model.model'):
        modelFile = get_file_path(folder_name=modelStoreFolderName, file_name=modelName)
        #modelFile = os.path.join(os.path.dirname(__file__), modelStoreFolderName, modelName)
        joblib.dump(value=self.model, filename=modelFile)    
        print('model saved successfully')
        
    
    @staticmethod    
    def getModel(self, modelStoreFolderName: str, modelName='model.model'):
        modelFile = get_file_path(folder_name=modelStoreFolderName, file_name=modelName)
        #modelFile = os.path.join(os.path.dirname(__file__), modelStoreFolderName, modelName)
        self.loadedModel = joblib.load(filename=modelFile)
        print('Model loaded successfully')
        return self.loadedModel
   
   
        
    
def recommendItem(self, userId, modelStoreFolderName, 
                  modelName,
                  #indexed,
                  numberOfItems,
                  getModel: callable, 
                  data, 
                  itemCol: str = 'title_new',
                  
                  ):
    spark = SparkSession.builder.appName('recommend')\
                        .getOrCreate()
    allItem = spark.read.csv(data, inferSchema=True, header=True)
    self.userId = userId
    self.model = getModel(modelStoreFolderName=modelStoreFolderName, modelName=modelName)
    unique_movies = allItem.select(itemCol).distinct()
    a = unique_movies.alias('a')
    watched_movies = allItem.filter(allItem['userId']==self.userId)\
                            .select(itemCol).distinct()
    b = watched_movies.alias('b') 
    total_movies = a.join(b, a.itemCol == b.itemCol, how='left')

    remaining_movies = total_movies.where(col("b.title_new").isNull())\
        .select(a.itemCol).distinct()

    remaining_movies = remaining_movies.withColumn('userId', 
                                                lit(int(self.userId)))

    recommendations = self.model.transform(remaining_movies)\
                                    .orderBy('prediction', 
                                            ascending=False
                                            )
                                    
    movie_title = IndexToString(inputCol=itemCol,
                                outputCol='title', 
                                labels=self.model.labels
                                )

    final_recommendations = movie_title.transform(recommendations)
    
    if numberOfItems > len(final_recommendations):
        print('The number of items specified to be recommended is more than \
                the actual recommendations'
                )

    return final_recommendations.head(n=numberOfItems)  #.show(10, False)


    


