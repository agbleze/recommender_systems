from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import joblib



class AlternateLeastSquaresModel(object):
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
    
    def saveModel(self, modelStoreDir: str):
        
        
    
    
        



