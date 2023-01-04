from pyspark.sql import SparkSession 
from pyspark.sql.functions import col, lit, rand
from pyspark.ml.feature import StringIndexer, IndexToString



class DataTransformer(object):
    """Class with methods for preprocessing data using pyspark

    Args:
        object (_type_): _description_
    """
    def __init__(self, dataDirpath, useSpark = True, sessionName = 'rs'):
        """Initializes the DataTransformer class by creating a spark session and
        loading the data.

        Args:
            dataDirpath (str): location of data. The data is expected to be a csv file
            useSpark (_type_): Defaults to True indicating that the spark will be used
            sessionName (str): Name of the spark session
        """
        
        if useSpark:
            self.spark = SparkSession.builder.appName(sessionName).getOrCreate()
            
        self.data = self.spark.read.csv(dataDirpath)
    
    
    
    def convertColumnToNumeric(self, inputCol="title", outputCol="title_new"):
        self.inputCol = inputCol
        self.outputCol = outputCol
        
        stringIndexer = StringIndexer(inputCol=self.inputCol,
                                    outputCol=self.outputCol
                                    )
        model = stringIndexer.fit(self.data)

        self.indexed = model.transform(self.data)
        return self.indexed
    
    
    def splitData(self, trainingSize=0.75, testingSize=0.25, seed=2022):
        self.trainingSize = trainingSize
        self.testingSize = testingSize
        self.seed = seed
        
        self.trainData, self.testData = self.indexed.randomSplit(weights=[self.trainingSize,
                                                                          self.testingSize
                                                                          ],
                                                                 seed=self.seed
                                                                )
        
        
    @property
    def trainTestDataSize(self):
        
        return self.trainData.count(), self.testData.count()
    





stringIndexer = StringIndexer(inputCol="title",
                              outputCol="title_new"
                              )


