from pyspark.sql import SparkSession 
from pyspark.sql.functions import col, lit, rand
from pyspark.ml.feature import StringIndexer, IndexToString
import os



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
            
        self.data = self.spark.read.csv(dataDirpath, header=True)
    
    
    
    def convertColumnToNumeric(self, inputCol="title", outputCol="title_new"):
        """Method to convert a string column to a numeric column

        Args:
            inputCol (str, optional): The column containing the string to be converted. Defaults to "title".
            outputCol (str, optional): The column name to give to the numeric feature after converting from string
                    . Defaults to "title_new".

        Returns:
            pyspark dataframe: pysaprk dataframe containing a column named as that of outputCol provided
        """
        self.inputCol = inputCol
        self.outputCol = outputCol
        
        stringIndexer = StringIndexer(inputCol=self.inputCol,
                                    outputCol=self.outputCol
                                    )
        model = stringIndexer.fit(self.data)

        self.indexed = model.transform(self.data)
        return self.indexed
    
    
    def splitData(self, trainingSize=0.75, testingSize=0.25, seed=2022):
        """ 
        Method to split the data into training and testing sets

        Args:
            trainingSize (float, optional): The percentage of the data to be used for training. Defaults to 0.75.
            
        Returns:
            pyspark dataframe: Training dataset 
        """
        self.trainingSize = trainingSize
        self.testingSize = testingSize
        self.seed = seed
        
        self.trainData, self.testData = self.indexed.randomSplit(weights=[self.trainingSize,
                                                                          self.testingSize
                                                                          ],
                                                                 seed=self.seed
                                                                )
        
        return self.trainData, self.testData
        
        
    @property
    def trainTestDataSize(self):
        
        return self.trainData.count(), self.testData.count()
    







