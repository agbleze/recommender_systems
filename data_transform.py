from pyspark.sql import SparkSession 
from pyspark.sql.functions import col, lit, rand



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
    
    
    
    def convert_column_to_numeric(self):
        pass



