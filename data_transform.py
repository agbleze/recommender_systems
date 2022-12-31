from pyspark.sql import SparkSession 
from pyspark.sql.functions import col, lit, rand



class DataTransformer(object):
    def __init__(self, dataDirpath, useSpark = True, sessionName = 'rs'):
        if useSpark:
            self.spark = SparkSession.builder.appName(sessionName).getOrCreate()
            
        self.data = self.spark.read.csv(dataDirpath)
    
    
    
    def convert_column_to_numeric(self):
        pass



