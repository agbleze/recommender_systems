
#%%
from pyspark.sql import SparkSession


#%%create spark session

spark = SparkSession.builder.appName('lin_reg')\
    .getOrCreate()


#%%

datafile = '/Volumes/Elements/Dataset_movie/movie_ratings_df.csv'


# %%
df = spark.read.csv(datafile, inferSchema=True, header=True)

#%%
df.count(), len(df.columns) # number of rows and columns

#%%






# %%
