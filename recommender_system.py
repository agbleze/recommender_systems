
#%%
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand, col, lit

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

df.printSchema()

#%%
df.orderBy(rand()).show(10, False)

# %%

df.groupBy('userId').count().orderBy('count', ascending=False)\
    .show(10, False)

#%%

df.groupBy('title').count().orderBy('count', ascending=False)\
    .show(10, False)


#%% feature engineering

from pyspark.ml.feature import StringIndexer, IndexToString


#%%

stringIndexer = StringIndexer(inputCol="title",
                              outputCol="title_new"
                              )

model = stringIndexer.fit(df)

indexed = model.transform(df)


#%%
indexed.show(10)

#%%
indexed.groupBy('title_new').count()\
    .orderBy('count', ascending=False)\
        .show(10, False)

#%% splitting the dataset

train, test = indexed.randomSplit(weights=[0.75, 0.25],
                                  seed=2022
                                  )

#%%
train.count(), test.count()







# %% train model

from pyspark.ml.recommendation import ALS

#%%

rec = ALS(maxIter=10, regParam=0.01, userCol='userId',
          itemCol='title_new', ratingCol='rating',
          nonnegative=True, coldStartStrategy='drop'
          )

rec_model = rec.fit(train)


#%%prediction and evaluation on test data

predicted_ratings = rec_model.transform(test)

predicted_ratings.printSchema()

#%%
predicted_ratings.orderBy(rand()).show(10)

#%% evaluation
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(metricName='rmse',
                                predictionCol='prediction', 
                                labelCol='rating'
                                )
rmse = evaluator.evaluate(predicted_ratings)



#%%
unique_movies = indexed.select('title_new').distinct()

# %%
unique_movies.count()

# %%

a = unique_movies.alias('a')

#%%
user_id=85

#%%
watched_movies = indexed.filter(indexed['userId']==user_id)\
    .select('title_new').distinct()
    
 
#%%
b = watched_movies.alias('b')    
    
#%%

total_movies = a.join(b, a.title_new == b.title_new, how='left')


total_movies.show(10, False)

remaining_movies = total_movies.where(col("b.title_new").isNull())\
    .select(a.title_new).distinct()
    
remaining_movies.count()


#%%

remaining_movies = remaining_movies.withColumn('userId', 
                                               lit(int(user_id)))


remaining_movies.show(10, False)

#%%
recommendations = rec_model.transform(remaining_movies)\
    .orderBy('prediction', ascending=False)

recommendations.show(5, False)

# %%
movie_title = IndexToString(inputCol='title_new',
                            outputCol='title', labels=model.labels
                            )


final_recommendations = movie_title.transform(recommendations)

final_recommendations.show(10, False)




# %%
