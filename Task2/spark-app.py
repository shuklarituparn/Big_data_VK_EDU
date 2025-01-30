from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, unix_timestamp
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("Spark Homework") \
    .master("yarn") \
    .getOrCreate()

ratings = spark.read.option("header", "true").csv("hdfs://master:8020/user/ratings.csv")
tags = spark.read.option("header", "true").csv("hdfs://master:8020/user/tags.csv")

ratings_count = ratings.count()
tags_count = tags.count()

unique_movies = ratings.select("movieId").distinct().count()
unique_users = ratings.select("userId").distinct().count()

good_ratings = ratings.filter(col("rating") >= 4.0).count()

joined = ratings.join(tags, ["userId", "movieId"])
time_diff = joined.withColumn("timeDiff", unix_timestamp(col("timestamp_tags")) - unix_timestamp(col("timestamp_ratings"))) \
                 .select(avg("timeDiff")).first()[0]

avg_ratings = ratings.groupBy("userId").agg(avg("rating").alias("avgRating"))
overall_avg = avg_ratings.select(avg("avgRating")).first()[0]

tokenizer = Tokenizer(inputCol="tag", outputCol="words")
words_data = tokenizer.transform(tags)

hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurized_data = hashing_tf.transform(words_data)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_data)
rescaled_data = idf_model.transform(featurized_data)

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
model = lr.fit(rescaled_data)

predictions = model.transform(rescaled_data)
evaluator = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

results = [
    f"stages:{spark.sparkContext.statusTracker.getStageInfo().size} tasks:{spark.sparkContext.statusTracker.getActiveStageIds().size}",
    f"filmsUnique:{unique_movies} usersUnique:{unique_users}",
    f"goodRating:{good_ratings}",
    f"timeDifference:{time_diff}",
    f"avgRating:{overall_avg}",
    f"rmse:{rmse}"
]

spark.sparkContext.parallelize(results).saveAsTextFile("hdfs://master:8020/sparkExperiments.txt")

spark.stop()