#!/bin/bash

hdfs dfs -mkdir -p /createme       
hdfs dfs -rm -r /delme             
echo "This is a test file." | hdfs dfs -put - /nonnull.txt 

hadoop jar $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-*.jar \
    wordcount /shadow.txt /output

hdfs dfs -cat /output/part-r-* | grep -w "Innsmouth" | awk '{print $2}' > count.txt
hdfs dfs -put count.txt /whataboutinsmouth.txt
