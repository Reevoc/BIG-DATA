
from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand


def main():

	# CHECKING NUMBER OF CMD LINE PARAMTERS
	assert len(sys.argv) == 5, "Usage: python3 G097WH1.py <K> <H> <S> <database>"

	# SPARK SETUP
	conf = SparkConf().setAppName('HomeWork 1 Group 97').setMaster("local[*]")
	sc = SparkContext(conf=conf)

	# INPUT READING

	# 1. Read number of partitions
	K = sys.argv[1]
	assert K.isdigit(), "K must be an integer"
	K = int(K)

	H = sys.argv[2]
	assert H.isdigit(), "H must be an integer"
	H = int(H)

	country = str(sys.argv[3])
	
	# 2. Read input file and subdivide it into K random partitions
	data_path = sys.argv[4]
	assert os.path.isfile(data_path), "File or folder not found"
	rawData = sc.textFile(data_path,minPartitions=K).cache()
	#TransactionID,ProductID, Description, Quantity int, InvoiceDate date, UnitPrice float, CustomerID int, Country
	
	## Task 1
	
	rawData.repartition(numPartitions=K)
	print("Number of rows = ", rawData.count())
	print("TASK 1 DONE")

	## Task 2
	
	productCustomer = (rawData.flatMap(lambda x: filtering_map(x, country))
		.groupByKey()
		.flatMap(lambda x: [(x[0][0], x[0][1])]))
	productCustomer.persist()
	print("Product-Customer Pairs = ", productCustomer.count())
	print("TASK 2 DONE")
	
	## Taks 3
	productPopularity1 = (productCustomer.mapPartitions(popularity_counter_partition)
										.groupByKey()
										.mapValues(lambda vals: sum(vals)))
	#print(productPopularity1.collect())
	print("TASK 3 DONE")

	## Task 4
	'''
	Repeats the operation of the previous point using a combination of map/mapToPair 
	and reduceByKey methods (instead of mapPartitionsToPair/mapPartitions) and calling
	 the resulting RDD productPopularity2.'''
	productPopularity2 = (productCustomer.flatMap(popularity_counter)
										.reduceByKey(lambda x, y: x+y))
	#print(productPopularity2.collect())
	print("TASK 4 DONE")

	## Task 5
	if H>0:
		sorted_popularity=(productPopularity1.flatMap(lambda pair: [(pair[1], pair[0])])
											.sortByKey(ascending=False)
											.take(H))
		print("Top ", H, " Products and their Popularities")
		for elem in sorted_popularity:
			print("Product ", elem[1], " Popularity ", elem[0])
	
		print("TASK 5 DONE")

	### Task 6

	if H==0:
		sorted_pop1 = (productPopularity1.sortByKey()
										.collect())
		for elem in sorted_pop1:
			print("Product ", elem[1], " Popularity ", elem[0])
		#print("POPULARITY 1: ", sorted_pop1)
		sorted_pop2 = (productPopularity2.sortByKey()
										.collect())
		#print("POPULARITY 2: ", sorted_pop2)
		for elem in sorted_pop1:
			print("Product ", elem[1], " Popularity ", elem[0])
		
		print("TASK 6 DONE")



def filtering_map(line, country):
	line = line.split(",")
	line[3]=int(line[3])
	if line[3]>0 and (country == "all" or line[7]==country):
		return[((line[1], line[6]), 1)]
	return []

def popularity_counter_partition(partition):
	new_partition=[]
	for pair in partition:
		pair=(pair[0], 1)
		new_partition.append(pair)
	return new_partition

def popularity_counter(pair):
	return [(pair[0], 1)]
		

if __name__ == "__main__":
	main()
