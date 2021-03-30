import sys, math
import numpy as np
from scipy.spatial import distance

#http://www.scipy.org/
try:
	from numpy import dot
	from numpy.linalg import norm
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))


def cosine(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors :
		cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
	return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))

#term frequency
def tf(query, queryList):
	# print('tf:'+queryList.count(query) / len(queryList))
	return queryList.count(query) / len(queryList)

def n_containing(query, documents):
	return sum(1 for document in documents if query in document)

def ln(x):
    n = 1000.0
    return n * ((x ** (1/n)) - 1)

#idf score
def idf(word, documents):
	# print('idf:'+math.log(len(documents) / (1 + n_containing(word, documents))))
	return ln(len(documents) / (1 + n_containing(word, documents)))

#tf-idf
def tfidf(word, wordList, documents):
	return tf(word, wordList)*idf(word, documents)

#calculate the similarity of 2 vector with Cosine Similarity
def cosine(vector1, vector2):
	return float(dot(vector1, vector2) / (norm(vector1) * norm(vector2)))

#calculate the similarity of 2 vector with Euclidean Distance
def eu(vector1, vector2):
	# return float(norm(np.array(vector1)-np.array(vector2)))
	return float(distance.euclidean(vector1, vector2))

