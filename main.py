import argparse
from VectorSpace import VectorSpace
import sys, getopt, os, util
import numpy as np

def main(query):

  #create vector space model instance
  vectorSpace = VectorSpace(documents)

  #caculate different conmbinations
  tf_cos = vectorSpace.TF_Cosine(query)
  tf_euclidean = vectorSpace.TF_Euclidean(query)
  tfidf_cos = vectorSpace.TFIDF_Cosine(query)
  tfidf_euclidean = vectorSpace.TFIDF_Euclidean(query)

  #sort with top five score
  top5_tf_cos = sorted(list(zip(indexList, tf_cos)), reverse=True, key=lambda x: x[1])[:5]
  top5_tf_euclidean = sorted(list(zip(indexList, tf_euclidean)), reverse=False, key=lambda x: x[1])[:5]
  top5_tfidf_cos = sorted(list(zip(indexList, tfidf_cos)), reverse=True, key=lambda x: x[1])[:5]
  top5_tfidf_euclidean = sorted(list(zip(indexList, tfidf_euclidean)), reverse=False, key=lambda x: x[1])[:5]


  #print out the output
  print('Term Frequency Weighting + Cosine Similarity:')
  print_top(top5_tf_cos)

  print('Term Frequency Weighting + Euclidean Distance:')
  print_top(top5_tf_euclidean)

  print('TF-IDF Weighting + Cosine Similarity:')
  print_top(top5_tfidf_cos)

  print('TF-IDF Weighting + Euclidean Distance:')
  print_top(top5_tfidf_euclidean)


  #Relevance Feedback

  #get the document of the first score of the tfidf + cosine similarity by given query
  indx_fb = indexList.index(top10_tfidf_cos[0][0])
  fb = documents[indx_fb]

  #the new query term weighting scheme is [1 * original query + 0.5 * feedback query]
  feedback_vector = vectorSpace.makeFeedbackVector(fb)
  query_vector = np.array(vectorSpace.makeTfIdfVector(query))
  rf_vector = query_vector+feedback_vector

  # evaluate the relevance vector with each document by tfidf + cosine similarity
  rf_tfidf_cos = []
  for documentTFIDFVector in vectorSpace.documentTFIDFVectors:
      rf_tfidf_cos.append(util.cosine(rf_vector, documentTFIDFVector))

  top5_rf_tfidf_cos = sorted(list(zip(indexList, rf_tfidf_cos)), reverse=True, key=lambda x: x[1])[:5]

  #print out the output
  print('Relevance Feedback + TF-IDF Weighting + Cosine Similarity:')
  print_top(top5_rf_tfidf_cos)


def print_top(top_10):
  print('DocID         Score')
  for each in top_5:
    print('{}    {}'.format(each[0], each[1]))
  print('')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Project 1')
  parser.add_argument('-q', '--query', dest='query')
  args = parser.parse_args()
  #input query
  query = args.query

  #documents contents and contents' index
  documents = []
  indexList = []

  #go through all documents
  for file in os.listdir("./EnglishNews"):
    if file.endswith(".txt"):
        file_path = os.path.join("./EnglishNews", file)
        with open(file_path, 'r') as f:
          documents.append(f.read())
        indexList.append(file[:10])
  
  #main execution function
  main(query)

  






