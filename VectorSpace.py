from pprint import pprint
from Parser import Parser
import util
import numpy as np
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('averaged_perceptron_tagger')

np.seterr(divide='ignore', invalid='ignore')

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []
    documentTFVectors = []
    documentTFIDFVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Tidies terms
    parser=None

    documents = []


    def __init__(self, documents=[]):
        self.documentVectors=[]
        self.documentTFVectors=[]
        self.documentTFIDFVectors =[]
        self.parser = Parser()
        self.documents = documents
        if(len(documents)>0):
            self.build(documents)


    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        # self.documentVectors = [self.makeVector(document) for document in documents]

        #calculate each document's tf vector & tf-idf vector
        self.documentTFVectors = [self.makeTfVector(document) for document in documents]
        self.documentTFIDFVectors = [self.makeTfIdfVector(document) for document in documents]

        #print(self.vectorKeywordIndex)
        #print(self.documentVectors)


    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)


    def makeTfVector(self, word):
        """make tf vector"""

        wordList = self.parser.tokenise(word)
        wordList = self.parser.removeStopWords(wordList)

        tf_vector =[0]*len(self.vectorKeywordIndex)
        for each in wordList:
            try:
                tf = util.tf(each, wordList)
                tf_vector[self.vectorKeywordIndex[each]]=tf
            except:
                continue
        
        # print([i for i, e in enumerate(tf_vector) if e != 0])

        return tf_vector

    def makeTfIdfVector(self, word):
        """make tfidf vector"""

        wordList = self.parser.tokenise(word)
        wordList = self.parser.removeStopWords(wordList)

        tfidf_vector =[0]*len(self.vectorKeywordIndex)
        for each in wordList:
            try:
                tfidf = util.tfidf(each, wordList, self.documents)
                tfidf_vector[self.vectorKeywordIndex[each]]=tfidf
            except:
                continue

        return tfidf_vector

    def makeFeedbackVector(self, word):
        """make feedback vector"""

        wordList = self.parser.tokenise(word)
        wordList = self.parser.removeStopWords(wordList)

        result = nltk.pos_tag(wordList)
        fb=[]
        for each in result:
            if ('VB' in each[1] or 'NN' in each[1]):
                fb.append(each[0])
        # print(fb)
        return np.array(self.makeTfIdfVector(' '.join(fb)))*0.5


    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
        return vector

    def TF_Cosine(self, query):
        queryTFVector = [self.makeTfVector(query)]

        tf_cos = []
        for documentTFVector in self.documentTFVectors:
            tf_cos.append(util.cosine(queryTFVector, documentTFVector))
         
        return tf_cos

    def TFIDF_Cosine(self, query):
        queryTFIDFVector = [self.makeTfIdfVector(query)]

        tfidf_cos = []
        for documentTFIDFVector in self.documentTFIDFVectors:
            tfidf_cos.append(util.cosine(queryTFIDFVector, documentTFIDFVector))
         
        return tfidf_cos

    def TF_Euclidean(self, query):
        queryTFVector = self.makeTfVector(query)
        tf_eu = []
        for documentTFVector in self.documentTFVectors:
            tf_eu.append(util.eu(queryTFVector, documentTFVector))
        return tf_eu

    def TFIDF_Euclidean(self, query):
        queryTFIDFVector = self.makeTfIdfVector(query)
        tfidf_eu = []
        for documentTFIDFVector in self.documentTFIDFVectors:
            tfidf_eu.append(util.eu(queryTFIDFVector, documentTFIDFVector))
        return tfidf_eu

    # def buildQueryVector(self, termList):
    #     """ convert query string into a term vector """
    #     query = self.makeVector(" ".join(termList))
    #     return query


    # def related(self,documentId):
    #     """ find documents that are related to the document indexed by passed Id within the document Vectors"""
    #     ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
    #     #ratings.sort(reverse=True)
    #     return ratings


    # def search(self,searchList):
    #     """ search for documents that match based on a list of terms """
    #     queryVector = self.buildQueryVector(searchList)

    #     ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
    #     #ratings.sort(reverse=True)
    #     return ratings

###################################################
