import pandas as pd
import numpy as np
from gensim.models.wrappers import LdaMallet
from sklearn.metrics.pairwise import cosine_similarity
from gensim.corpora import Dictionary
from gensim import corpora
import pickle
import os


"""This class trains the Latent Dirichlet Allocation (LDA) Model on 
   painting description corpus.,
   we want to compare the paintings by computing a similarity measure : cosine similarity"""

class LdaTraining:
    
    
        path_to_mallet_bin = "/resources/mallet-2.0.6/bin/mallet" #path has to be absolute
        os.environ['MALLET_HOME'] = "/resources/mallet-2.0.6/" #path has to be absolute
        path_save_score = 'resources/datasets/'
        path_save_outputs = 'resources/matrices/lda/'
        path_save_model = 'resources/models/'
        path_to_listwords = 'resources/datasets/preprocessed/list_words.txt'
        path_to_dict = 'resources/datasets/preprocessed/dict'
        path_to_corpus = 'resources/datasets/preprocessed/corpus'
        painting_df = pd.read_csv('resources/datasets/ng-dataset.csv')
     
        def __init__(self, num_topics):
            self.num_topics = num_topics
                        
        
        def load_list_words(self, path_to_listwords):
            """Load the list of words"""
            with open(path_to_listwords, "rb") as fp:   # Unpickling
                list_words = pickle.load(fp)
            return list_words
            
        
        def load_dictionary(self, path_to_dict):
            """Load the dictionary"""
            dictionary = Dictionary.load(path_to_dict)
            return dictionary
        
        
        def load_corpus(self, path_to_corpus):
            """Load the corpus"""
            corpus = corpora.MmCorpus(path_to_corpus)
            return corpus
            
            
        def LdaModel(self, num_topics, corpus, dictionary):
            """Create a LDA topic model
            Input:
                    num_topics: number of topics for the model
                    corpus: gensim corpus
                    ditionary: gensim dictionary
            Output:
                    lda_model: a topic model using Latent Dirichlet Allocation (LDA)
            """
            lda_model = LdaMallet(mallet_path=self.path_to_mallet_bin, num_topics=num_topics, corpus=corpus, id2word=dictionary, random_seed=123)
            return lda_model
        
        
        def transform_output(self, lda_model, corpus):
            """Transform the topic document matrix into an ordered array of topic distribution
            Input:
                    lda_model: LDA model
                    corpus: gensim corpus
            Output:
                    lda_model: a topic model using Latent Dirichlet Allocation (LDA)
            """
            topdoc_mat = lda_model[corpus]
            topdoc_sorted = self.sort_tuples(topdoc_mat)
            lda_output = np.asarray(topdoc_sorted)
            return lda_output
        
        
        def sort_tuples(self, topdoc_mat):
            """Sort the tuples (topic, distribution) in a numeric ascending order and drop the topic index 
            [(3,0.02), (1, 0.1), (2,0.03), ...] => [(1, 0.1), (2, 0.03), (3,0.02), ...] => [0.1, 0.03, 0.02]
            Input:
                    topdoc_mat: matrix topic distribution / document
            Output:
                    sorted tuples with index removed
            """
            # Reordering the topics in ascending order (0,1,2,3...) so we can compare them using a similarity metrics
            for i in range(len(topdoc_mat)):
                topdoc_mat[i] = sorted(topdoc_mat[i], key=lambda tup: (tup[0], tup[1]))
                for j in range(len(topdoc_mat[i])):
                    topdoc_mat[i][j] = topdoc_mat[i][j][1]
            return topdoc_mat
        
        
        def save_output(self, lda_output, path_save_outputs):
            np.save(path_save_outputs+'lda-output', lda_output)
        
        
        def save_cosine(self, cos_mat, path_save_outputs):
            np.save(path_save_outputs+'cosine-mat', cos_mat)
            
        
        def save_pairwise_score(self, painting_df, cos_mat, path_save_score):
            list_tuples = []
            for i, list_score in enumerate(cos_mat):
                for k, score in enumerate(list_score):
                    list_tuples.append((i, k, score))
            sim_df = pd.DataFrame(list_tuples).rename(columns={0: 'painting_1', 1: 'painting_2', 2:'score'})
            sim_df['painting_1'] = sim_df['painting_1'].apply(lambda x: painting_df.iloc[x].painting_id)
            sim_df['painting_2'] = sim_df['painting_2'].apply(lambda x: painting_df.iloc[x].painting_id)
            sim_df = sim_df.loc[sim_df['painting_1'] != sim_df['painting_2']]
            #sim_df.to_csv(path_save_score+'lda-scores')
            sim_df.to_csv('C:/Users/aghenda/Documents/Datasets/lda-scores.csv')
            
        
        def main(self):
            list_words = self.load_list_words(self.path_to_listwords)
            dictionary = self.load_dictionary(self.path_to_dict)
            corpus = self.load_corpus(self.path_to_corpus)
            lda_model = self.LdaModel(self.num_topics, corpus, dictionary)
            lda_model.save(self.path_save_model+'lda.model')
            lda_output = self.transform_output(lda_model, corpus)
            self.save_output(lda_output, self.path_save_outputs)
            cos_mat = cosine_similarity(lda_output)
            self.save_cosine(cos_mat, self.path_save_outputs)
            self.save_pairwise_score(self.painting_df, cos_mat, self.path_save_score)
            
            
if __name__=='__main__':
    lda = LdaTraining(10)
    lda.main()
            
        
        
            
        
    
    
