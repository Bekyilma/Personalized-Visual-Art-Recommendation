import gensim
import pandas as pd
import numpy as np
from gensim.models.wrappers import LdaMallet
import sys

""" This class is creates a list of n recommendations that are the most similar to a list of paintings
    liked by the user. It uses a Latent Dirichlet Allocation approach which expresses the paintings as 
     a distribution of topics. Topics are themselves a distribution of words. """

class QueryLdaModel:
    
        path_to_model = 'resources/models/lda.model'
        path_to_cos_mat = 'resources/matrices/lda/cosine-mat.npy'
        path_to_topdoc_mat = 'resources/matrices/lda/lda-output.npy'
        painting_df = pd.read_csv('resources/datasets/ng-dataset.csv')
        
        
        def __init__(self, painting_list, n):
            self.painting_list = painting_list
            self.n = n
        
        
        def load_model(self, path_to_model):
            """Load the LDA model"""
            lda_model = LdaMallet.load(path_to_model)
            return lda_model
        
        def load_cosine_matrix(self, path_to_cos_mat):
            """Load the cosine similarity matrix"""
            cos_sim_mat = np.load(path_to_cos_mat)
            return cos_sim_mat
        
        def load_topdoc_matrix(self, path_to_topdoc_mat):
            """Load the topic-document matrix"""
            topdoc_mat = np.load(path_to_topdoc_mat)
            return topdoc_mat
        
        
        def pid2index(self, painting_df, painting_id):
            """From the painting ID, returns the index of the painting in the painting dataframe
            Input: 
                    painting_df: dataframe of paintings
                    painting_list: list of paintings ID (e.g ['000-02T4-0000', '000-03WC-0000...'])
            Output:
                    index_list: list of the paintings indexes in the dataframe (e.g [32, 45, ...])
            """
            try: 
                index = painting_df.loc[painting_df['painting_id'] == painting_id].index[0]
            except IndexError as ie:
                index = "Painting ID '" + painting_id + "' not found in dataset."
            return index
        
        def pidlist2indexlist(self, painting_df, painting_list):
            """From a list of painting ID, returns the indexes of the paintings
            Input: 
                    painting_df: dataframe of paintings
                    painting_list: list of paintings ID (e.g ['000-02T4-0000', '000-03WC-0000...'])
            Output:
                    index_list: list of the paintings indexes in the dataframe (e.g [32, 45, ...])
            """
            index_list = [self.pid2index(painting_df, painting_id) for painting_id in painting_list]
            return index_list
        
        def index2pid(self, painting_df, index):
            """From the index, returns the painting ID from the paintings dataframe
            Input: 
                    painting_df: dataframe of paintings
                    index: index of the painting in the dataframe
            Output:
                    pid: return the painting ID (e.g: 000-02T4-0000 )
            """
            try: 
                pid = painting_df.iloc[index].painting_id
            except IndexError as ie:
                pid = "Index '" + index + "' not found in dataset."
            return pid
        
        def indexlist2pidlist(self, painting_df, index_list):
            """From a list of indexes, returns the painting IDs
            Input: 
                    painting_df: dataframe of paintings
                    index_list: list of the painting indexes in the dataframe
            Output:
                    pid: list of paintings ID 
            """
            pids_list = [self.index2pid(painting_df, index) for index in index_list]
            return pids_list
                
        
        def recommend_paintings(self, painting_df, painting_list, cos_mat, n):
            """Recommand paintings for a user based on a list of items that were liked
            Input: 
                    painting_df: dataframe of paintings
                    painting_list: list of paintings index liked by a user
                    cos_sim_mat: Cosine Similarity Matrix
                    n: number of recommendation wanted
            Output:
                    a list of indexes for recommended paintings 
            """
            n_painting = len(painting_list)
            score_list = []
            index_list = self.pidlist2indexlist(painting_df, painting_list)
            for index in index_list:
                score = cos_mat[index]
                score[index] = 0
                score_list.append(score)
            score_list = np.sum(score_list, 0)/n_painting 
            top_n_index = sorted(range(len(score_list)), key=lambda i: score_list[i], reverse=True)[:n]
            top_n_pids = self.indexlist2pidlist(painting_df, top_n_index)
            return top_n_pids
        
        
        def main(self):
            model = self.load_model(self.path_to_model)
            cos_mat = self.load_cosine_matrix(self.path_to_cos_mat)
            topdoc_mat = self.load_topdoc_matrix(self.path_to_topdoc_mat)
            pids_list = self.recommend_paintings(self.painting_df, self.painting_list, cos_mat, self.n)
            return pids_list
        
        
if __name__ == "__main__":
    lda = QueryLdaModel(['000-01DF-0000', '000-0168-0000', '000-019M-0000', '000-043Q-0000'], 10)
    pids_list = lda.main()
    print(pids_list)
    
        
        
        
        
        
        
  
