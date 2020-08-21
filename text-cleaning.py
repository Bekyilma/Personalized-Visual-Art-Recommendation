import spacy
import pandas as pd
import numpy as np
import re
import gensim
from gensim.corpora import Dictionary
from gensim import corpora
import pickle


"""This class is used to perform text cleaning on a painting description dataset. 
    Multiple Natural Language Processing (NLP) technqiues are used in order remove 
    meaningless information from the dataset."""

class PreprocessingText:
    
    
        path_to_dataset = 'resources/datasets/ng-dataset.csv'
        path_to_preprocessed = 'resources/datasets/preprocessed/'
        

        def clean_dataframe(self, df):
            """We apply a transformation to each row of the dataframe using the function replace_break_balise
            Input: 
                    df: dataframe of paintings
            Output:
                    df: dataframe were break lines are removed
            """
            pd.set_option("display.max_colwidth", 1000)
            df['merged_description'] = df['merged_description'].apply(self.replace_break_balise)
            return df
        
        
        def replace_break_balise(self, text):
            """From the index, returns the painting ID from the paintings dataframe
            Input: 
                    text
            Output:
                    text: breaklines + <p> </p> balises removed
            """
            text = text.replace('\n', '')
            text = text.replace('<p>', '')
            text = text.replace('</p>', '')
            return text


        def dataframe2text(self, df):
            """We transform the dataframe into a raw text format 
            Input: 
                    df: dataframe of paintings
            Output:
                    text
            """
            text = df['merged_description'].to_string(index=False)
            text = re.sub(' +', ' ', text).replace('<br>', '')
            text = text.replace('\n ', ' \n')
            return text


        def add_stopwords(self, text, nlp):
            """Add additional stopwords"""
            # Removing additional stopwords related to our context
            my_stop_words = ["'s", 'be', 'work', 'painting', 'early', 'small', 'know', 'appear', 'depict', 'tell', 'type', 'apparently', 'paint', 'show', 'probably', 'picture', 'left', 'right', 'date', 'suggest', 'hold', 'de', 'see', 'represent', 'paint']
            for stopword in my_stop_words:
                nlp.vocab[stopword].is_stop = True
            doc = nlp(text)
            return doc, nlp

        def clean_text(self, doc, nlp):
            """Remove stopwords, punctuation and numbers from the text"""
            # we add some words to the stop word list
            texts, article, skl_texts = [], [], []
            for w in doc:
                # if it's not a stop word or punctuation mark, add it to our article!
                if w.text != '\n' and not w.is_stop and not nlp.vocab[w.lemma_].is_stop and not w.is_punct and not w.like_num:
                    # we add the lematized version of the word
                    article.append(w.lemma_)
                # if it's a new line, it means we're onto our next document
                if w.text == '\n':
                    skl_texts.append(' '.join(article))
                    texts.append(article)
                    article = []
            for li in texts:
                if ' ' in li:
                    li.remove(' ')
            return texts


        # We wrap all the preprocessing in one function for future reusability
        def preprocess(self, text):
            nlp = spacy.load('en_core_web_sm')
            nlp.max_length = 10000000
            doc, nlp = self.add_stopwords(text, nlp)
            texts = self.clean_text(doc, nlp)
            bigram = gensim.models.Phrases(texts)
            list_words = [bigram[line] for line in texts]
            dictionary = Dictionary(texts)
            dictionary.filter_extremes(no_below=10, no_above=0.2)
            corpus = [dictionary.doc2bow(word) for word in list_words]
            return list_words, dictionary, corpus
        
        def save_listwords(self, list_words, path_to_preprocessed):
            with open(path_to_preprocessed+'list_words.txt', "wb") as fp:   #Pickling
                pickle.dump(list_words, fp)
                
                
        def save_dictionary(self, dictionary, path_to_preprocessed):
            dictionary.save(path_to_preprocessed+'dict')
            
        
        def save_corpus(self, corpus, path_to_preprocessed):
            corpora.MmCorpus.serialize(path_to_preprocessed+'corpus', corpus)


        def main(self):
            df = pd.read_csv(self.path_to_dataset)
            cleaned_df = self.clean_dataframe(df)
            text = self.dataframe2text(cleaned_df)
            list_words, dictionary, corpus = self.preprocess(text)
            self.save_listwords(list_words, self.path_to_preprocessed)
            self.save_dictionary(dictionary, self.path_to_preprocessed)
            self.save_corpus(corpus, self.path_to_preprocessed)
            return list_words, dictionary, corpus
    
    
if __name__ == "__main__":
    p = PreprocessingText()
    texts, dictionary, corpus = p.main()
    
