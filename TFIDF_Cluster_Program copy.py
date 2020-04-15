import numpy as np
import pandas as pd
import random
import os
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import hstack
import time

class ClusterProgram():
    
    def __init__(self, *argv):
        """
        takes in a list of csv paths and concatenates them
        """
        print("\n***** ClusterProgram *****\n")
        
        #concatenates all datasets into one long list
        print('Reading in files...')
        output = []
        for file_path in argv:
            df = pd.read_csv(file_path)
            df.columns = ['id', 'words']
            output += list(df['words'])
        
        #gets rid of exact matches, changes all to lower case, and randomly shuffles list
        print('preprocessing terms...')
        output = list(set(output))
        output = [x.lower() for x in output]
        random.shuffle(output)
        

        #self.dataset is the whole training set
        self.dataset = output
        
        #self.Vectorizer is the type of Vectorizer we will use to transform the data
        self.Vectorizer = None 
        
        #Indicates whether to use a character count vectorizer or a word tfidf vectorizer
        self.indicator = ''
        
        #Will be used during clustering
        self.batches = []
        
        #Will be the final way we store the clusters
        self.clusters = {}
    
    #Function which asks the user whether there are many mispelled words in the dataset.  If there are
    #we will use the character based Vectorizer, if not we use the word-based vectorizer.
    def ask_about_indicator(self):
        indicator = self.indicator
        while indicator != 'y' and indicator != 'n':
            indicator = input("will you be trying to match many mis-spelled words? (y or n): ")

        if indicator == 'y':
            self.indicator = 'char'
        elif indicator == 'n':
            self.indicator = 'word'
    
    #Function which breaks down dataset into batches each with less than 100,000 terms by continuously
    #clustering the data into two groups, then splitting each of those groups ect.
    def get_smaller_clusters(self, batch):
        
        #base case
        if len(batch) < 100000:
            self.batches.append(batch)
            if len(self.batches)%25 == 0:
                print('data has been broken into', len(self.batches), 'batches ...')
                
        #recursive step
        else:
            features = self.Vectorizer.transform(batch)
            k_means = MiniBatchKMeans(n_clusters=2,batch_size=1000).fit(features)
    
            df = pd.DataFrame({'label':k_means.labels_ , 'terms': batch})
            cluster1 = list(df.loc[df['label'] == 0]['terms'])
            cluster2 = list(df.loc[df['label'] == 1]['terms'])
            
            self.get_smaller_clusters(cluster1)
            self.get_smaller_clusters(cluster2)
            
    #Function which initializes the vectorizer, fits it, and then calls get_smaller_clusters in order to
    #break the data set into smaller batches
    def update_batches(self):
        self.ask_about_indicator()
        
        if self.indicator == 'word':
            print('Vectorizing by word tfidf...')
            self.Vectorizer = TfidfVectorizer(ngram_range=(1,1))
            self.Vectorizer.fit(self.dataset)

        elif self.indicator == 'char':
            print('Vectorizing by character count...')
            self.Vectorizer = CountVectorizer(ngram_range=(4,4), binary=False, analyzer='char')
            self.Vectorizer.fit(self.dataset)
        
        print('breaking down data into batches...')
        self.get_smaller_clusters(self.dataset)
    
    
    #Once we have called update_batches, we call get_clusters which clusters each of the batches.
    #If the batch is less than 15 terms, it will leave it alone.  If it is more, it will cluster it
    #with a number centers proportional to the batch size, or a max of 1500 centers
    def get_clusters(self):

        batch_index = 0
        for batch in myCluster.batches:
            batch_index += 1
            print(batch_index)
            print('##################')
            if len(batch) <= 15:
                self.clusters[batch_index] = pd.DataFrame({'term':batch, 'label': np.zeros(len(batch))})
            else:
                n_clusters = math.ceil(len(batch)/15)
                if n_clusters > 1500:
                    n_clusters = 1500
                batch_size = math.floor(len(batch)/10)
                features = self.Vectorizer.transform(batch)
                print('clustering batch of size', len(batch), 'with', n_clusters, 'centers')
                start_time = time.time()
                k_means = MiniBatchKMeans(n_clusters=n_clusters,batch_size=batch_size).fit(features)
                print('fitting took', start_time - time.time(), 'seconds')
                print('')

                self.clusters[batch_index] = pd.DataFrame({'term': batch, 'label': k_means.labels_})
    
    #once we have clustered all the terms, find_other_terms_in_label takes in a term and 
    #outputs a list of all the terms it was clustered with
    def find_other_terms_in_label(self, term):
        for batch_num in self.clusters.keys():
            this_batch = self.clusters[batch_num]
            term_row = list(this_batch.loc[this_batch['term'] == term]['label'])
            if len(term_row) > 0:
                label = term_row[0]
                return list(this_batch.loc[this_batch['label'] == label]['term'])
        return None
                

        
        
#Function which returns the precision and recall of the ClusteringProgram
def evaluate(test_set_path, myCluster):
    fn = 0
    fp = 0
    tn = 0
    tp = 0
    
    
    test_df = pd.read_csv(test_set_path)
    test_df = test_df.drop(['Unnamed: 0'], axis=1)
    test_terms = test_df.columns
    test_matrix = np.array(test_df)
    
    for i in range(test_matrix.shape[0]):
        term1 = test_terms[i]
        i_terms = myCluster.find_other_terms_in_label(term1, myCluster.clusters)
        for j in range(test_matrix.shape[1]):
            if i > j:
                true_match_indicator = test_matrix[i][j]
                term2 = test_terms[j]
                predicted_match_indicator = 1*(term2 in i_terms)
                
                if true_match_indicator == 0 and predicted_match_indicator == 0:
                    tn += 1
                elif true_match_indicator == 0 and predicted_match_indicator == 1:
                    fp += 1
                elif true_match_indicator == 1 and predicted_match_indicator == 0:
                    print('')
                    print('false negative:')
                    print('term 1:', term1, 'term 2:', term2)
                    print('')
                    fn += 1
                elif true_match_indicator == 1 and predicted_match_indicator == 1:
                    print('')
                    print('true positive:')
                    print('term 1:', term1, 'term 2:', term2)
                    print('')
                    tp += 1
                    
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
                
    return precision, recall

#runs clustering program, finds the precision and recall on both test sets
myCluster = ClusterProgram('csvs/amicus_org_names.csv', 'csvs/bonica_org_names.csv')
myCluster.update_batches()
myCluster.get_clusters()
print('Results For Representative Test Set')
precision,recall = evaluate('csvs/outputs/testmatrix_labeled.csv', myCluster)
print('')
print('##################################')
print('precision:', precision)
print('recall:', recall)
print('')
print('Results For Match Dense Test Set')
print('')
print('Results For Match Dense Test Set')
precision,recall = evaluate('csvs/handcoded_test.csv.csv', myCluster)
print('')
print('##################################')
print('precision:', precision)
print('recall:', recall)
print('')
print('Results For Match Dense Test Set')
print('')
