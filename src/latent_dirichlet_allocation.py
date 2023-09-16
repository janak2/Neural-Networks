from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


# load data (TODO this only uses dataset in sklearn)
dataset = fetch_20newsgroups(categories= None, shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

print ("Data length: ", len(documents))
print(dataset.target_names)
#16
space_indices = dataset.target == 10
gun_indices = dataset.target == 11
# number of terms included in the bag of words
no_features = 10000

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()
#print ("Feature names", tf_feature_names)

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H[:10]):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                       for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        # for doc_index in top_doc_indices:
        #     print (documents[doc_index])


no_topics = 100

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=30.,random_state=1).fit(tf)

score = lda.score(tf)

print ("Score: ", score)
lda_W = lda.transform(tf)
print(lda_W.shape)
lda_H = lda.components_
print(lda_H.shape)

no_top_words = 10
no_top_documents = 1


# display_topics(lda_H, lda_W, tf_feature_names, documents, no_top_words, no_top_documents)

space_distribution = np.sum(lda_W[space_indices], axis=0).argsort()[:-4:-1]
gun_distribution = np.sum(lda_W[gun_indices], axis=0).argsort()[:-4:-1]

for topic_idx, topic in enumerate(lda_H[space_distribution]):
    print ("Topic %d:" % space_distribution[topic_idx])
    print (" ".join([tf_feature_names[i]
                   for i in topic.argsort()[:-10 - 1:-1]]))

for topic_idx, topic in enumerate(lda_H[gun_distribution]):
    print ("Topic %d:" % gun_distribution[topic_idx])
    print (" ".join([tf_feature_names[i]
                   for i in topic.argsort()[:-10 - 1:-1]]))