from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


# load data (TODO this only uses dataset in sklearn)
dataset = fetch_20newsgroups(categories= None, shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

print ("Data length: ", len(documents))

# number of terms included in the bag of words
no_features = 10000

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()
#print ("Feature names", tf_feature_names)

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
	i = 0
	for topic_idx, topic in enumerate(H):
		if i==3:
			break
		i+=1
		print ("Topic %d:" % (topic_idx))
		print (" ".join([feature_names[i]
						for i in topic.argsort()[:-no_top_words - 1:-1]]))
		#top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
		# for doc_index in top_doc_indices:
		#     print (documents[doc_index])

for no_topics in range(40,50,10):
# Run LDA
	lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

	score = lda.score(tf)
	print ("#topic:", no_topics)
	print ("Score: ", score)
	lda_W = lda.transform(tf)
	lda_H = lda.components_


	no_top_words = 20
	no_top_documents = 1

	display_topics(lda_H, lda_W, tf_feature_names, documents, no_top_words, no_top_documents)
