import pandas as pd 
import re
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer , TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import numpy as np

train = pd.read_csv("train.csv")
#print("Training Set: {} \n Shape: {} \n columns: {}".format(len(train) , train.shape , train.columns))
test = pd.read_csv("test.csv")
#print("Testing Set: {} \n Shape: {} \n columns: {}".format(len(test) , test.shape , test.columns))


def clean_tweet(df , text_field):

	df[text_field] = df[text_field].str.lower()
	df[text_field] = df[text_field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "",elem))

	return df

train_clean = clean_tweet(train, 'tweet')
test_clean = clean_tweet(test, 'tweet')

#UPSAMPLING
train_majority = train_clean[train_clean.label==0]
train_minority = train_clean[train_clean.label==1]

train_minority_upsampled = resample(train_minority ,replace=True ,n_samples = len(train_majority) , random_state = 123)

train_upsampled = pd.concat([train_minority_upsampled , train_majority])
print(train_upsampled['label'].value_counts())

#downsampling
train_majority = train_clean[train_clean.label==0]
train_minority = train_clean[train_clean.label==1]

train_majority_downsampled = resample(train_majority, 
                                 replace=True,  
                                 n_samples=len(train_minority),   
                                 random_state=123)
train_downsampled = pd.concat([train_majority_downsampled, train_minority])
print(train_downsampled['label'].value_counts())


pipeline_sgd = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf',  TfidfTransformer()),
    ('nb', SGDClassifier(max_iter=1000)),
])



X_train , X_test , Y_train , Y_test = train_test_split(train_upsampled['tweet'] , train_upsampled['label'] , random_state=0)


model = pipeline_sgd.fit(X_train , Y_train)
y_predict =model.predict(X_test)
 
from sklearn.metrics import f1_score

print(f1_score(Y_test , y_predict))