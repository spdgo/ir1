# boolean
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
print("Demonstration of boolean Retrieval Model using Bitwise operationson Term Document Matrix of a corpus ")
corpus={
    'this is the first document',
    'this document is the second document',
    'And this is the third one',
    'Is this the first document'
    }
print("The corpus is :")  
print(corpus)
vectorizer=CountVectorizer()
x=vectorizer.fit_transform(corpus)
df=pd.DataFrame(x.toarray(),columns=vectorizer.get_feature_names())
print("The generated data frame")
print(df)
print("Query processing on term document incidence matrix:")
print("1.Find all document ids for query this AND first")
alldata=df[(df['this']==1)&(df['first']==1)]
print("Document ids where both 'this' and 'first' terms are present are :",alldata.index.tolist())
print("2.Find all document ids for query this OR first")
alldata=df[(df['this']==1)|(df['first']==1)]
print("Document ids where either 'this' or 'first' terms are present are :",alldata.index.tolist())
print("3.Find all document ids for query NOT AND")
alldata=df[(df['and']!=1)]
print("Document ids whereand terms is not present are :",alldata.index.tolist())
