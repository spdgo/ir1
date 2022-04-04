(1)aim:- Write a program to demonstrate bitwise operation.
*import pandas as pd

*from sklearn.feature_extraction.text import CountVectorizer
 
*print("Demonstration of boolean Retrieval Model using Bitwise operationson Term Document Matrix of a corpus ")

*corpus={
    'this is the first document',
    'this document is the second document',
    'And this is the third one',
    'Is this the first document' 
    }
    
*print("The corpus is :") 

*print(corpus)

*vectorizer=CountVectorizer()

*x=vectorizer.fit_transform(corpus)

*df=pd.DataFrame(x.toarray(),columns=vectorizer.get_feature_names())

*print("The generated data frame")

*print(df)

*print("Query processing on term document incidence matrix:")

*print("1.Find all document ids for query this AND first")

*alldata=df[(df['this']==1)&(df['first']==1)]

*print("Document ids where both 'this' and 'first' terms are present are :",alldata.index.tolist())

*print("2.Find all document ids for query this OR first")

#alldata=df[(df['this']==1) pipe sign (df['first']==1)]

*print("Document ids where either 'this' or 'first' terms are present are :",alldata.index.tolist())

*print("3.Find all document ids for query NOT AND")

*alldata=df[(df['and']!=1)]

*print("Document ids whereand terms is not present are :",alldata.index.tolist())


(2) Implement Page Rank Algorithm.

*import numpy as np

*import scipy as sc

*import pandas as pd

*from fractions import Fraction
*def float_format(vector, decimal):
*    return np.round((vector).astype(np.float), decimals=decimal)
 

*dp = Fraction(1,3)
 
*M = np.matrix([[0,0,1],
        [Fraction(1,2),0,0],
        [Fraction(1,2),1,0]])
 
*E= np.zeros((3,3))
*E[:] = dp
 
*d = 0.8

*A = d * M + ((1-d) * E)

*r = np.matrix([dp, dp, dp])
*r = np.transpose(r)
*previous_r = r
*for it in range(1,100):
*    r = A * r
*    print(float_format(r,3))
  
*    if (previous_r==r).all():
*        break
*    previous_r = r
*print ("Final:\n", float_format(r,3))
*print ("sum", np.sum(r))


(3)Implement Dynamic programming algorithm for computing the edit distance between strings s1 and s2. (Hint. Levenshtein Distance)
def editdistance(s1,s2):
    rows=len(s1)+1
    cols=len(s2)+1
    matrix=[[0 for x in range(cols)] for x in range(rows)]
    for i in range(1,rows):
        matrix[i][0]=i
        for i in range(1,cols):
            matrix[0][i]=i
            for i in range(1,cols):
                for j in range(1,rows):
                    if(s1[j-1]==s2[i-1]):
                        cost=0;
                    else:
                        cost=1
                        matrix[j][i]=min(matrix[j-1][i]+1,
                                         matrix[j][i-1]+1,
                                         matrix[j-1][i-1]+cost)
    print("Calculation matrix of edit distance:")
    for m in range (rows):
        print(matrix[m])
    print("Edit distance b/w these two strings '"+s1+"' and '"+s2+"' is:")
    return matrix[j][i]
s1=input("Enter the first string: ")
s2=input("Enter the second string: ")
print(editdistance(s1,s2))
