(1)aim:- Write a program to demonstrate bitwise operation term document  incidence matrix.
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


(2) Implement Page Rank Algorithm.(pip install numpy pip install scipy pip install pandas pip install fraction and import all in shell)

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
*def editdistance(s1,s2):
*   rows=len(s1)+1
*   cols=len(s2)+1
*    matrix=[[0 for x in range(cols)] for x in range(rows)]
*    for i in range(1,rows):
*        matrix[i][0]=i
*        for i in range(1,cols):
*            matrix[0][i]=i
*            for i in range(1,cols):
*                for j in range(1,rows):
*                    if(s1[j-1]==s2[i-1]):
*                        cost=0;
*                    else:
*                       cost=1
*                        matrix[j][i]=min(matrix[j-1][i]+1,
*                                         matrix[j][i-1]+1,
*                                         matrix[j-1][i-1]+cost)
*    print("Calculation matrix of edit distance:")
*    for m in range (rows):
*        print(matrix[m])
*    print("Edit distance b/w these two strings '"+s1+"' and '"+s2+"' is:")
*    return matrix[j][i]
*s1=input("Enter the first string: ")
*s2=input("Enter the second string: ")
*print(editdistance(s1,s2))

(4)Aim: Write a program to Compute Similarity between two text documents.(pip install nltk pip install stopwords and in shell, import nltk , nltk.download('stopwords') ,nltk.download('punkt')  ]
from nltk.corpus import stopwords

*from nltk.tokenize import word_tokenize

*x=open("F:\Information Retrieval\doc1.txt","r").read()

*y=open("F:\Information Retrieval\doc2.txt","r").read()

*print("Doc1:",x)

*print("Doc2:",y)

x_list=word_tokenize(x)

*y_list=word_tokenize(y)

*print("Tokenizing...")

*print("Tokenizing doc1",x_list)

*print("Tokenizing doc2",y_list)


*sw=stopwords.words('english')

*print("Stop words",sw)

*l1=[];l2=[]

*x_set={w for w in x_list if not w is sw}

*y_set={w for w in y_list if not w is sw}

*print("Removing stop words...")

*print("Removing from doc1 ",x_set)

*print("Removing from doc2 ",y_set)

*rvector=x_set.union(y_set)

*for w in rvector:

*    if w in x_set:

*        l1.append(1)

*    else:

*        l1.append(0)

*    if w in y_set:

*        l2.append(1)

*    else:

*        l2.append(0)

*c=0

*for i in range(len(rvector)):

*c+=l1[i]*l2[i]

*cosine=c/float((sum(l1)sum(l2))*0.5)

*print("Computing Similarity between two text documents...")

*print("Similarity: ",cosine)

(7)Write a program for Pre-processing of a Text Document: stop word removal.  (pip install nltk pip install stopwords and in shell, import nltk , nltk.download('stopwords') ,nltk.download('punkt')  ]

in python shell
>>>import nltk 

>>>nltk.download (‘stopwords’) 

>>>nltk.download (‘punkt’)

>>>from nltk.corpus import stopwords 

>>>print(stopwords.words(‘english’))

7a}Program for Stop word Removal from a Text  (pip install nltk pip install stopwords and in shell, import nltk , nltk.download('stopwords') ,nltk.download('punkt')  ]

*from nltk.corpus import stopwords

*from nltk.tokenize import word_tokenize

*print("TEXT PRE- PROCESSING: STOP AND REMOVAL FROM TEXT")

*example_sent = "This is a sample sentence, showing off the stop words filtration"

*stop_words = set(stopwords.words('english'))

*word_tokens =word_tokenize(example_sent)

*filtered_sentence =[]

*for w in word_tokens:

*    if w not in stop_words:

*        filtered_sentence.append(w)

*print("The text : ")

*print(example_sent)

*print(word_tokens)

*print(filtered_sentence)


7b}Stopword removal operations in a file (pip install nltk pip install stopwords and in shell, import nltk , nltk.download('stopwords') ,nltk.download('punkt')  ]

*from nltk.corpus import stopwords

*from nltk.tokenize import word_tokenize

*print("TEXT PRE- PROCESSING: STOP AND REMOVAL FROM FILE CONTENT")

*stop_words = set(stopwords.words('english'))

*file1 = open("D:/WORD FILES/EH_P1.txt")

*line =file1.read()

*print("The File Content : ")

*words = line.split()

*print(words)

*for r in words:

*    if not r in stop_words:

*        appendFile =open('D:/WORD FILES/IR_P3-2.txt','a')

*        appendFile.write(" "+r)

*        appendFile.close()

*appendFile=open('D:/WORD FILES/IR_P3-2.txt')

*line=appendFile.read()

*print("THE FILE CONTENT AFTER STOPWORD REMOVAL : ")

*print(line)


(8)Write a program for mining Twitter to identify tweets for a specific period and identify 
trends and named entities. (pip install tweepy and import all)

*import tweepy

*consumer_key = 't8DqtMUPqJgzPTTdbp6M2qc12'

*consumer_secret = 'Zsl7I1LqTDts3t680eymS9s3s7GdQ91RUXYtZKDEfcAEG9D9Cy'

*access_token = '1023576532743217153-20yUpIX6uXuDxbTQS3AmhFbN6HkcC6'

*access_secret = '3oG0Uira2Uy23nGO38QsGN1KaGh1Sj5OZMu8CmNtOdp9P'

*auth = tweepy.OAuthHandler(consumer_key,consumer_secret)

*auth.set_access_token(access_token,access_secret)

*ap1=tweepy.API(auth)

*user=ap1.verify_credentials()

*print(user.name)

*name="nytimes"

*tweetCount=20

*print("printing tweets of new york times user\n")

*results=ap1.user_timeline(user_id=name,count=tweetCount)

*for tweet in results:
*    print(tweet.text)


*print("printing world trends\n\n")

*WORLD_WOE_ID=1

*US_WOE_ID=23424977

*world_trends=ap1.available_trends()[1]

*for tweet in world_trends:

*    print(world_trends)

*print("searching for a query:")

*query="modi"

*language="en"

*results=ap1.search_tweets(q=query,lang=language)

*for tweet in results:

*    print(tweet.user.screen_name, "Tweeted: ", tweet.text, tweet.created_at)

(9)Write a program to implement simple web crawler.(pip install requests  pip install bs4 and import all that)


*import requests

*from bs4 import BeautifulSoup

*url=("www.amazon.in")

*code=requests.get("https://"+url)

*plain=code.text


*s=BeautifulSoup(plain)

*for link in s.find_all('a'):

*    print(link.get('href'))


(10)Write a program to parse XML text, generate Web graph and compute topic specific page rank. (pip  install csv pip instll requests and import all )


*import csv

*import requests

*import xml.etree.ElementTree as ET

*def loadRSS():

*    url='https://www.mysitemapgenerator.com/?action=download&xmlfile=5078132_4.xml'

*    resp=requests.get(url)

*    with open('topnewsfeed.xml','wb') as f:

*        f.write(resp.content)

*def parseXML(xmlfile):

*    tree=ET.parse(xmlfile)

*    root=tree.getroot()

*    newsitems=[]

*    for item in root.findall('./channel/item'):

*        news={}

*        for child in item:

*            if child.tag=='{http://search.yahoo.com/mrss/}content':

*                news['media']=child.attrib['url']

*            else:

*                news[child.tag]=child.text.encode('utf8')

*        newsitems.append(news)

*    return newsitems


*def savetoCSV(newsitems,filename):

*    fields=['guid','title','pubDate','description','link','media']

*    with open(filename,'w') as csvfile:

*        writer=csv.DictWriter(csvfile,fieldnames=fields)

*        writer.writeheader()

*        writer.writerows(newsitems)

*def main():

*    loadRSS()

*    newsitems=parseXML('topnewsfeed.xml')

*    print(newsitems)

*    savetoCSV(newsitems,'topnews.csv')

*if __name__=="__main__":

*    main()

