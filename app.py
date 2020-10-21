# ML Packages
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np
import scipy as sp
import pickle

#NLP
import re
import nltk
import numpy as np


from nltk.corpus import stopwords
english_stops = stopwords.words('english')

# machine learning modules
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# text processing modules

from nltk.stem.porter import PorterStemmer
import csv
from collections import Counter, OrderedDict
from flask import Flask,render_template,request,url_for
from sklearn.ensemble import RandomForestClassifier




model=pickle.load(open('model2.pkl','rb'))



def simple_tokenizer(text):
    #remove non letters
        text = re.sub("[^a-zA-Z]", " ", text)
        tokens = nltk.word_tokenize(text)    
        tokens = [elem for elem in tokens if (len(elem) > 2 and elem not in english_stops)]
        return tokens

app = Flask(__name__)

@app.route("/")
def index():
	return render_template("index.html")


        
data1=pd.read_csv('C:/Users/Suraj S/Downloads/camsonline sep2019-aug2020 - export-1 - 2020-08-27T212503.82.csv')
@app.route("/",methods=['POST'])
def predict():
    
    
    g=data1
    g['word_count']=g['Campaign Subject'].apply(lambda x : len(str(x).split(" ")))
    g[['Campaign Subject','word_count']].head()
    data1["Percentage_open"]=(data1['Open']/data1['Delivered'])*100
    f=np.percentile(data1["Percentage_open"], 25)
    data1["Open_Percentage"]=np.where(data1['Percentage_open']>=f,'1','0')
    count_vect = CountVectorizer(analyzer = 'word', ngram_range=(1,2),tokenizer=simple_tokenizer, lowercase=True)
    review_tf = count_vect.fit_transform(data1['Campaign Subject'])
    review_tf = review_tf.toarray()
    df = pd.DataFrame(review_tf, columns=count_vect.get_feature_names())
    # Merging td-idf weight matrix with original DataFrame
    model = pd.merge(data1, df, left_index=True, right_index=True)
    ml_model = model.drop(['Campaign Name', 'Campaign Subject', 'List', 'Segment', 'Sent','Delivered', 'Open', 'Click', 'Bounced ', 'Unique Open','Unsubscribe ','Spam ','VirtualSpam','SendDate','Percentage_open'], axis=1)

    # Create X & y variables for Machine Learning
    X = ml_model.drop('Open_Percentage', axis=1)
    y = ml_model['Open_Percentage']
    # Create a train-test split of these variables
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
    ran=RandomForestClassifier(n_estimators=40)
    ran.fit(X_train,y_train)
    predictions=ran.predict(X_test)
    # Compile arrays of columns (words) and feature importances
    # Compile arrays of columns (words) and feature importances
    fi = {'Words':ml_model.drop('Open_Percentage',axis=1).columns.tolist(),'Importance':ran.feature_importances_}

# Bung these into a dataframe, rank highest to lowest then slice top 20
    
    # Plot the graph!
    Importances = pd.DataFrame(fi).sort_values('Importance',ascending=False).head(10)  
    for i in range(len(Importances)): 
              print(Importances.iloc[i,0], Importances.iloc[i, 1])   
        
    
    def analysis():
        m=g.sort_values(by=['Open'],ascending=False)
        SendDate=m[["SendDate","Open"]]
        word_count=m[["Open","word_count","SendDate"]].head()
        
        return SendDate
        #return '<li class="list-group-item d-flex justify-content-between align-items-center"> ' , SendDate,'<span class="badge badge-primary badge-pill">',word_count,'</span></li>'
  
    return render_template('index.html',prediction_text='{}'.format(Importances))
        
        
        
if __name__ == '__main__':
	app.run(host="127.0.0.1",port=8080,debug=True)