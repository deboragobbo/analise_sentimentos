

import nltk
nltk.download('rslp')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import re
import pandas as pd
import matplotlib.pyplot as plt

def generate_piechart(df_result):

    labels = df_result.columns.tolist()
    sizes = df_result.values.tolist()[0]
    color = ['lightskyblue', 'lightcoral']
    explode = (0.15, 0)

    fig1, ax1 = plt.subplots(figsize=(5,5))
    ax1.pie(sizes, labels=labels,  explode=explode, shadow=True, autopct='%1.1f%%',  startangle=140, colors=color)

    ax1.set_title('Polaridade do Sentimento', fontsize=15)

    ax1.axis('equal')
    plt.show()

    print("Quantidade de sentencas positivas: {}".format(df_result['positive'].values.tolist()[0]))

    print("Quantidade de sentencas negativas: {}".format(df_result['negative'].values.tolist()[0]))


def remove_brackets(column):
    for x in range(1,len(column)):
        return(re.sub('[\[\]]','',repr(column)))

stop_words = nltk.corpus.stopwords.words('portuguese')
stemmer = nltk.stem.RSLPStemmer()

print("Lendo dados")

df = pd.read_csv("dataset.csv", sep=",",encoding="utf-8")

print("Agrupando dados")

df.groupby('sentiment').count()

#df.drop(columns=['id', 'text_en'], axis=1, inplace=True)

print("Removendo dados")

del df["id"]
del df["text_en"]

print("Quantificando classes")

df['classification'] = df["sentiment"].replace(["neg", "pos"],[0, 1])

text_lower = [t.lower() for t in df['text_pt']]
df['text_pt'] = text_lower

print("processando")

word_tokens = [word_tokenize(df['text_pt'][x]) for x in  range(0,len(df['text_pt']))]
filtered_sentence = [w for w in word_tokens if not w in stop_words]
line = []
text_tokenized = word_tokenize((remove_brackets(filtered_sentence)))
line =  [stemmer.stem(word) for word in text_tokenized]
for x in range(0,len(df['text_pt'])):
    df['text_pt'][x] = (remove_brackets(line))

#for x in range(0,len(df['text_pt'])):
#    word_tokens = word_tokenize(df['text_pt'][x])
#    filtered_sentence = [w for w in word_tokens if not w in stop_words]
#    line=[]
#    text_tokenized = word_tokenize((remove_brackets(filtered_sentence)))
#    line =  [stemmer.stem(word) for word in text_tokenized]
#    df['text_pt'][x] = (remove_brackets(line))

print("modelando")

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

cv = CountVectorizer(lowercase=True, stop_words=None, ngram_range = (1,2), tokenizer = token.tokenize)

text_counts= cv.fit_transform(df['text_pt'])

X_train, X_test, y_train, y_test = train_test_split(text_counts, df['classification'], test_size=0.34, random_state=1, shuffle=True)

clf = MultinomialNB().fit(X_train, y_train)

print("treinando")

y_predicted= clf.predict(X_test)

print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, y_predicted).round(3))

with open('texto_teste.txt', "r") as file_teste:
    phrase = file_teste.read().split('.')

stemmer = nltk.stem.RSLPStemmer()

df_result = pd.DataFrame()

neg,pos=0,0

print("processando 2")

line =  [stemmer.stem(word) for word in [w for w in [ word_tokenize(phrase[x]) for x in range(0,len(phrase)-1) ] if not w in stop_words]]
line = (remove_brackets(line))
value_trans = cv.transform([line])
predict_phrase = clf.predict(value_trans)
if predict_phrase==0:pos+=1
else:neg+=1
df_result['positive'] = [pos]
df_result['negative'] = [neg]

#for x in range(0,len(phrase)-1):
#    text_tokenized = word_tokenize(phrase[x])
#    filtered_sentence = [w for w in text_tokenized if not w in stop_words]
#    line =  [stemmer.stem(word) for word in filtered_sentence]
#    line = (remove_brackets(line))
#    value_trans = cv.transform([line])
#    predict_phrase = clf.predict(value_trans)
#    if predict_phrase==0:pos+=1
#    else:neg+=1
#    df_result['positive'] = [pos]
#    df_result['negative'] = [neg]

print("gerando grafico")
    
generate_piechart(df_result)

df.sentiment.value_counts().plot(kind='bar')

print(confusion_matrix(y_predicted, clf.predict(X_test)))

print (pd.crosstab(y_predicted, clf.predict(X_test),rownames=['real'], colnames=['predito'], margins=True))