
# coding: utf-8

# In[1]:


# Preparando ambiente (importando bibliotecas e downloads...)

get_ipython().system('pip install nltk')
import nltk
nltk.download('rslp')
nltk.download('stopwords')
nltk.download('punkt')
import re
import pandas as pd


# In[36]:


df = pd.read_csv("C:\\Users\\Usuario\\Desktop\\Monografia\\dataset.csv", sep=",",encoding="ISO-8859-1")


# In[37]:


# Separacao dos dados por sentimento
df.groupby('sentiment').count()


# In[38]:


# Remove columns e create column
df.drop(columns=['id', 'text_en'], axis=1, inplace=True)
df['classification'] = df["sentiment"].replace(["neg", "pos"],[0, 1])

# Texto para minusculo
text_lower = [t.lower() for t in df['text_pt']]
df['text_pt'] = text_lower

df.head(5)


# In[27]:



# funcao para remover brackets
def remove_brackets(column):
    for x in range(1,len(column)):
        return(re.sub('[\[\]]','',repr(column)))


# In[7]:


get_ipython().run_cell_magic('time', '', "\nfrom nltk.tokenize import word_tokenize \nstop_words = nltk.corpus.stopwords.words('portuguese')\nstemmer = nltk.stem.RSLPStemmer()\n\n# Trabalhar com stemmer e stopwords da base de treinamento/teste\n\nfor x in range(0,len(df['text_pt'])):\n\n    # Remover as stop words do texto\n    word_tokens = word_tokenize(df['text_pt'][x]) \n    filtered_sentence = [w for w in word_tokens if not w in stop_words] \n    \n    # Remover sufixos \n    line=[]\n    text_tokenized = word_tokenize((remove_brackets(filtered_sentence)))\n    line =  [stemmer.stem(word) for word in text_tokenized]\n    df['text_pt'][x] = (remove_brackets(line))")


# In[28]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

# Regex para remover alguns valores do dataset  (simbolos, numeros...)
token = RegexpTokenizer(r'[a-zA-Z0-9]+')

# Cria o 'vetorizador' de acordo com os parametros abaixo
cv = CountVectorizer(lowercase=True,stop_words=None,ngram_range = (1,2),
                     tokenizer = token.tokenize)

# Matrixsparse da representação da coluna  text_pt
text_counts= cv.fit_transform(df['text_pt'])


# In[9]:



# Vocabulario
cv.vocabulary_


# In[29]:


# Importando biliotecas para selecao de amostra, modelo e avaliação do modelo.
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Divindo no dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(text_counts, df['classification'], 
                                                    test_size=0.34, random_state=1, 
                                                    shuffle=True)
# Criar modelo e treinar
clf = MultinomialNB().fit(X_train, y_train)

# Fazendo  predict do valor de X para teste de acuracidade
y_predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, y_predicted).round(3))


# In[42]:


with open('C:\\Users\\Usuario\\Desktop\\monografia\\texto_testei.txt', "r") as file_teste:
    phrase = file_teste.read().split('.')


# In[43]:


#Importar stemmer novamente
stemmer = nltk.stem.RSLPStemmer()

# Criar dataframe
df_result = pd.DataFrame()


# Fazer a tokanização, remocao de stop words e 
# transformar os dados para predict
neg,pos=0,0
for x in range(0,len(phrase)-1):

    # Texto tokenizado
    text_tokenized = word_tokenize(phrase[x])

    # Remove stop words do texto
    filtered_sentence = [w for w in text_tokenized if not w in stop_words] 

    # Cria stemmer do texto input
    line =  [stemmer.stem(word) for word in filtered_sentence]
    line = (remove_brackets(line))

    # Criar prediction para cada frase
    value_trans = cv.transform([line])
    predict_phrase = clf.predict(value_trans)

    # Contar por tipo de prediction (positivo e negativo)
    if predict_phrase==0:pos+=1
    else:neg+=1

# Salvar valores no dataframe
df_result['positive'] = [pos]
df_result['negative'] = [neg]


# In[44]:


def generate_piechart(df_result):
    
    import matplotlib.pyplot as plt
    labels = df_result.columns.tolist()
    sizes = df_result.values.tolist()[0]
    color = ['lightskyblue', 'lightcoral']
    explode = (0.15, 0)

    fig1, ax1 = plt.subplots(figsize=(5,5))
    ax1.pie(sizes, labels=labels,  explode=explode,
            shadow=True, autopct='%1.1f%%',  startangle=140, colors=color)

    ax1.set_title('Polaridade do Sentimento', fontsize=15)

    ax1.axis('equal')
    plt.show()
   
   # print("Quantity by phrases: {}".format(len(phrase)-1))
    print("Quantidade de sentenças positivas: {}".format(df_result['positive']
                                                     .values.tolist()[0]))
    
    print("Quantidade de sentenças negativas: {}".format(df_result['negative']
                                                     .values.tolist()[0]))
    
# Gerar gráfico    
generate_piechart(df_result)


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.sentiment.value_counts().plot(kind='bar')


# In[21]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_predicted, clf.predict(X_test)))


# In[22]:


print (pd.crosstab(y_predicted, clf.predict(X_test),
            rownames=['real'], colnames=['predito'], margins=True))


# In[45]:


print("gerando grafico")
    
generate_piechart(df_result)

df.sentiment.value_counts().plot(kind='bar')

print(confusion_matrix(y_predicted, clf.predict(X_test)))

print (pd.crosstab(y_predicted, clf.predict(X_test),rownames=['real'], colnames=['predito'], margins=True))

