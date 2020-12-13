
# coding: utf-8

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use ('ggplot')


# In[23]:


arquivo = pd.read_excel("planilha_de_repasse.xlsx")


# In[24]:


arquivo.head()


# In[25]:


arquivo.describe()


# In[26]:


# Criando nova coluna
arquivo['Conciliação'] = 'vazio'


# In[27]:



arquivo.head()


# In[28]:


#colsToDrop = ["Data do pedido","ID do pedido Magazine Luiza","ID do pedido Magazine Luiza","ID da transação","Número da nota fiscal","Nome do cliente","Parcela atual","Total de parcelas","Valor líquido da parcela","Valor da antecipação","% Taxa de antecipação","Valor bruto do pedido","Valor da comissão","Valor bruto seller","Origem","Observações","external_withdraw_id"]
#arquivo.drop(colsToDrop, axis=1)


# In[29]:


#renomeando as colunas
arquivo.columns=['data_transacao','id_pedido_seller','metodo_pagamento','comissao_ml_parcela','valor_bruto_parcela','comissão','conciliacao']


# In[30]:


arquivo.head()


# In[31]:


arquivo['metodo_pagamento'].replace(['Cartão de Crédito','Boleto'], '1')


# In[32]:


arquivo['metodo_pagamento'].replace(['Estorno'], '2')


# In[33]:


arquivo['metodo_pagamento'].replace(['Transferência'], '3')


# In[34]:


arquivo['metodo_pagamento'].replace(['NaN'], '0')


# In[35]:


arquivo['valor_bruto_parcela'].replace(['NaN'], '0')


# In[49]:


if(arquivo.metodo_pagamento==1 or arquivo.metodo_pagamento==2):
    arquivo["conciliacao"] = arquivo["valor_bruto_parcela"] * arquivo["comissao"] > 0, 'Conciliado', 'Nâo Conciliado'
else 
    arquivo["conciliacao"] = arquivo["valor_bruto_pedido"] * arquivo["comissao"] == arquivo["valor_liquido_parcela"], 'Conciliado', 'Nâo Conciliado'
     


# In[50]:


arquivo.describe()

       


# In[51]:


arquivo.count()


# In[54]:


pd.value_counts(arquivo['Conciliado'])


# In[ ]:


pd.value_counts(arquivo['Não Conciliado'])

