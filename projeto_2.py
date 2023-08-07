import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Previsão de Renda",
     page_icon="https://travelpedia.com.br/wp-content/uploads/2018/09/dinheiro-icon.png",
)

st.write('# Análise exploratória da previsão de renda')

renda = pd.read_csv('./input/previsao_de_renda.csv')

#plots
fig, ax = plt.subplots(8,1,figsize=(10,70))
renda[['posse_de_imovel','renda']].plot(kind='hist', ax=ax[0])
st.write('## Gráficos ao longo do tempo')
sns.lineplot(x='data_ref',y='renda', hue='posse_de_imovel',data=renda, ax=ax[1])
ax[1].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='posse_de_veiculo',data=renda, ax=ax[2])
ax[2].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='qtd_filhos',data=renda, ax=ax[3])
ax[3].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_renda',data=renda, ax=ax[4])
ax[4].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='educacao',data=renda, ax=ax[5])
ax[5].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='estado_civil',data=renda, ax=ax[6])
ax[6].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_residencia',data=renda, ax=ax[7])
ax[7].tick_params(axis='x', rotation=45)
sns.despine()
st.pyplot(plt)

st.write('## Gráficos bivariada')
fig, ax = plt.subplots(7,1,figsize=(10,50))
sns.barplot(x='posse_de_imovel',y='renda',data=renda, ax=ax[0])
sns.barplot(x='posse_de_veiculo',y='renda',data=renda, ax=ax[1])
sns.barplot(x='qtd_filhos',y='renda',data=renda, ax=ax[2])
sns.barplot(x='tipo_renda',y='renda',data=renda, ax=ax[3])
sns.barplot(x='educacao',y='renda',data=renda, ax=ax[4])
sns.barplot(x='estado_civil',y='renda',data=renda, ax=ax[5])
sns.barplot(x='tipo_residencia',y='renda',data=renda, ax=ax[6])
sns.despine()
st.pyplot(plt)

st.write('## Modelo')

renda = (renda.drop('Unnamed: 0', axis=1)
        .assign(data_ref = lambda x: pd.to_datetime(x['data_ref']))
        .assign(qt_pessoas_residencia = lambda x: x['qt_pessoas_residencia'].astype(int))
        .fillna(0)
        .assign(tempo_emprego=lambda x: np.where(x['tempo_emprego'] < 0, 0, x['tempo_emprego']))
        )

train, test = train_test_split(renda, test_size=0.25, random_state=42)

modelo = '''
            renda ~ sexo
                        + tipo_renda
                        + tipo_renda
                        + idade
                        + tempo_emprego        
          '''
md = smf.ols(modelo, data = train)
reg = md.fit_regularized(method = 'elastic_net'
                         , refit = True
                         , L1_wt = 1
                         , alpha = 1)

train['resid'] = reg.resid
train['pred'] = reg.fittedvalues
st.text(reg.summary())

plt.close('all')
sns.scatterplot(x = 'renda', y = 'resid', data = train)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Resíduo')
st.pyplot(plt)

plt.close('all')
plt.plot(train['renda'], train['tempo_emprego'], '.')
plt.plot(train['pred'], train['tempo_emprego'], 'r.')
plt.title('Base de treino')
st.pyplot(plt)


test['pred'] = reg.predict(test)

plt.close('all')
plt.plot(test['renda'], test['tempo_emprego'], '.')
plt.plot(test['pred'], test['tempo_emprego'], 'r.')
plt.title('Base de teste')
st.pyplot(plt)





