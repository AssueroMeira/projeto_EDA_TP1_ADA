# projeto_EDA_TP1_ADA
# **Análise das precipitações e temperatura em Fortaleza/Ceará.**

![imagem-1](https://github.com/AssueroMeira/projeto_EDA_TP1_ADA/assets/39604515/9b53660e-3e30-440c-8212-34d33cc525d0)
Imagem gerada com auxílio do ChatGPT.

# **Introdução**

Este é um projeto de análise exploratória e visualização de dados sobre a série histórica de temperatura e precipitações de chuva, em Fortaleza, capital do Ceará. 
O presente trabalho é parte da conclusão da disciplina de Práticas de Programação 1, da trilha de Data Science, ministrada pelo prof. Jorge Chamby-Diaz, [ADA Tech](https://ada.tech/), e promovida pelo Santander (Programa: Santander Coders).

# Ferramentas, Habilidades e Conceitos aplicados neste projeto:


*   Python;
*   Pandas;

*   Matplotlib;
*   Seaborn;

*   Storytelling;
*   Descrição das variáveis;

*   Limpeza da base de dados;
*   Análise Univarida e Multivariada;

*   Medidas estatísticas;
*   Comparações;

*   Tendências;
*   Gráficos.

# **Sobre o Dataset**

A base de dados de dados foi retirada da plataforma [Kaggle](https://www.kaggle.com/datasets/paulogladson/clima-fortaleza). A referida base foi importada do BDMEP, que briga dados meteorológicos diários em forma digital, de séries históricas das várias estações meteorológicas convencionais da rede de estações do INMET com milhões de informações, referentes às medições diárias, de acordo com as normas técnicas internacionais da Organização Meteorológica Mundial. (https://portal.inmet.gov.br/dadoshistoricos) Dados APENAS de Fortaleza, dentre os quais, tem-se:
 'Data',
 'Hora_UTC',
 'PRECIPITAÇÃO_TOTAL__HORÁRIO_(mm)',
 'PRESSAO_ATMOSFERICA_AO_NIVEL_DA_ESTACAO__HORARIA_(mB)',
 'PRESSÃO_ATMOSFERICA_MAX.NA_HORA_ANT._(AUT)_(mB)',
 'PRESSÃO_ATMOSFERICA_MIN._NA_HORA_ANT._(AUT)_(mB)',
 'RADIACAO_GLOBAL_(Kj/m²)',
 'TEMPERATURA_DO_AR_-_BULBO_SECO__HORARIA_(°C)',
 'TEMPERATURA_DO_PONTO_DE_ORVALHO_(°C)',
 'TEMPERATURA_MÁXIMA_NA_HORA_ANT._(AUT)_(°C)',
 'TEMPERATURA_MÍNIMA_NA_HORA_ANT._(AUT)_(°C)',
 'TEMPERATURA_ORVALHO_MAX._NA_HORA_ANT._(AUT)_(°C)',
 'TEMPERATURA_ORVALHO_MIN._NA_HORA_ANT._(AUT)_(°C)',
 'UMIDADE_REL._MAX._NA_HORA_ANT._(AUT)_(%)',
 'UMIDADE_REL._MIN._NA_HORA_ANT._(AUT)_(%)',
 'UMIDADE_RELATIVA_DO_AR__HORARIA_(%)',
 'VENTO__DIREÇÃO_HORARIA_(gr)_(°_(gr))',
 'VENTO__RAJADA_MAXIMA_(m/s)',
 'VENTO__VELOCIDADE_HORARIA_(m/s)',
 'municipio',
 'RADIACAO_GLOBAL_(KJ/m²)',
 'Diaano',
 'TEMPERATURA_MEDIA',
 'UMIDADE_MEDIA',
 'date_month',
 'date_year',
 'date_day'

 ## A informação sobre as colunas da tabela, foram extraídas conforme código abaixo:

```python
# Importando as bibliotecas:
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress

# Salvando o caminho do arquivo em uma variável.
arquivo = '/content/FORTALEZA.csv' # Certifique-se de carregar o arquivo e/ou atualizar este caminho, a critério de onde e como este código será executado.

# Lendo o arquivo para um Dataframe. O uso do argumento 'sep=' é necessário pois os arquivos no Brazil costumam usar ';' como separador.
dfOriginal = pd.read_csv(arquivo, sep = ';') 

# Para facilitar a visualização criei uma variável para salvar as colunas do dataframe em uma lista.
listaNomeColunas = dfOriginal.columns.tolist()

listaNomeColunas
```
# **Questões de negócio.**
#### 1. Fortaleza é uma cidade com temperaturas constantes durante dos anos, com temperaturas entre 24 e 32° C. Execelente para férias.
#### 2. Ultimamente a sensação de calor tem aumentado na cidade de Fortaleza, o que pode ser consequência de uma tendência de alteração climática.


*   Quais as tendências de temperatura ao longo do ano?
*   Existe alguma correlação entre temperatura e precipitação?
*   Quais foram os dias mais quentes e mais frios registrados?
*   Existem evidências de mudanças climáticas no local ao longo do tempo?
*   Modelo preditivo (K-NN) para prever possibilidade de chuva.

# **Transformação dos Dados.**

#### O primeiro passo para esta etapa de transformação dos dados foi extrair mais informação da base e entender melhor a estrutura dos dados. Neste sentido, usou-se os comandos abaixo:

```python
print(dfOriginal.info())
```

#### Em seguida, obtivemos uma visão geral do Dataframe.

```python
print(dfOriginal.info())
```

#### Esta primeira análise retorna algumas informações importantes:


1.   Total de 28 colunas;
2.   183672 entradas;
3.   A coluna 'Hora_UTC' (index 2) possui valores em formato distinto. Iniciando com entradas, 'hh:mm', e, em seguida, dos dados passam a ter outra formatação, 'hhmm UTC';
4.   A coluna 'Data' (index 1) está como 'object'. Sendo necessário um casting para datetime, possibilitando um trabalho mais fácil com as datas.
5.   A coluna 'RADIACAO_GLOBAL_(Kj/m²)' (index 7) e 'RADIACAO_GLOBAL_(KJ/m²)' (index 21) possuem valores nulos e valores imprecisos quando comparadas.
6.   A coluna 'TEMPERATURA_MEDIA' (index 23) possui valores não padronizados, sendo necessário refazer o cálculo e arredondar as casas decimais para o limite de 2 casas.
7.   O nome das colunas precisam de edição para melhor a leitura e entendimento.

# **Limpeza e tratamento dos dados.**

```python
# Tratar a coluna Hora_UTC para deixar os valores no mesmo formato

def tratamentoDeColunas(dataFrame):
    'Função realiza o tratamento dos dados e retorna um novo arquivo .csv.'
    # A análise da coluna Data mostrou que os dados são do tipo Object e estão
    # em formatos diferentes. Parte dos dados, no arquivo original está no
    # formato 'hh:mm' e outra parte foi digitada como 'hhmmm UTC'. A estratégia
    # foi assegurar que a coluna 'Data' esteja no formato datetime e padronizar
    # a formatação dos dados em 'hh:mm'.

    # Assegurando que a coluna 'Data' está no formato datetime
    dataFrame['Data'] = pd.to_datetime(dataFrame['Data'])

    # Criando a máscara para identificar as linhas a partir de 01/01/2019
    mascara1 = dataFrame['Data'] >= '2019-01-01'

    # Aplicando a máscara e modificando os valores da coluna 'Hora_UTC'
    # A função lambda é utilizada para garantir que a operação de string seja aplicada corretamente
    dataFrame.loc[mascara1, 'Hora_UTC'] = dataFrame.loc[mascara1, 'Hora_UTC'].apply(lambda x: x.replace(' UTC', '').strip()[:2] + ':' + x.strip()[2:4] if 'UTC' in x else x)

    # A outra parte do tratamento recai sobre as colunas 'RADIACAO_GLOBAL_(Kj/mÂ²)'
    # e 'RADIACAO_GLOBAL_(KJ/mÂ²)', que são semelhantes, porém com dados células
    # vazias e dados divergentes. Para garantir uma melhor análise, essas colunas
    # serão retiradas, pois, pode ter ocorrido falhas na inserção dos dados.

    # Retirando as colunas 'RADIACAO_GLOBAL_(Kj/mÂ²)' e 'RADIACAO_GLOBAL_(KJ/mÂ²)'

    dataFrame = dataFrame.drop(labels=['RADIACAO_GLOBAL_(Kj/m²)', 'RADIACAO_GLOBAL_(KJ/m²)'], axis=1)

    # Outro tratamento necessário foi na coluna 'TEMPERATURA_MEDIA' que possui
    # resultados em formatos diversos.

    # Padronizando os valores da coluna 'TEMPERATURA_MEDIA'

    dataFrame['TEMPERATURA_MEDIA'] = (dataFrame['TEMPERATURA_MÁXIMA_NA_HORA_ANT._(AUT)_(°C)'] + dataFrame['TEMPERATURA_MÍNIMA_NA_HORA_ANT._(AUT)_(°C)']) / 2

    # Para padronizar o resultado, arredonda-se o resultado para 2 casas decimais.

    dataFrame['TEMPERATURA_MEDIA'] = dataFrame['TEMPERATURA_MEDIA'].round(2)

    # Editando o nome das colunas para melhorar a análise e o entendimento.
    dataFrame.columns = ['Id', 'Data', 'Hora', 'Preciptação total (mm)', 'Pressão atmosférica (mB)', 'Pressão Atm. (MAX) (mB)', 'Pressão Atm. (MIN) (mB)',
                         'Temperatura do ar (°C)', 'Temp. do ponto de orvalho (°C)', 'Temp. (MAX) (°C)', 'Temp. (MIN) (°C)', 'Temp. Orvalho (MAX) (°C)',
                         'Temp. Orvalho (MIN) (°C)', 'Umidade Rel. (MAX) (%)', 'Umidade Rel. (MIN) (%)', 'Umidade Rel. Ar Horária (%)',
                         'Direção Horária do vento (gr)', 'Rajada máxima de vento (m/s)', 'Velocidade Horária do Vento (m/s)', 'Município',
                         'Dia do ano', 'Temperatura média', 'Umidade média', 'Mês', 'Ano', 'Dia']

    # Definindo o caminho do novo arquivo
    caminhoNovoArquivo = '/content/FORTALEZA_MODIFICADO.csv'

    # Salvando o DataFrame modificado
    dataFrame.to_csv(caminhoNovoArquivo, encoding='utf-8-sig', index=False)

    # Retornando o caminho do novo arquivo
    return caminhoNovoArquivo
```

##### **Visualização do arquivo modificado.**

```python
dfModificado1 = pd.read_csv(tratamentoDeColunas(dfOriginal))

dfModificado1
```

# **Análise dos Dados.**

#### As questões levantadas sobre os dados buscam respostas dentro da variação de temperatura e chuvas na cidade de Fortaleza. Sendo assim, a análise dos dados se inicia com a identificação de Outliers para as colunas: Temperatura do ar (°C)', 'Temperatura média', 'Preciptação total (mm)':

```python
# Converter a coluna 'Data' para datetime
dfModificado1['Data'] = pd.to_datetime(dfModificado1['Data'])

# Função para detectar e retornar outliers
def detectarOutliers(dataFrame, coluna):
  Q1 = dataFrame[coluna].quantile(0.25)
  Q3 = dataFrame[coluna].quantile(0.75)
  IQR = Q3 - Q1
  limite_inferior = Q1 - 1.5 * IQR
  limite_superior = Q3 + 1.5 * IQR
  outliers = dataFrame[(dataFrame[coluna] < limite_inferior) | (dataFrame[coluna] > limite_superior)]
  return outliers[['Data', coluna]]

# Detectando outliers para as colunas especificadas
for coluna in ['Temperatura do ar (°C)', 'Temperatura média', 'Preciptação total (mm)']:
  outliers = detectarOutliers(dfModificado1, coluna)
  print(f"Outliers na coluna {coluna}:")
  print(outliers, '\n')

# Plotando os boxplots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(y=dfModificado1['Temperatura do ar (°C)'], orient='v')
plt.title('Temperatura do ar')
plt.savefig('/content/boxplot_temperatura_do_ar.png')  # Salvando o gráfico

plt.subplot(1, 3, 2)
sns.boxplot(y=dfModificado1['Temperatura média'], orient='v')
plt.title('Temperatura média')
plt.savefig('/content/boxplot_temperatura_media.png')  # Salvando o gráfico

plt.subplot(1, 3, 3)
sns.boxplot(y=dfModificado1['Preciptação total (mm)'], orient='v')
plt.title('Precipitação total')
plt.savefig('/content/boxplot_precp_total.png')  # Salvando o gráfico

plt.tight_layout()
plt.show()
plt.close()  # Fechando a figura para começar uma nova
```

![boxplot_precp_total](https://github.com/AssueroMeira/projeto_EDA_TP1_ADA/assets/39604515/6726ba22-8e51-4762-824d-843ac9412b64)

#### O gráfico 'Precipitação total' tem muitos vaores próximos a zero no espaço interquartil, justificando o formato do gráfico. Esse dados faz sentido, pois em Fortaleza, como em muitas cidade do Nordeste a precipitação é muita baixa, praticamente zero durante a maior parte do ano. Logo abaixo, o código retorna estatísticas descritivas sobre esta coluna, comprovando essa hipótese.

```python
# Este código mostra os valores mínimos e máximos da coluna, bem como a média, e os quartis.
dfModificado1['Preciptação total (mm)'].describe()
```

## **Tendências de temperatura ao longo do ano.**
#### Agora, vamos observar qual a tendência de temperatura ao longo do ano, dentro de uma série temporal.

```python
def tendenciaDeTemperaturaAoLongoDoAno(dataFrame):
  'Função que retornar um dataframe e um gráfico com a tendência.'
  # Calcula a média e o desvio padrão anual da temperatura.
  estatisticas_anuais = dataFrame.groupby('Ano')['Temperatura do ar (°C)'].agg(['mean', 'std']).reset_index()

  # Calculando a média ± desvio padrão para visualização
  estatisticas_anuais['mean_plus_std'] = estatisticas_anuais['mean'] + estatisticas_anuais['std']
  estatisticas_anuais['mean_minus_std'] = estatisticas_anuais['mean'] - estatisticas_anuais['std']

  # Plotando a tendência da temperatura média anual e as linhas de desvio padrão
  plt.figure(figsize=(12, 6))
  plt.plot(estatisticas_anuais['Ano'], estatisticas_anuais['mean'], marker='o', linestyle='-', color='blue', label='Média Anual')
  plt.fill_between(estatisticas_anuais['Ano'], estatisticas_anuais['mean_minus_std'], estatisticas_anuais['mean_plus_std'], color='blue', alpha=0.2, label='Desvio Padrão')

  plt.title('Tendência da Temperatura Média Anual com Desvio Padrão')
  plt.xlabel('Ano')
  plt.ylabel('Temperatura Média (°C)')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.savefig('/content/serie_temp_media_anual.png')  # Salvando o gráfico
  plt.show()
  plt.close()  # Fechando a figura para começar uma nova

  return estatisticas_anuais

media_mensal = tendenciaDeTemperaturaAoLongoDoAno(dfModificado1)
```

![serie_temp_media_anual](https://github.com/AssueroMeira/projeto_EDA_TP1_ADA/assets/39604515/35a8be22-6b32-4af5-a713-a3ac1314fb51)

#### A temperatura ao longo dos anos mostra uma tendência estável na cidade de Fortaleza, com registros em torno de 27°C em todos os anos. Contudo, percebe-se uma certa mudança, após 2020. O que parece indicar uma alteração climática.
#### Em todos esses anos, o dia mais quente foi 31/01/2024 com 33.8°C e o dia mais frio foi 31/07/2008 com 20.55°C. O código abaixo retorna esse resultado.

```python
def diasExtremosTemperatura(dataFrame, colunaTemperatura='Temperatura média', colunaData='Data'):
  """
  Função para encontrar os dias mais quentes e mais frios registrados.

  Parâmetros:
  - dataFrame: DataFrame contendo os dados climáticos.
  - colunaTemperatura: Nome da coluna que contém os valores de temperatura.
  - colunaData: Nome da coluna que contém as datas.

  Retorna:
  - Uma tupla contendo informações sobre o dia mais quente e o dia mais frio.
  """

  # Encontrar o valor máximo de temperatura e a data correspondente
  tempMax = dataFrame[colunaTemperatura].max()
  diaMaisQuente = dataFrame[dataFrame[colunaTemperatura] == tempMax][colunaData].iloc[0]

  # Encontrar o valor mínimo de temperatura e a data correspondente
  tempMin = dataFrame[colunaTemperatura].min()
  diaMaisFrio = dataFrame[dataFrame[colunaTemperatura] == tempMin][colunaData].iloc[0]

  return (f"O dia mais quente foi {diaMaisQuente} com {tempMax}°C",
          f"O dia mais frio foi {diaMaisFrio} com {tempMin}°C")

resultado = diasExtremosTemperatura(dfModificado1)
print(resultado[0])
print(resultado[1])
```

## Pensando na relação entre chuva e temperatura. Existe alguma correlação entre a temperatura e as precipitações?

```python
def CorrelacaoTemperaturaPrecipitacao(dataFrame):

  # Calcular o coeficiente de correlação de Pearson
  coeficienteCorrelacao = dataFrame['Temperatura média'].corr(dataFrame['Preciptação total (mm)'])
  print(f"Coeficiente de correlação de Pearson entre temperatura e precipitação: {coeficienteCorrelacao}")

  # Visualizar a relação com um gráfico de dispersão
  plt.figure(figsize = (10, 6))
  sns.scatterplot(data = dataFrame, x='Temperatura média', y='Preciptação total (mm)', alpha=0.6)
  plt.title('Relação entre Temperatura e Precipitação')
  plt.xlabel('Temperatura Média (°C)')
  plt.ylabel('Precipitação Total (mm)')
  plt.grid(True)
  plt.savefig('/content/rel_temp_prec.png')  # Salvando o gráfico
  plt.show()
  plt.close()  # Fechando a figura para começar uma nova

  return coeficienteCorrelacao

CorrelacaoTemperaturaPrecipitacao(dfModificado1)
```

#### Coeficiente de correlação de Pearson entre temperatura e precipitação: -0.14615613112226258

![rel_temp_prec](https://github.com/AssueroMeira/projeto_EDA_TP1_ADA/assets/39604515/7f1f9b6d-c56c-4bb6-b088-57bde5cf90c5)

#### O coeficiente de Pearson, que avalia a correlação entre duas variáveis, mostra uma correlação negativa entre temperatura e precipitação. Sendo assim, quanto menor a temperatura, maior a precipitação. Contudo, observa-se que essa correlação não é tão forte assim, onde a precipitação deve depender de outras variáveis.

## Após observar o comportamento da temperatura ao longo do tempo e sua relação com as precipitações. É possível analisar se há uma tendência de mudança climática na cidade de Fortaleza?

```python
# Converter a coluna 'Data' para datetime
dfModificado1['Data'] = pd.to_datetime(dfModificado1['Data'])

# Extrair o ano da coluna 'Data'
dfModificado1['Ano'] = dfModificado1['Data'].dt.year

# Calcular a média anual da temperatura
media_anual_temperatura = dfModificado1.groupby('Ano')['Temperatura do ar (°C)'].mean().reset_index()

# Realizar regressão linear
slope, intercept, r_value, p_value, std_err = linregress(media_anual_temperatura['Ano'], media_anual_temperatura['Temperatura do ar (°C)'])

# Calcular a linha de tendência
linha_tendencia = intercept + slope * media_anual_temperatura['Ano']

# Plotar a média anual de temperatura e a linha de tendência
plt.figure(figsize=(10, 6))
plt.plot(media_anual_temperatura['Ano'], media_anual_temperatura['Temperatura do ar (°C)'], label='Média Anual')
plt.plot(media_anual_temperatura['Ano'], linha_tendencia, 'r', label=f'Linha de Tendência (slope={slope:.4f})')
plt.xlabel('Ano')
plt.ylabel('Temperatura Média Anual (°C)')
plt.title('Tendência da Temperatura Média Anual em Fortaleza')
plt.legend()
plt.grid(True)
plt.savefig('/content/tend_climatica.png')  # Salvando o gráfico
plt.show()
plt.close()  # Fechando a figura para começar uma nova

# Avaliar a significância da tendência
if p_value < 0.05:
    print("Existe uma tendência significativa.")
else:
    print("Não existe uma tendência significativa.")
```

![tend_climatica](https://github.com/AssueroMeira/projeto_EDA_TP1_ADA/assets/39604515/e6a5c184-47da-4123-9b78-21c07724f402)

#### O gráfico acima mostra uma tendência significativa quanto à mudança climática na cidade de Fortaleza.

# **Insights**
#### 1. A vaiação de temperatura em Fortaleza permanecia constante durante os anos.
#### 2. Onde o dia mais quente foi 31/01/2024 com 33.8°C e o dia mais frio foi 31/07/2008 com 20.55°C.
#### 3. Embora temperaturas mais baixas indiquem maior chance de precipitação, nem sempre podem ocorrer chuvas.
#### 4. Há uma forte tendência de mudança climática na cidade de Fortaleza, cuja temperatura deve aumentar com o passar dos anos.

# **Recomendações**
#### 1. Fortaleza é uma cidade excelente para passar férias durante qualquer dia do ano, se seu objetivo for praia e sol.
#### 2. Os gestores da cidade devem se preocupar com efeitos do aquecimento global, pois há uma tendência de aquecimento, o que pode comprometer a agricultura e o turismo.

# **Conclusão**
#### A cidade de Fortaleza possui temperaturas agradáveis e constantes durante o ano, com poucas oscilações, sendo perfeita para aproveitar o calor e a praia em qualquer época do ano. Conutudo a cidade passa por um processo de alteração climática, que pode prejudicar tanto o turismo, como a agricultura local.



