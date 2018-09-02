import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as pkl
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler



#Configuração

pd.set_option('display.max_columns', 100)

cols_scouts_def = ['CA','CV','DD','DP','FC','GC','GS','RB','SG'] # alphabetical order
cols_scouts_atk = ['A','FD','FF','FS','FT','G','I','PE','PP'] # alphabetical order
cols_scouts = cols_scouts_def + cols_scouts_atk

scouts_weights = np.array([-2.0, -5.0, 3.0, 7.0, -0.5, -6.0, -2.0, 1.7, 5.0, 5.0, 1.0, 0.7, 0.5, 3.5, 8.0, -0.5, -0.3, -3.5])

ROUND_TO_PREDICT = 38

df = pd.read_csv('./data/dados_agregados.csv')
print(df.shape)
df.head(10)

#Processo de limpeza dos dados

print("Dimensões originais dos dados: ", df.shape)

# remove todas as linhas cujo scouts são NANs
df_clean = df.dropna(how='all', subset=cols_scouts)
print('qtde. de jogadores com scouts: ', df_clean.shape[0])

# remove todas as linhas com rodada == 0
df_clean = df_clean[df_clean['Rodada'] > 0]
print("qtde. de linhas após eliminação da rodada 0: ", df_clean.shape[0])

# remove técnicos e jogadores sem posição
df_clean = df_clean[(df_clean['Posicao'] != "tec") & (~df_clean['Posicao'].isnull())]
print("qtde. de linhas com posições válidas: ", df_clean.shape[0])

# remove todos os jogadores que não participaram de alguma rodada
df_clean = df_clean[(df_clean['Participou'] == True) | (df_clean['PrecoVariacao'] != 0)]
print("qtde. de linhas com jogadores que participaram de alguma rodada: ", df_clean.shape[0])

# altera os Status = NAN para 'Provável'
df_clean.loc[df_clean.Status.isnull(), 'Status'] = 'Provável'

# atualiza nomes dos jogadores sem ids e remove jogadores sem nome
df_ids =  df.groupby('AtletaID')['Apelido'].unique()
dict_ids = dict(zip(df_ids.index, [str(v[-1]) for v in df_ids.values]))
dict_ids = {k:v for k,v in dict_ids.items() if v != 'nan'}
df_clean['Apelido'] = df_clean['AtletaID'].map(dict_ids)
df_clean = df_clean[~df_clean['Apelido'].isnull()]
print("qtde. de jogadores com nome: ", df_clean.shape[0])

# preenche NANs restantes com zeros (verificado antes!)
df_clean.fillna(value=0, inplace=True)

print("Dimensão dos dados após as limpezas: ", df_clean.shape)
df_clean.head(10)


#Atualização dos times para jogadores
df_teams = pd.read_csv('./data/times_ids.csv')
df_teams = df_teams.dropna()
print(df_teams.shape)
df_teams.head()


# do not run this cell twice!
dict_teams_id = dict(zip(df_teams['id'], df_teams['nome.cartola']))
dict_teams_id.update(dict(zip(df_teams['cod.older'], df_teams['nome.cartola'])))

df_clean['ClubeID'] = df_clean['ClubeID'].astype(np.int).map(dict_teams_id)
df_clean = df_clean.dropna()

print(df_clean.shape)
df_clean.head()

#Atualização dos scouts cumulativos referentes ao ano de 2015

def get_scouts_for_round(df, round_):
    suffixes = ('_curr', '_prev')
    cols_current = [col + suffixes[0] for col in cols_scouts]
    cols_prev = [col + suffixes[1] for col in cols_scouts]

    df_round = df[df['Rodada'] == round_]
    if round_ == 1: return df_round

    df_round_prev = df[df['Rodada'] < round_].groupby('AtletaID', as_index=False)[cols_scouts].max()
    df_players = df_round.merge(df_round_prev, how='left', on=['AtletaID'], suffixes=suffixes)

    # if is the first round of a player, the scouts of previous rounds will be NaNs. Thus, set them to zero
    df_players.fillna(value=0, inplace=True)

    # compute the scouts
    df_players[cols_current] = df_players[cols_current].values - df_players[cols_prev].values

    # update the columns
    df_players.drop(labels=cols_prev, axis=1, inplace=True)
    df_players = df_players.rename(columns=dict(zip(cols_current, cols_scouts)))
    df_players.SG = df_players.SG.clip_lower(0)

    return df_players


df_scouts = df_clean[df_clean['ano'] != 2015]
df_scouts_2015 = df_clean[df_clean['ano'] == 2015]

n_rounds = df_scouts_2015['Rodada'].max()

if np.isnan(n_rounds):
    df_scouts = df_clean
else:
    for i in range(1, n_rounds + 1):
        df_round = get_scouts_for_round(df_scouts_2015, i)
        print("Dimensões da rodada #{0}: {1}".format(i, df_round.shape))
        df_scouts = df_scouts.append(df_round, ignore_index=True, sort=True)

print(df_scouts.shape)
df_scouts.head()

# Verificar se a coluna de pontuação dos jogadores condiz com o scout

def check_scouts(row):
    return np.sum(scouts_weights*row[cols_scouts])

players_points = df_scouts.apply(check_scouts, axis=1)
errors = np.where(~np.isclose(df_scouts['Pontos'].values, players_points))[0]
print("qtde. de jogadores com pontuação diferente dos scouts: ", errors.shape)
df_scouts.iloc[errors, :].tail(10)

# remove such players with wrong pontuation (DO NOT RUN TWICE!)
df_scouts.reset_index(drop=True, inplace=True)
df_scouts.drop(df.index[errors], inplace=True)
print(df_scouts.shape)
df_scouts.head()

# Remover linhas duplicadas

df_scouts.drop_duplicates(subset=['AtletaID', 'ano']+cols_scouts, keep='first', inplace=True)

print("Dimensões dos dados após toda a limpeza de dados: ", df_scouts.shape)
df_scouts.to_csv('./data/dados_agregados_limpos.csv', index=False)


#Criação das amostras

df_samples = pd.read_csv('./data/dados_agregados_limpos.csv')
print("Dados para amostra", df_samples.shape)
df_samples.head()


# seleciona somente as colunas de interesse para usar como atributos
cols_of_interest = df_samples.columns.difference(['Apelido', 'Status', 'Participou', 'dia', 'mes']).values.tolist()

# 'Rodada' e 'ano' serão usadas para criar amostras
cols_info = ['Rodada', 'ano']

df_samples = df_samples[cols_of_interest]
df_samples.head()

teams_full = pd.Series(df_samples['ClubeID'].unique()).sort_values().values

def dict_positions(to_int = True):
    dict_map = {'gol':1, 'zag':2, 'lat':3, 'mei':4, 'ata':5}
    return  dict_map if to_int else dict(zip(dict_map.values(), dict_map.keys()))

def dict_teams(to_int = True):
    teams_map = {team:(index+1) for index, team in enumerate(teams_full)}
    return teams_map if to_int else dict(zip(teams_map.values(), teams_map.keys()))

print(dict_positions(), dict_teams(), sep='\n')

# mapeia "casa", "atletas.clube_id" and "Posicao" para números inteiros
df_samples['ClubeID'] = df_samples['ClubeID'].map(dict_teams(to_int=True))
df_samples['Posicao'] = df_samples['Posicao'].map(dict_positions(to_int=True))
df_samples['variable'] = df_samples['variable'].map({'home.team':1, 'away.team':2})
df_samples.head()


df_samples.to_csv('./data/dados_agregados_amostras.csv', index=False)

# Treinamento do modelo utilizando Redes Neurais Artificais, apenas com dados de 2017.

df_samples = pd.read_csv('./data/dados_agregados_amostras.csv')
df_samples = df_samples[df_samples.ano == 2017]
print(df_samples.shape)
df_samples.head()


def create_samples(df, round_train, round_pred):
    '''Create a Dataframe with players from round_train, but with 'Pontos' of round_pred'''
    df_train = df[df['Rodada'] == round_train]
    df_pred = df[df['Rodada'] == round_pred][['AtletaID', 'Pontos']]
    df_merge = df_train.merge(df_pred, on='AtletaID', suffixes=['_train', '_pred'])

    df_merge = df_merge.rename(columns={'Pontos_train': 'Pontos', 'Pontos_pred': 'pred'})

    return df_merge

df_train = pd.DataFrame(data=[], columns=list(df_samples.columns) + ['pred'])
n_rounds = df_samples['Rodada'].max()

for round_train, round_pred in zip(range(1, n_rounds), range(2, n_rounds + 1)):
    df_round = create_samples(df_samples, round_train, round_pred)
    print('qtde. de jogadores que participaram na rodada {0:=2} (train) e na rodada {1:=2} (pred): {2:=4}'.format(
        round_train, round_pred, df_round.shape[0]))
    df_train = df_train.append(df_round, ignore_index=True)

print("Dimensões dos dados de treinamento: ", df_train.shape)


import warnings
warnings.filterwarnings("ignore")

samples = df_train[df_train.columns.difference(['AtletaID', 'Rodada','pred'])].values.astype(np.float64)
scores  = df_train['pred'].values
print(samples.shape, scores.shape)

steps = [('MinMax', MinMaxScaler()), ('NN', MLPRegressor(solver='adam', activation='identity', learning_rate_init=1e-2, momentum=0.9, max_iter=2000))]
pipe = Pipeline(steps)
params = dict(NN__hidden_layer_sizes=[(50,50,50,50), (50,100,50), (50,100,100,50)])

reg = GridSearchCV(pipe, params, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=10)
reg.fit(samples, scores)
print(reg.best_params_, reg.best_score_)

scores_pred = reg.predict(samples)

plt.figure(figsize=(16,6))
plt.plot(range(scores.shape[0]), scores, color='blue')
plt.plot(range(scores_pred.shape[0]), scores_pred, color='red')

pkl.dump(reg, open('./modelo/nn1.pkl', 'wb'), -1)

# Predições - Carregar modelo treinado e dizer melhores jogadores para uma próxima rodada.

df_test = pd.read_csv('./data/dados_agregados_limpos.csv')
df_test = df_test[df_test.ano == 2017]
reg = pkl.load(open('./modelo/nn1.pkl', 'rb'))


def to_samples(df):
    df_samples = df[cols_info+cols_of_interest].copy()
    df_samples['ClubeID'] = df_samples['ClubeID'].map(dict_teams(to_int=True))
    df_samples['Posicao'] = df_samples['Posicao'].map(dict_positions(to_int=True))
    df_samples['variable'] = df_samples['variable'].map({'home.team':1, 'away.team':2})
    df_samples.reset_index(drop=True, inplace=True)
    return df_samples

def predict_best_players(df_samples, reg, n_players=11):
    samples = df_samples[df_samples.columns.difference(['AtletaID', 'Rodada', 'ano'])].values.astype(np.float64)

    pred = reg.predict(samples)
    best_indexes = pred.argsort()[-n_players:]
    return df_samples.iloc[best_indexes]

def predict_best_players_by_position(df_samples, reg, n_gol=5, n_zag=5, n_lat=5, n_mei=5, n_atk=5):
    df_result = pd.DataFrame(columns=df_samples.columns)
    for n_players, pos in zip([n_gol, n_zag, n_lat, n_mei, n_atk], range(1, 6)):
        samples = df_samples[df_samples['Posicao'] == pos]
        df_pos = predict_best_players(samples, reg, n_players)
        df_result = df_result.append(df_pos)

    return df_result

df_rodada = df_test[(df_test['Rodada'] == (ROUND_TO_PREDICT-1)) & (df_test['Status'] == "Provável")]
df_samples = to_samples(df_rodada)
print(df_samples.shape)



