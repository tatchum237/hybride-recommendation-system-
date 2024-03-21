import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf



def recomandation(inputUser, n, attract):
    
  v0 = inputUser
  w = np.load('poids_ub/w.npy')
  hb = np.load('poids_ub/hb.npy')
  vb = np.load('poids_ub/vb.npy')
  hh0 = tf.nn.sigmoid(tf.matmul([v0], w) + hb)
  vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(w)) + vb)
  rec = vv1


  attraction_score = attract.assign(recommandationScore = rec[0])
  return attraction_score.sort_values(['recommandationScore'], ascending=False)[:n]

def used_based(ratings, n):
   data = pd.read_csv('data/attract.csv')
   attract = np.zeros(104)
   print(data.shape)
   for rating in ratings:
      attract[rating['attraction_id']] = rating['note']
   inputUser = tf.convert_to_tensor(attract, "float32")
   return recomandation(inputUser, n, data)



def recomandati(inputUser, n, attract):
  v0 = inputUser
  w = np.load('poids_ib/w.npy')
  hb = np.load('poids_ib/hb.npy')
  vb = np.load('poids_ib/vb.npy')
  hh0 = tf.nn.sigmoid(tf.matmul([v0], w) + hb)
  rec = hh0.numpy().flatten()

  attraction_score = attract.assign(recommandationScore = rec)
  return attraction_score.sort_values(['recommandationScore'], ascending=False)[:n]

def item_based(ratings, n):
   data = pd.read_csv('data/attract.csv')
   trY = np.load('poids_ib/trY.npy')
   inputUsers = tf.zeros(shape=(9048, ))
   inputUser = tf.zeros(shape=(9048, ))
   note = ratings[0]['note']
   at = ratings[0]['attraction_id']
   for rating in ratings:
        if note < rating["note"]:
           note = rating["note"]
           at = rating['attraction_id']
    
   inputUser = tf.convert_to_tensor(trY[at - 1], "float32")

   return recomandati(inputUser, n,data)





def get_recommendations(title, n):
    data = pd.read_csv('data/attract.csv')
    cosine_sim = np.load('poids_cb/cosine_sim0.npy')
    indices = pd.Series(data.index, index=data['attraction_id']).drop_duplicates()
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the attractions based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: np.max(x[1]), reverse=True)
    #sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar attractions
    sim_scores = sim_scores[1:n+1]

    # Get the attraction indices
    movie_indices = [i[0] for i in sim_scores]

    # Get the attraction scores
    movie_score = [i[1] for i in sim_scores]
    data1 = data.iloc[movie_indices]

    data1 = data1.assign(recommandationScore = movie_score)
    # Return the1 top 10 most similar attractions
    return data1

def content_based(ratings, n):
       note = ratings[0]['note']
       at = ratings[0]['attraction_id']
       for rating in ratings:
         if note < rating["note"]:
            note = rating["note"]
            at = rating['attraction_id']
       return get_recommendations(at, n)




def reconstr(id):
    # Création d'un jeu de données fictif
    data = pd.read_csv('data/attract.csv')
    indices = pd.Series(data.index, index=data['attraction_id']).drop_duplicates()
    # Get the index of the movie that matches the title
    return data.iloc[id]

def combine(data1, data2):

    A1 = pd.DataFrame(data1)
    A2 = pd.DataFrame(data2)
    a = 0.5  

   # Fusionner les DataFrames en utilisant la colonne 'id' comme clé de fusion
    merged_df = pd.merge(A1, A2, on='attraction_id', how='outer')

   # Remplacer les valeurs manquantes par 0
    merged_df.fillna(0, inplace=True)

   # Calculer la combinaison pondérée
    merged_df['combinaison_ponderee'] = a * merged_df['recommandationScore_x'] + (1 - a) * merged_df['recommandationScore_y']
    merged_df = merged_df.sort_values(['combinaison_ponderee'], ascending=False)[:8]

   # Afficher le DataFrame résultant avec la combinaison pondérée
    re = []
    for index, row in merged_df.iterrows():
       re.append(reconstr(row['attraction_id']))
      
    re = pd.DataFrame(re)
    re = re.assign(combinaison_ponderee = merged_df['combinaison_ponderee'])
       

    return merged_df['combinaison_ponderee'], re

   
   


   