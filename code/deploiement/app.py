import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_star_rating import st_star_rating
import tensorflow as tf

import utils

# Changer la couleur du fond
st.markdown(
    """
    <style>
    .reportview-container {
        background: #add8e6; /* Bleu clair */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Création d'un jeu de données fictif
data = pd.read_csv('data/attract.csv')

# Titre de l'application
st.title('les attractions à noter')
@st.cache_data(experimental_allow_widgets=True)
def getdata(data):
    return data.sample(n=5)


data_ = getdata(data)
cols = st.columns(len(data_))

 
 
# Fonction pour afficher l'interface utilisateur
def display_ratings(data):
    ratings = []
    i = 0
    image_column_width = 200
    rating_column_width = 300
    titre_column_width = 200
    # Afficher chaque image avec un champ de notation
    for index, row in data.iterrows():
        col0, col1, col2 = st.columns([titre_column_width, image_column_width, rating_column_width])
        #with cols[i]:
        i += 1
        with col0:
          st.write(row['category'])
        with col1:
          st.image(row['image'], width=image_column_width)
        with col2:
          stars = st_star_rating(" ", maxValue=5, defaultValue=1, key=row['attraction_id'])
        #rating = st.slider(f"Noter l'image {row['attraction_id']} (sur 5) :", min_value=0, max_value=5, value=0, step=1)
 
    
        ratings.append({'attraction_id': row['attraction_id'], 'note': stars / 5})

    # Bouton pour envoyer les notes
    if st.button('Envoyer les notes'):
       
        userbased = utils.used_based(ratings, 20)
        itembased = utils.item_based(ratings, 20)
        #st.write(userbased)
        #st.write(itembased)
        combi, cfc = utils.combine(userbased, itembased)
        content = utils.content_based(ratings, 2)

        #st.write(cfc)

        #st.write(combi)
        
        #st.write(content)

        #for index, row in combi.iterrows():

        st.write('Les attractions susceptibles de vous intéresser.')
                
        for index, row in content.iterrows():
            col0, col1, col2 = st.columns([titre_column_width, image_column_width, rating_column_width])
        #with cols[i]:
            i += 1
            with col0:
                st.write(row['category'])
            with col1:
                st.image(row['image'], width=image_column_width)
            with col2:
                st.write(row['price'])

                
        for index, row in cfc.iterrows():
            col0, col1, col2 = st.columns([titre_column_width, image_column_width, rating_column_width])
        #with cols[i]:
            i += 1
            with col0:
                st.write(row['category'])
            with col1:
                st.image(row['image'], width=image_column_width)
            with col2:
                st.write(row['price'])
            





        st.map(content,
         latitude='lat',
         longitude='lng',
         size=[100, 150, 200],
         color='#00ff00',
         )
        
        st.map(cfc,
         latitude='lat',
         longitude='lng',
         size=[100, 150, 200],
         color='#00ff00',
         )

        
# Appeler la fonction pour afficher l'interface utilisateur
display_ratings(data_)







