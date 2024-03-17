import streamlit as st
import numpy as np
import joblib

st.title("Prediction des sentiments")
st.markdown("Cette application analyse les phrases et nous donne le sentiment approuvé dans cette phrase ")

# 
model = joblib.load(filename="model_joblib")

#Fonction d'inférence
def infe(lemmatize_joined):
    new_data = ("lemmatize_joined")
    pred = model.predict(new_data.reshape(1,-1))
    return pred

# L'utilisateur saisie le text de son twitt
st.text_input(label="Pseudo :")
lemmatize_joined = st.text_area(label= "Votre X(tweet) :" )

# bouton de prediction
if st.button("Prediction"):
    prediction = infe(lemmatize_joined)

    result= "Votre X(twitt) est "+str(prediction)
    st.success(result)




