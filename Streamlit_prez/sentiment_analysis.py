#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 21:38:42 2022

@author: anak
"""

from nltk.tokenize import word_tokenize

 
import string
from nltk import word_tokenize, pos_tag
import matplotlib.pyplot as plt 
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk import FreqDist

from wordcloud import WordCloud, STOPWORDS 

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
import pickle

from PIL import Image
import requests
import json
import re
from nltk.tokenize import word_tokenize
from gensim import matutils, models
import gensim 
from gensim import corpora 
from gensim.models.coherencemodel import CoherenceModel

# libraries for visualization 
import pyLDAvis 
import pyLDAvis.gensim_models

import scipy.sparse

#import pyLDAvis.gensim 
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk import FreqDist
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
from sklearn import datasets
import altair as alt
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import plotly.graph_objects as go
import warnings
import streamlit as st
from bokeh.plotting import figure
import squarify
import plotly.io as pio
import streamlit_wordcloud as wordcloud
import json
from wordcloud import WordCloud

pio.templates.default ='ggplot2'



warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")


data = pd.read_csv(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/Analyzed_Data_Avene.csv")
data2= pd.read_csv(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/Analyzed_Data_Roche.csv")
data3= pd.read_csv(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/Analyzed_Data_Vichy.csv")

data2= data2.dropna()
data2= data2.drop(columns='Unnamed: 0')

st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/image1.png", use_column_width=True)
st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/image2.svg", use_column_width=True)
st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/image1.png", use_column_width=True)


st.sidebar.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/imagesidebar.png", use_column_width=True)
st.sidebar.write("###  WELCOME ⚜️!")
st.sidebar.write("You can set up different display options here below")
st.sidebar.write("")
st.sidebar.write("#####  CONTENT")
description = st.sidebar.checkbox(" Project Insights")
st.sidebar.write("  1️⃣ Business Case 🛍️ ")
st.sidebar.write("  2️⃣ Project Overview")
st.sidebar.write("  3️⃣ Our Data")

st.sidebar.write("")


eda = st.sidebar.checkbox(" EDA")
st.sidebar.write("  1️⃣ Stars Report 💫 ")
st.sidebar.write("  2️⃣ Most Used Words")
st.sidebar.write("  3️⃣ Characters and Words Study 🔍 ")

st.sidebar.write("")

sentimentfe = st.sidebar.checkbox(" Sentiment Analysis Features")
st.sidebar.write("  1️⃣ Polarity Analysis")
st.sidebar.write("  2️⃣ Subjectivity Analysis ")
st.sidebar.write("  3️⃣ Polarity Vs Subjectivity Analysis ")
st.sidebar.write("  4️⃣ Good and Bad Reviews 👍👎 ")
st.sidebar.write("  5️⃣ Extreme Polarity Reviews")
st.sidebar.write("")

Models = st.sidebar.checkbox(" Sentiment Analysis Model")
st.sidebar.write("  1️⃣ Clusters")
st.sidebar.write("")

conclusion = st.sidebar.checkbox(" Main Results")
st.sidebar.write("  1️⃣ Main Results")

with st.container():
    st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/description.svg", use_column_width=True)
    if description:
        with st.container():
            with st.expander("Business Case"):
                st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/slideBusinessCase.png", use_column_width=True)

        with st.container():
            with st.expander("Project Overview"):
                st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/SlideProjectOverview.png", use_column_width=True)
            
        with st.container():
            with st.expander("Our Data"):
                st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/slideData.png", use_column_width=True)

        
        

with st.container():
    st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/eda.svg", use_column_width=True)
    if eda:
        with st.container():
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Avène", data.Text.count(),"👍Reviews👎",delta_color="off")
            col2.metric("La Roche-Posay", data2.Text.count(),"👍Reviews👎",delta_color="off")
            col3.metric("Vichy",data3.Text.count(),"👍Reviews👎",delta_color="off")
        #EDA PART
        
        
        with st.container():
            st.markdown("## Stars Report " )
            st.markdown("Distribution of the Number Reviews per Stars ✨")
            avene = st.checkbox('Avene')
            roche = st.checkbox('La Roche-Posay') 
            vichy= st.checkbox('Vichy')
            
            with st.container(): 
                
            
                col1,col2,col3=st.columns([1,1,1])
                if avene:
                    with col1:
                        # Create a pieplot
                        count_stars = data.groupby(data.Stars, as_index = False).agg({'Text':'count'})
                        names = ['1 star','2 star','3 star','4 star','5 star']
                        
                        fig = px.pie(count_stars, values='Text', names=names,title='Percentage of Rides/Cluster')
                        fig = go.Figure(data=[go.Pie(labels=names, values=count_stars.Text, hole=.4)])
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(showlegend=False)
                        fig.update_layout(title="Avène")
                        st.plotly_chart(fig, use_container_width=True)
                        
                
                
                if roche:
                    with col2:
                    
                                # Create a pieplot
                        count_stars = data2.groupby(data2.Stars, as_index = False).agg({'Text':'count'})
                        names = ['1 star','2 star','3 star','4 star','5 star']
                        
        
                        fig = go.Figure(data=[go.Pie(labels=names, values=count_stars.Text, hole=.4)])
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(showlegend=False)
                        fig.update_layout(showlegend=False)
                        fig.update_layout(title="La Roche-Posay")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        
                if vichy:
                    with col3:
                    
                                # Create a pieplot
                        count_stars = data3.groupby(data3.Stars, as_index = False).agg({'Text':'count'})
                        names = ['1 star','2 star','3 star','4 star','5 star']
                        
                        fig = px.pie(count_stars, values='Text', names=names,title='Percentage of Rides/Cluster')
                        fig = go.Figure(data=[go.Pie(labels=names, values=count_stars.Text, hole=.4)])
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(showlegend=False)
                        fig.update_layout(title="Vichy")
                        st.plotly_chart(fig, use_container_width=True) 
                
        with st.container():
            st.markdown("## Most Used Words " )
            st.markdown("Which words were used the most in the Reviews? Let's Check! " )
            check2=st.radio(
             "Brands",
             ('Avène', 'La Roche-Posay', 'Vichy'))
            if check2=='Avène':
                all_words1 = ' '.join([text for text in data.Text])
                all_words1 = all_words1.split()
                fdist = FreqDist(all_words1)
                words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
                
                d = words_df.nlargest(columns="count", n=20)
                d = d.sort_values(by='count', ascending=True)
                d['freq_rel'] = d['count']/(sum(d['count']))
                
                fig = go.Figure(go.Bar(
                    x=d['freq_rel'],
                    y=d.word,
                    orientation='h'))
                fig.update_layout(title="🔝 Avène")
                st.plotly_chart(fig, use_container_width=True)
        
            if check2=='La Roche-Posay':
                
                good_polarity_reviews2 = data2[data2.polarity > 0.7]
                all_words4 = ' '.join([text for text in data2.Text])
                all_words4 = all_words4.split()
            
                fdist = FreqDist(all_words4 )
                words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
                
                d = words_df.nlargest(columns="count", n=20)
                d = d.sort_values(by='count', ascending=True)
                d['freq_rel'] = d['count']/(sum(d['count']))
                
                fig = go.Figure(go.Bar(
                    x=d['freq_rel'],
                    y=d.word,
                    orientation='h'))
                fig.update_layout(title="🔝 La Roche-Posay")
                st.plotly_chart(fig, use_container_width=True)
                
            if check2=='Vichy':
                
                good_polarity_reviews = data3[data3.polarity > 0.7]
                all_words3 = ' '.join([text for text in data3.Text])
                all_words3 = all_words3.split()
            
                fdist = FreqDist(all_words3)
                words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
                
                d = words_df.nlargest(columns="count", n=20)
                d = d.sort_values(by='count', ascending=True)
                d['freq_rel'] = d['count']/(sum(d['count']))
                
                fig = go.Figure(go.Bar(
                    x=d['freq_rel'],
                    y=d.word,
                    orientation='h'))
                fig.update_layout(title="🔝 Vichy")
                st.plotly_chart(fig, use_container_width=True)            
        
        with st.container():
            st.markdown("## Characters and Words Study 🔍 " )
            check3=st.radio(
             "Which Brand you would like to check?",
             ('Avène ', 'La Roche-Posay ', 'Vichy '))
            
            with st.container():
                if check3=='Avène ':
                    col1,col2,col3=st.columns([1,2,3])
                    col1.metric("Mean #️⃣ Characters ", round(data.char_count.mean(),1))
                    col2.metric("Mean #️⃣ Words ", round(data.word_count.mean(),1),)
                    col3.metric("Mean Lenght Words",round(data.avg_word.mean(),1))
                    
                if check3=='La Roche-Posay ':
                    col1,col2,col3=st.columns([1,2,3])
                    col1.metric("Mean #️⃣ Characters ", round(data2.char_count.mean(),1))
                    col2.metric("Mean #️⃣ Words ", round(data2.word_count.mean(),1))
                    col3.metric("Mean Lenght Words",round(data2.avg_word.mean(),1))
                    

                if check3=='Vichy ':
                    col1,col2,col3=st.columns([1,2,3])
                    col1.metric("Mean #️⃣ Characters ", round(data3.char_count.mean(),1))
                    col2.metric("Mean #️⃣ Words ", round(data3.word_count.mean(),1))
                    col3.metric("Mean Lenght Words",round(data3.avg_word.mean(),1))                        
            
            if check3=='Avène ':
                col1,col2,col3=st.columns([1,1,1])
                with col1:
                    
                    fig = px.histogram(data, x=data.char_count, nbins=20)
                    fig.update_layout(title="#️⃣ Characters per Reviews")
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col2:
                    fig = px.histogram(data, x=data.word_count, nbins=20)
                    fig.update_layout(title="#️⃣ Words per Reviews")
                    st.plotly_chart(fig, use_container_width=True)
        
                with col3:
                    fig = px.histogram(data, x=data.avg_word, nbins=20)
                    fig.update_layout(title="Average Word Lenght 📶")
                    st.plotly_chart(fig, use_container_width=True)            
                    
            if check3=='La Roche-Posay ':
                col1,col2,col3=st.columns([1,1,1])
                with col1:
                    
                    fig = px.histogram(data2, x=data2.char_count, nbins=20)
                    fig.update_layout(title="#️⃣ Characters per Reviews")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:  
                    fig = px.histogram(data2, x=data2.word_count, nbins=20)
                    fig.update_layout(title="#️⃣ Words per Reviews")
                    st.plotly_chart(fig, use_container_width=True)
                with col3:    
                    fig = px.histogram(data2, x=data2.avg_word, nbins=20)
                    fig.update_layout(title="Average Word Lenght 📶")
                    st.plotly_chart(fig, use_container_width=True)
                    
            if check3=='Vichy ':
                col1,col2,col3=st.columns([1,1,1])
                with col1:
                    
                    fig = px.histogram(data3, x=data3.char_count, nbins=20)
                    fig.update_layout(title="#️⃣ Characters per Reviews")
                    st.plotly_chart(fig, use_container_width=True) 
                with col2:
                    fig = px.histogram(data3, x=data3.word_count, nbins=20)
                    fig.update_layout(title="#️⃣ Words per Reviews")
                    st.plotly_chart(fig, use_container_width=True)            
                with col3:    
                    fig = px.histogram(data3, x=data3.avg_word, nbins=20)
                    fig.update_layout(title="Average Word Lenght 📶")
                    st.plotly_chart(fig, use_container_width=True)

#Polarity study
with st.container():
    st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/sentimentana.svg", use_column_width=True)
    if sentimentfe:
        with st.container():
            st.markdown("## Polarity Analysis " )
            col1,col2=st.columns([1,1])
            with col1:
                
                axex = st.radio("Insights",
                 ('Polarity Distribution','Polarity Vs Stars', 'Polarity Vs Punctuation', 'Polarity Vs Number of Words','Polarity Vs Number of Uppercase'))
            with col2:  
                st.markdown("#### Brands " )
                datap=pd.DataFrame()
                datap=data
                if st.button('Avène'):
                    datap=data
                if st.button('La Roche-Posay'):
                    datap=data2
                if st.button('Vichy'):
                    datap=data3 
                    
            with st.container():
                
                if axex=='Polarity Distribution':    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=datap['polarity']))
                    fig.update_layout(title="Polarity Score Distribution")
                    fig.update_traces(opacity=0.70)
                    st.plotly_chart(fig, use_container_width=True)
                    
            with st.container():
                if axex=='Polarity Vs Stars':
                    x='Stars'
                    fig = px.box(datap, y="polarity", x=x)
                    fig.update_layout(title="Polarity Vs Number of Stars 🌟")
                    st.plotly_chart(fig, use_container_width=True)
            with st.container():       
                if axex=='Polarity Vs Punctuation':
                    x='punctuation'
                    fig = px.box(datap, y="polarity", x=x)
                    fig.update_layout(title='Polarity Vs Punctuation')
                    st.plotly_chart(fig, use_container_width=True)
            with st.container():    
                if axex=='Polarity Vs Number of Words':
                    x='word_count'
                    fig = px.box(datap, y="polarity", x=x)
                    fig.update_layout(title='Polarity Vs Number of Words')
                    st.plotly_chart(fig, use_container_width=True)
                    
            with st.container():
                if axex=='Polarity Vs Number of Uppercase':
                    x='upper'
                    fig = px.box(datap, y="polarity", x=x)
                    fig.update_layout(title='Polarity Vs Number of Uppercase')
                    st.plotly_chart(fig, use_container_width=True)
            
        
        
        
        # Subjectivity Study
        
        
        
        with st.container():
            st.markdown("## Subjectivity Analysis " )
            col1,col2=st.columns([1,1])
            with col1:
                
                axex2 = st.radio("Insights",('Subjectivity Distribution','Subjectivity Vs Stars', 'Subjectivity Vs Punctuation', 'Subjectivity Vs Number of Words','Subjectivity Vs Number of Uppercase'))
            
            with col2:  
                st.markdown("#### Brands " )
                datas=pd.DataFrame()
                datas=data
                if st.button('Avène '):
                    datas=data
                if st.button('La Roche-Posay '):
                    datas=data2
                if st.button('Vichy '):
                    datas=data3 
                    
            with st.container():
                if axex2=='Subjectivity Distribution':    
                    fig= go.Figure()
                    fig.add_trace(go.Histogram(x=datas['subjectivity']))
                    fig.update_layout(title="Subjectivity Score Distribution")
                    
                    fig.update_traces(opacity=0.70)
                    st.plotly_chart(fig, use_container_width=True)
            with st.container():
                if axex2=='Subjectivity Vs Stars':
                    x='Stars'
                    fig = px.box(datas, y="subjectivity", x=x)
                    fig.update_layout(title="Subjectivity Vs Number of Stars 🌟")
                    st.plotly_chart(fig, use_container_width=True)
            with st.container():  
                if axex2=='Subjectivity Vs Punctuation':
                    x='punctuation'
                    fig = px.box(datas, y="subjectivity", x=x)
                    fig.update_layout(title='Subjectivity Vs Punctuation')
                    st.plotly_chart(fig, use_container_width=True)
            with st.container():   
                if axex2=='Subjectivity Vs Number of Words':
                    x='word_count'
                    fig = px.box(datas, y="subjectivity", x=x)
                    fig.update_layout(title='Subjectivity Vs Number of Words')
                    st.plotly_chart(fig, use_container_width=True)
            with st.container():       
        
                if axex2=='Subjectivity Vs Number of Uppercase':
                    x='upper'
                    fig = px.box(datas, y="subjectivity", x=x)
                    fig.update_layout(title='Subjectivity Vs Number of Uppercase')
                    st.plotly_chart(fig, use_container_width=True)
        
        #Subjectivity VS Polarity
        
        
        
        
        
        
                    
        with st.container():
            st.markdown("## Polarity and Subjectivity Analysis" )
            
        
            option2 = st.selectbox(
             'Select one Brand',
             ('Avène   ', 'La Roche-Posay   ', 'Vichy   '))
            
            if option2=='Avène   ':
                datat=data
            if option2=='La Roche-Posay   ':
                datat=data2
            if option2=='Vichy   ':
                datat=data3
            col1,col2=st.columns([1,1])
            with col1:
                
                
                fig=px.scatter(datat, x='polarity', y='subjectivity', animation_frame="Stars",trendline="ols")
                fig.update_layout(title='Subjectivity Vs Polarity')
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:    
        
                df2= pd.DataFrame()
                for i in range (1,6):
                  df2.loc[i,'polarity']= (datat.loc[data.Stars == i].polarity.mean())
                  df2.loc[i,'subjectivity']= (datat.loc[data.Stars == i].subjectivity.mean())
                  
                  
                fig=px.scatter(df2, x='polarity', y='subjectivity',text=df2.index,symbol=df2.index)
                fig.update_traces(textposition="bottom right")
                fig.update_layout(
                title="Mean Polarity Vs Mean Subjectivity",
                xaxis_title='⬅ Negative -- Positive ⮕',
                yaxis_title='⬅ Facts -- Opinions ⮕',
                legend_title="Legend Title",
                #font=dict(
                    #family="Courier New, monospace",
                    #size=18,
                    #color="RebeccaPurple"
                #)
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            
        
            
            
        
            
        
            
        
            
            
#wordcloud
#señalarpalabras
#Pie

        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        with st.container():
            st.markdown("## Good and Bad Reviews " )
            dfp=data
            
            #Definition function
            def regroup_text(dataSerie):
                text = '' 
                for val in dataSerie: 
                    val = str(val) 
                    tokens = val.split() 
                    text += " ".join(tokens)+" "
                return text
            
            def nouns(text):
                '''Given a string of text, tokenize the text and pull out only the nouns.'''
                is_noun = lambda pos: pos[:2] == 'NN'
                tokenized = word_tokenize(text)
                all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
                return ' '.join(all_nouns)
            
            def adj(text):
                '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
                is_adj = lambda pos: pos[:2] == 'JJ'
                tokenized = word_tokenize(text)
                adj = [word for (word, pos) in pos_tag(tokenized) if is_adj(pos)] 
                return ' '.join(adj)
            
            def nouns_adj(text):
                '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
                is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
                tokenized = word_tokenize(text)
                nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
                return ' '.join(nouns_adj)
            
            def freq_words_in_text(text):
                all_words = text.split()
               
                fdist = FreqDist(all_words)
                words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
                
                
                d = words_df.nlargest(columns="count", n=20)
                d = d.sort_values(by='count', ascending=True)
                d['freq_rel'] = d['count']/(sum(d['count']))
                plt.barh(d.word,d['freq_rel'])
                plt.title('Most frequent words')
                plt.xlabel('Percentage')
                plt.show()
            
            
            
            
            
            
            if st.button('Avène  '):
                dfp=data
            if st.button('La Roche-Posay  '):
                dfp=data2
            if st.button('Vichy  '):
                dfp=data3
            if st.button('All  '):
                dfp=pd.concat([data, data2,data3], ignore_index=True, sort=False)
                
            
            with st.container():
                st.markdown("### Most Used Words for All Reviews " )
                df_good_reviews = dfp
                corpus_good_reviews =[]
                freq=[]
                freq = df_good_reviews['Text'].str.split()
                freq = freq.values.tolist()
                freq = [word for i in freq for word in i]
                freq_words =''
                freq_words+=" ".join(freq)+" "
            
                wordcloud_freq = WordCloud(background_color = "white", max_words = 20,colormap='bone').generate(freq_words)
                plt.imshow(wordcloud_freq, interpolation = 'bilinear')
                plt.axis("off")
                st.pyplot()
                

            with st.container():
                st.markdown("### Top Word Analysis ➡️ Nouns" )

                def regroup_text(dataSerie):
                    text = '' 
                    for val in dataSerie: 
                        val = str(val) 
                        tokens = val.split() 
                        text += " ".join(tokens)+" "
                    return text
                
                def nouns(text):
                    '''Given a string of text, tokenize the text and pull out only the nouns.'''
                    is_noun = lambda pos: pos[:2] == 'NN'
                    tokenized = word_tokenize(text)
                    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
                    return ' '.join(all_nouns)
                
                def adj(text):
                    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
                    is_adj = lambda pos: pos[:2] == 'JJ'
                    tokenized = word_tokenize(text)
                    adj = [word for (word, pos) in pos_tag(tokenized) if is_adj(pos)] 
                    return ' '.join(adj)
                
                def nouns_adj(text):
                    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
                    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
                    tokenized = word_tokenize(text)
                    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
                    return ' '.join(nouns_adj)
                
                def freq_words_in_text(text):
                    all_words = text.split()
                   
                    fdist = FreqDist(all_words)
                    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
                    
                    d = words_df.nlargest(columns="count", n=20)
                    d = d.sort_values(by='count', ascending=True)
                    d['freq_rel'] = d['count']/(sum(d['count']))
                    fig = go.Figure(go.Bar(
                    x=d['freq_rel'],
                    y=d.word,
                    orientation='h'))
                    st.plotly_chart(fig, use_container_width=True)
                
                
                with st.container():
                    col1,col2=st.columns([1,1])
                    
                    
                    with col1: 
                        col1.markdown("## Good Reviews")
                        good_reviews = dfp[(dfp.Stars == 5)]
                        bad_reviews = dfp[((dfp.Stars == 1) | (data.Stars == 2))]
                        nouns_gr = nouns(regroup_text(good_reviews.Text))
                        freq_words_in_text(nouns_gr)
                        st.markdown("### Top Word Analysis ➡️ Adjectives" )
                    
                    with col2:
                        col2.markdown("## Bad Reviews")
                        noun_gr = nouns(regroup_text(bad_reviews.Text))
                        freq_words_in_text(noun_gr)

                with st.container():
                    col1,col2=st.columns([1,1])
                    
                    
                    with col1: 
                        col1.markdown("## Good Reviews")
                        adj_gr = adj(regroup_text(good_reviews.Text))
                        freq_words_in_text(adj_gr)
                    
                    with col2:
                        col2.markdown("## Bad Reviews")
                        adj_gr = adj(regroup_text(bad_reviews.Text))
                        freq_words_in_text(adj_gr)
                
        with st.container():
            with st.expander("Polarity Reviews Extremes"):
                st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/slideExtremesPolarityReviews.png", use_column_width=True)





        
with st.container():
    st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/models.svg", use_column_width=True)
    if Models:            

        with st.container():
            with st.expander("Topic Modeling"):
                st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/SlideTopicModeling.png", use_column_width=True)


        with st.container():
            st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/clusters.png", use_column_width=True)

with st.container():
    st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/mainresults.svg", use_column_width=True)
    if conclusion:
        with st.container():
            with st.expander("Main Results"):
                st.image(r"/Users/anak/DAFT_NOV_21_01-main/module_3/Final_project/SLIDEMainResults2.png", use_column_width=True)
