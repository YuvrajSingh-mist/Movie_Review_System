import streamlit as st
import time
import pandas  as pd
import pickle
import nltk
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import gensim
import plotly.express as px
from streamlit_card import card

import time
from tqdm import tqdm
from bs4 import BeautifulSoup
import imdb


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from tqdm import tqdm
from scrapy.selector import Selector

#Downloading some dependencies
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords


import spacy

nlp = spacy.load('en_core_web_sm')


# https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl



#Loading the final model.pkl file

file_name = 'final_model_movie_reviews_sentiment_analysis.pkl'
open_file = open(file_name, 'rb')
final_model = pickle.load(open_file)
open_file.close()


#Loading the word2vec model.pkl file

file_name = 'word2vec_model.pkl'
open_file = open(file_name, 'rb')
word2vec_model = pickle.load(open_file)
open_file.close()


#Getting thwe movie reviews
def fetch_reviews(final_name):
    
    number = get_movie_id(final_name)
    driver = webdriver.Edge()
    url = 'https://www.imdb.com/title/tt{}/reviews?sort=submissionDate&dir=asc&ratingFilter=0'.format(number)

    time.sleep(1)
    driver.get(url)

    # print(driver.title)
    time.sleep(1)


    #Extracting the review count

    sel = Selector(text = driver.page_source)
    review_counts = sel.css('.lister .header span::text').extract_first().replace(',', '').split(' ')[0]

    total = int(int(review_counts) / 25)
    # print(total)
    # #Click on the 'load more' buttion n times

    try:

        for i in tqdm(range(total)):

                css_selector = 'load-more-trigger'
                driver.find_element(By.ID, css_selector).click()


                #Extracting the reviews and Title

                reviews = driver.find_elements(By.CSS_SELECTOR, 'div.review-container')

                review_title_list = []
                review_list = []

                for i in tqdm(reviews):
                    try:
                        sel2 = Selector(text = i.get_attribute('innerHTML'))

                        try:
                                review = sel2.css('.text.show-more__control::text').extract_first()

                        except:
                                    review = np.NaN
                        try:
                                review_title = sel2.css('a.title::text').extract_first().replace('\n', '')
                        except:
                                review_title = np.Nan

                        review_list.append(review)
                        review_title_list.append(review_title)

                    except Exception as e:
                        error_url_list.append(url)
                        error_msg_list.append(e)

                reviews_df = pd.DataFrame({
                            'Review': review_list,
                            'Title': review_title_list
                    })   

    except:
            pass

    reviews_df.to_csv('fetched_reviews.csv')
    return reviews_df




def preprocess():

    #Lowercasing

    reviews_and_title_temp['Review'] = reviews_and_title_temp['Review'].str.lower()


    # Removing HTML tags

    import re

    def remove_html(text):
        pattern = re.compile('<.*?>')
        return pattern.sub(r'', text)

    reviews_and_title_temp['Review'] = reviews_and_title_temp['Review'].apply(remove_html)
    # print(df.head())


    #Remove @

    def remove_at_the_rate(text):

        ls = []
        new = []

        ls = nlp(text)

        for word in ls:
            if word.text != "@":
                new.append(word.text)

        return ' '.join(new)

    reviews_and_title_temp['Review'] = reviews_and_title_temp['Review'].apply(remove_at_the_rate)



    #Removing URL

    import re

    def remove_url(text):
        pattern = re.compile(r'https?://\S+|www\.\S+')
        return pattern.sub(r'', text)

    reviews_and_title_temp['Review']= reviews_and_title_temp['Review'].apply(remove_url)
   


    #Remmove punctuation

    import string

    punc = string.punctuation

    def  remove_punc(text):

        return text.translate(str.maketrans('', '', punc))

    reviews_and_title_temp['Review']= reviews_and_title_temp['Review'].apply(remove_punc)



    from autocorrect import Speller

    check = Speller()

    def check_spell(text):

        return check(text)

    # train_df['Description_preprocessed'] = train_df['Description_preprocessed'].apply(check_spell)


    # Removing stop words

    stpwords = stopwords.words('english')

    def remove_stop_words(text):
        ls = []
        new = []

        ls = nlp(text)

        for word in ls:
            if word.text not in stpwords:

                new.append(word.text)

        return ' '.join(new)

    reviews_and_title_temp['Review'] = reviews_and_title_temp['Review'].apply(remove_stop_words)


    #Removing Contradictions

    import contractions

    def remove_contradictions(text):

        return " ".join([contractions.fix(word.text) for word in nlp(text)])

    reviews_and_title_temp['Review']= reviews_and_title_temp['Review'].apply(remove_contradictions)



    def Lemmetization(text):

        return " ".join([word.lemma_ for word in nlp(text)])



    reviews_and_title_temp['Review'] = reviews_and_title_temp['Review'].apply(Lemmetization)
    
    #Making the corpus
    
    def making_corpus_each(string):
        corpus = []
        corpus.append(string.split())

        return corpus
    
    reviews_and_title_temp['Review'] = reviews_and_title_temp['Review'].apply(making_corpus_each)
    
    global story 
    story = []
    
    def making_final_corpus(ls):
        story.append(ls)


    
    reviews_and_title_temp['Review'] = reviews_and_title_temp['Review'].apply(making_final_corpus)
    # print('----------------------------------------------------------------', len(story), story, story[0])
    
    #Converting the story list from 3d to 2d
    from itertools import chain
    global flatten_corpus 
    flatten_corpus = list(chain.from_iterable(story))
    # print('----------------------------------------------------------------', len(flatten_corpus), flatten_corpus[0], flatten_corpus)




def preprocess_strings(st):

    #Lowercasing

    st = st.lower()


    # Removing HTML tags

    import re

    def remove_html(text):
        pattern = re.compile('<.*?>')
        return pattern.sub(r'', text)

    st = remove_html(st)
    # print(df.head())


    #Remove @

    def remove_at_the_rate(text):

        ls = []
        new = []

        ls = nlp(text)

        for word in ls:
            if word.text != "@":
                new.append(word.text)

        return ' '.join(new)

    st = remove_at_the_rate(st)



    #Removing URL

    import re

    def remove_url(text):
        pattern = re.compile(r'https?://\S+|www\.\S+')
        return pattern.sub(r'', text)

    st = remove_url(st)
   


    #Remmove punctuation

    import string

    punc = string.punctuation

    def  remove_punc(text):

        return text.translate(str.maketrans('', '', punc))

    st = remove_punc(st)



    from autocorrect import Speller

    check = Speller()

    def check_spell(text):

        return check(text)

    # train_df['Description_preprocessed'] = train_df['Description_preprocessed'].apply(check_spell)


    # Removing stop words

    stpwords = stopwords.words('english')

    def remove_stop_words(text):
        ls = []
        new = []

        ls = nlp(text)

        for word in ls:
            if word.text not in stpwords:

                new.append(word.text)

        return ' '.join(new)

    st = remove_stop_words(st)


    #Removing Contradictions

    import contractions

    def remove_contradictions(text):

        return " ".join([contractions.fix(word.text) for word in nlp(text)])

    st = remove_contradictions(st)



    def Lemmetization(text):

        return " ".join([word.lemma_ for word in nlp(text)])



    st = Lemmetization(st)
      
    return st

    
final_embeddings = []
    
def embeddings_generation():

    #Building vocabulary
    vocabulary = set(word2vec_model.wv.index_to_key)
    
    
    for i in flatten_corpus:
        avg_embeddings = None
        for j in i:

            if j in vocabulary:

                if avg_embeddings is None:
                    avg_embeddings = word2vec_model.wv[j]
                else:
                    avg_embeddings = avg_embeddings + word2vec_model.wv[j]
        if avg_embeddings is not None:
            avg_embeddings = avg_embeddings / len(avg_embeddings)
            final_embeddings.append(avg_embeddings)

            
def fetch_corresponding_reviews(pos, neg):
    
    index = []
    review = []
    title = []
    
    for i in pos:
        
            index.append(i[0])
            review.append(reviews_and_title.iloc[i[0]]['Title'])
            title.append(reviews_and_title.iloc[i[0]]['Review'])
        
    pos_dict = ({

            'index':index,
            'Title': review,
            'Review': title
    })
        
        
    index = []
    review = []
    title = []
    
        
    for i in neg:
        
            index.append(i[0])
            review.append(reviews_and_title.iloc[i[0]]['Title'])
            title.append(reviews_and_title.iloc[i[0]]['Review'])
            
    neg_dict = ({

            'index':index,
            'Title': review,
            'Review':title
    })
        
    # print(neg_dict)
    # print(pos, neg)
    global pos_df 
    pos_df = pd.DataFrame(pos_dict) 
    global neg_df 
    neg_df = pd.DataFrame(neg_dict)
    
    
    
def senti_analysis(df):
    
    preprocess()
    
    embeddings_generation()
    
    # print(final_embeddings)
    y_pred = final_model.predict(final_embeddings)
    
    # print(y_pred)
    
    predict = list(enumerate(y_pred))
    # print(predict)
    pos = []
    neg = []
    
    for i in predict:
        if i[1] == 1:
            pos.append(i)
        elif i[1] == 0:
            neg.append(i)
    
  
    
    fetch_corresponding_reviews(pos, neg)
    
    
title = []

#Fetching the image of the movie
def fetch_list_of_movies():

    headers = {
       "User-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.200"
}
    for i in tqdm(range(1)):
        
        url = "https://www.themoviedb.org/movie?page={}".format(i)
        response = requests.get(url, headers=headers).text

        soup = BeautifulSoup(response, 'lxml')
        
        page_wrapper_div = soup.find_all('div', class_='page_wrapper')

        headings = []
        for item in page_wrapper_div:
            headings = item.find_all('h2')
        
        
        for i in headings:

            title.append(i.text)

    # print(title)
    # print(len(title))
    
    #Saving the data to a pickle file
    
    pickle.dump(title, open('title_list.pkl', 'wb'))


#Getting the homepage and image url    
def fetch_image(final_name):
    
    number = get_movie_id(final_name)
    url = 'https://api.themoviedb.org/3/find/tt{}?api_key=7f1ea36a1c2e4ec69b46aaa87b5d24d6&external_source=imdb_id'.format(number)
    headers = {
       "User-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.200"
}
   
    response = requests.get(url, headers=headers)
    
    data = response.json()

    image_url = 'https://image.tmdb.org/t/p/w400/' + data['movie_results'][0]['poster_path']
    homepage_url = 'https://www.themoviedb.org/movie/{}'.format(data['movie_results'][0]['id'])
    

    return image_url, homepage_url


#Getting the synopsis of the movie

def fetch_synopsis(final_name):
    
    number = get_movie_id(final_name)
    url = 'https://api.themoviedb.org/3/find/tt{}?api_key=7f1ea36a1c2e4ec69b46aaa87b5d24d6&external_source=imdb_id'.format(number)
    headers = {
       "User-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.200"
}
   
    response = requests.get(url, headers=headers)
    
    data = response.json()

    overview = data['movie_results'][0]['overview']
    
    return overview


#to be able to download a dataframe 
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')



#Getting the IMDB movie id
def get_movie_id(final_name):
    
    ia = imdb.Cinemagoer()
    movie = ia.search_movie(final_name)
    string = movie[0]
    movie_id = string.movieID
    return movie_id
    
    
#Giving our web app a title

st.markdown("<h1 style='text-align: center; color: white;'>Movies Review System with Sentiment Analysis</h1>", unsafe_allow_html=True)

# st.title("")


#Loading the title_list.pkl file

file_name = 'title_list.pkl'
open_file = open(file_name, "rb")
title_names = pickle.load(open_file)
open_file.close()


#Giving movie options to the user
name = st.selectbox("Choose your movie! ", options=title_names)
release_date = st.text_input("Please enter the entered movie's relaese date")

final_name = name + " " + "(" + release_date + ")"
print(final_name)





#Button to fetch reviews

if st.button('Get reviews!'): 

    #Progress bar

    bar = st.progress(0)
    
    tabs_titles = ['Movie', 'Sentiment Analysis of Reviews!', 'Visualization of the reviews!']
    
    tab2, tab3, tab4 = st.tabs(tabs_titles)
    
 
    with tab2:
        
        st.title(name)
        col1, col2 = st.columns([2, 2])
        
        with col1:


                img_url, home_url = fetch_image(final_name)
                st.image(img_url)
                st.write("[Explore](%s)" % home_url)


        bar.progress(30)
        time.sleep(1)

        with col2:
            st.markdown("<h3 style='text-align: center; color: white;'>Synopsis</h3>", unsafe_allow_html=True)
            text = fetch_synopsis(final_name)
            st.markdown("<h6 style='text-align: center; color: white;'>{}</h6>".format(text), unsafe_allow_html=True)

        bar.progress(50)
        time.sleep(1)

        reviews_and_title = fetch_reviews(final_name)
        reviews_and_title_temp = pd.DataFrame()
        reviews_and_title_temp['Review'] = reviews_and_title['Review']
        senti_analysis(reviews_and_title_temp)
        bar.progress(100)
        
            
        
    with tab3:
        
        with st.expander("All Reviews"):
        

                    
            st.dataframe(reviews_and_title, height=250, width=1000, hide_index=True)

            #Option to downlaod the csv file


            csv = convert_df(reviews_and_title)
            
            #THis reset the web page everytime so be careful!
            st.download_button(
                 label="Download data as CSV",
                 data=csv,
                 file_name='reviews_df.csv',
                 mime='text/csv',
             )
            
            
        with st.expander("Postive Reviews"):
        
            #Data
            
            st.dataframe(pos_df, height=250, width=1000, hide_index=True)
 
        
        with st.expander("Negative Reviews"):
            
            
            #Data
            st.dataframe(neg_df, height=250, width=1000, hide_index=True)
    
    with tab4:
          
        with st.expander(":bar_chart: Distribution fo Postive and Negative Reviews"):
             
            num_pos = pos_df.shape
            num_neg = neg_df.shape
            # print(num_neg, num_pos)
            dic = {
                'Criteria': ['Positive', 'Negative'],
                'Number': [num_pos[0], num_neg[0]]
            }
            
            chart_data = pd.DataFrame(dic)
            st.dataframe(chart_data, height=250, width=1000, hide_index=True)
#             fig = px.bar(
#                 x=chart_data.iloc[0],
#                 y=chart_data.iloc[1],
                
#             )
#             st.plotly_chart(fig)


        with st.expander("Most frequent words in Postitve reviews"):
        
            pos_corpus= []
            neg_corpus = []
     
            for i in pos_df['Review'].tolist():
                
                string_preprocessed = preprocess_strings(i)
                ls = string_preprocessed.split()
                print('neg:', string_preprocessed)
                
                for j in ls:
                    
                    pos_corpus.append(j)
                    
            for i in neg_df['Review'].tolist():
                
                string_preprocessed = preprocess_strings(i)
                ls = string_preprocessed.split()
                print('neg:', string_preprocessed)
                
                for j in ls:
                    
                    neg_corpus.append(j)

            from collections import Counter
            
            fig = px.bar(
                x=pd.DataFrame(Counter(pos_corpus).most_common(30))[0],
                y=pd.DataFrame(Counter(pos_corpus).most_common(30))[1],
                
            )
            st.plotly_chart(fig)
            
            
        
        with st.expander("Most frequent words in Negative reviews"):
  
            fig = px.bar(
                x=pd.DataFrame(Counter(neg_corpus).most_common(30))[0],
                y=pd.DataFrame(Counter(neg_corpus).most_common(30))[1],
               
            )
            st.plotly_chart(fig)


#ADD A REVIEWER'S RATING CHART TOO WITH AN AVERAGE RATING TOO
    
# fetch_list_of_movies()






