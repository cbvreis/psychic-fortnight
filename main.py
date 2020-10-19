import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pdfplumber as pdfplumber
import textract as txt
import re
import os

from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from string import punctuation
from nltk.probability import FreqDist
from collections import defaultdict

# NLP Pkgs
import spacy_streamlit
import spacy

nlp = spacy.load('pt')

import os
from PIL import Image


# import nltk
# nltk.download('punkt')
from heapq import nlargest
import streamlit as st
import warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd


def npl_resume(str):
    sentencas = sent_tokenize(str)
    palavras = word_tokenize(str.lower())
    f = open('stopwords.txt', 'r')
    _stopwords = f.read().split()
    f.close()

    stopwords = set(_stopwords + list(punctuation))
    palavras_sem_stopwords = [palavra for palavra in palavras if palavra not in stopwords]

    frequencia = FreqDist(palavras_sem_stopwords)
    sentencas_importantes = defaultdict(int)
    for i, sentenca in enumerate(sentencas):
        for palavra in word_tokenize(sentenca.lower()):
            if palavra in frequencia:
                sentencas_importantes[i] += frequencia[palavra]
    idx_sentencas_importantes = nlargest(4, sentencas_importantes, sentencas_importantes.get)


#   for i in sorted(idx_sentencas_importantes):
#       print(sentencas[i])

def extract_data(feed):
    data = []
    with pdfplumber.load(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            data.append(p.extract_text())
    return data


def count(str):
    f = open('stopwords.txt', 'r')
    stop_words = f.read().split()

    f.close()
    # stop_words = frozenset([data])
    finalCount = Counter()
    for line in str.split():
        words = [w for w in line.split(" ") if w not in stop_words]
        finalCount.update(words)  # update final count using the words list
    return finalCount


def remove_characters(str):
    # substituir caracteres especiais
    # str = re.sub(r'\W+', ' ', str)
    str = str.lower()
    return str





def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)




def main():
    menu = ["WORD_CLOUD", "TOKENIZACAO","RESUMO"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "WORD_CLOUD":
        try:
            st.title("Análise de texto - Arquivos PDF")
            uploaded_file = st.file_uploader("", type="pdf")

            if uploaded_file is not None:
                # print(uploaded_file)
                df = extract_data(uploaded_file)
    #            docx = nlp(str(df))
    #            spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe('ner').labels)


                # Create and generate a word cloud image:
                wordcloud = WordCloud().generate(str(df))

                # Display the generated image:
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.show()
                st.pyplot()
        except:
            pass

    elif choice == "TOKENIZACAO":
        try:
            st.title("Análise de texto - Arquivos PDF")
            uploaded_file = st.file_uploader("", type="pdf")

            if uploaded_file is not None:
                # print(uploaded_file)
                df = extract_data(uploaded_file)
                docx = nlp(str(df))
                spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe('ner').labels)

        except:
            pass

    elif choice == "PDF":
        st.title("Análise de texto - Arquivos PDF")
        uploaded_file = st.file_uploader("", type="pdf")

        if uploaded_file is not None:
            # print(uploaded_file)
            df = extract_data(uploaded_file)
            docx = nlp(str(df))
            spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe('ner').labels)
            #    st.write('You selected `%s`' % filename)
            # doc = remove_characters(_txt.decode('raw_unicode_escape'))
            # print(doc)
            # npl_resume(doc)

if __name__ == '__main__':
    main()
