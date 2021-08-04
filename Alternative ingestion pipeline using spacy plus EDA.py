# import libraries
import gensim
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import regex as re
import seaborn as sns

import en_core_web_sm, en_core_web_trf

from wordcloud import WordCloud
from gensim.corpora import Dictionary

# load the language models needed
nlp_sm = en_core_web_sm.load()
nlp_trf = en_core_web_trf.load()

# just a peculiarity my system needs at the moment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# load the data
vodafone_reviews = pd.read_csv('vodafone_reviews.csv')

### Very Preliminary EDA ###
# simple bar chart showing the reviews distribution
reviews = pd.value_counts(vodafone_reviews.score.values).sort_index()
plt.bar(x=reviews.index, height=reviews.values, color='#990000')
plt.xlabel('customer rating')
plt.ylabel('number of reviews')

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.show()

# alternatively... NPS-like
# define functions to classify each review into an NPS group
def create_NPS_group(row):
    if row.score <= 3:
        group = 'Detractor'
    elif row.score == 4:
        group = 'Passive'
    else:
        group = 'Promoter'
    return group

vodafone_reviews['nps_group'] = vodafone_reviews.apply(create_NPS_group, axis=1)

def create_NPS_class(row):
    if row.nps_group == 'Detractor':
        nps_class = -1
    elif row.nps_group == 'Passive':
        nps_class = 0
    else:
        nps_class = 1
    return nps_class

vodafone_reviews['nps_class'] = vodafone_reviews.apply(create_NPS_class, axis=1)

# plot the distribution of reviews by NPS group
reviews2 = pd.value_counts(vodafone_reviews.nps_group.values).sort_index()
plt.bar(x=reviews2.index, height=reviews2.values, color='#990000')
plt.xlabel('NPS group')
plt.ylabel('number of reviews')

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.show()

# calculate character and word lengths of titles and reviews
vodafone_reviews['title_num_chars'] = vodafone_reviews.title.apply(lambda x: len(x))
vodafone_reviews['review_num_chars'] = vodafone_reviews.review.apply(lambda x: len(x))

vodafone_reviews['title_num_words'] = vodafone_reviews.title.apply(lambda x: len(re.findall(r'\w+', x)))
vodafone_reviews['review_num_words'] = vodafone_reviews.review.apply(lambda x: len(re.findall(r'\w+', x)))

# plot the distribution of character and word lengths
plt.rcParams.update({'font.size': 9})

fig, axs = plt.subplots(1, 2)
axs[0].hist(vodafone_reviews.title_num_chars, bins=70, edgecolor='#E60000', color='#990000')
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].set_xlabel('Number of characters')
axs[0].set_ylabel('Number of titles')
axs[1].hist(vodafone_reviews.review_num_chars, bins=70, edgecolor='#E60000', color='#990000')
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].set_xlabel('Number of characters')
axs[1].set_ylabel('Number of reviews');
plt.show()

fig, axs = plt.subplots(1, 2)
axs[0].hist(vodafone_reviews.title_num_words, bins=25, edgecolor='#E60000', color='#990000')
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].set_xlabel('Number of words')
axs[0].set_ylabel('Number of titles')
axs[1].hist(vodafone_reviews.review_num_words, bins=70, edgecolor='#E60000', color='#990000')
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].set_xlabel('Number of words')
axs[1].set_ylabel('Number of reviews');
plt.show()

# plot 1 detractors vs promoters titles
# plot 2 detractors vs promoters reviews
promoter_detractor_reviews = vodafone_reviews[vodafone_reviews.nps_group.isin(["Promoter", "Detractor"])]

# characters
fig, axs = plt.subplots(1, 2)
p = sns.kdeplot(ax=axs[0], data=promoter_detractor_reviews, x='title_num_chars', hue='nps_group', fill=True,
                common_norm=False, palette=['#990000', '#4a4d4e'], legend=False)
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Number of characters in title")
annot = pd.DataFrame({
    'x': [30, 80],
    'y': [0.027, 0.0025],
    'text': ["Promoters", "Detractors",]
})
for point in range(0,len(annot)):
     p.text(annot.x[point], annot.y[point], annot.text[point], horizontalalignment='left')

p2 = sns.kdeplot(ax=axs[1], data=promoter_detractor_reviews, x='review_num_chars', hue='nps_group', fill=True,
                common_norm=False, palette=['#990000', '#4a4d4e'], legend=False)
p2.spines['right'].set_visible(False)
p2.spines['top'].set_visible(False)
p2.set_xlabel("Number of characters in review")
annot = pd.DataFrame({
    'x': [500, 1500],
    'y': [0.00175, 0.0002],
    'text': ["Promoters", "Detractors",]
})
for point in range(0,len(annot)):
     p2.text(annot.x[point], annot.y[point], annot.text[point], horizontalalignment='left')

# words
fig, axs = plt.subplots(1, 2)
p = sns.kdeplot(ax=axs[0], data=promoter_detractor_reviews, x='title_num_words', hue='nps_group', fill=True,
                common_norm=False, palette=['#990000', '#4a4d4e'], legend=False)
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Number of words in title")
annot = pd.DataFrame({
    'x': [8, 12],
    'y': [0.15, 0.02],
    'text': ["Promoters", "Detractors",]
})
for point in range(0,len(annot)):
     p.text(annot.x[point], annot.y[point], annot.text[point], horizontalalignment='left')

p2 = sns.kdeplot(ax=axs[1], data=promoter_detractor_reviews, x='review_num_words', hue='nps_group', fill=True,
                common_norm=False, palette=['#990000', '#4a4d4e'], legend=False)
p2.spines['right'].set_visible(False)
p2.spines['top'].set_visible(False)
p2.set_xlabel("Number of words in review")
annot = pd.DataFrame({
    'x': [150, 300],
    'y': [0.009, 0.001],
    'text': ["Promoters", "Detractors",]
})
for point in range(0,len(annot)):
     p2.text(annot.x[point], annot.y[point], annot.text[point], horizontalalignment='left')

# calculate and plot the correlation between nps_group, customer rating, and title and review lengths
numeric_features = vodafone_reviews.loc[:, ['score', 'nps_class', 'title_num_chars', 'review_num_chars', 'title_num_words', 'review_num_words']]
corr = numeric_features.corr()

fig, ax = plt.subplots()
mask = np.zeros_like(numeric_features.corr())
mask[np.triu_indices_from(mask)] = 1
sns.heatmap(numeric_features.corr(), mask=mask, ax=ax, annot=True, cmap='Reds')

### Data Cleansing and Normalisation Pipeline ###

def preprocess_review_titles():
    # convert the text to lower-case
    vodafone_reviews['lower_title'] = vodafone_reviews.title.str.lower()

    # correct curly apostrophes
    vodafone_reviews.lower_title = vodafone_reviews.lower_title.str.replace("’", "'", regex=False)

    # create a dictionary of common expansions in the english language
    contractions_dict = {"can't": "can not",
                         "won't": "will not",
                         "don't": "do not",
                         "n't":" not",
                         "'m":" am",
                         "'ll":" will",
                         "'d":" would",
                         "'ve":" have",
                         "'re":" are",
                         "'s": ""} # 's could be 'is' or could be possessive: it has no expansion

    # expand the contractions and add to dataframe as new variable
    exp_reviews = []
    for title in vodafone_reviews.lower_title:
        t = []
        for key, value in contractions_dict.items():
            if key in title:
                title = title.replace(key, value)
                t.append(title)
        exp_reviews.append(t)

    vodafone_reviews['clean_title'] = exp_reviews
    vodafone_reviews.clean_title = vodafone_reviews.apply(lambda x: x.clean_title[0] if len(x.clean_title) > 0 else x.lower_title, axis = 1)

    # the same approach could be applied to common miss-spellings, colloqualisms etc
    # part-of-speech tagging
    words = []
    poss = []
    pos_tags = []
    ner_types = []
    for title in vodafone_reviews.clean_title:
        word = []
        pos = []
        pos_tag = []
        ner_type = []
        t = nlp_trf(title)
        for w in t:
            word.append(w.text)
            pos.append(w.pos_)
            pos_tag.append(w.tag_)
            ner_type.append(w.ent_type_)
        words.append(word)
        poss.append(pos)
        pos_tags.append(pos_tag)
        ner_types.append(ner_type)

    vodafone_reviews['title_words'] = words
    vodafone_reviews['title_pos'] = poss
    vodafone_reviews['title_pos_tags'] = pos_tags
    vodafone_reviews['title_ner_types'] = ner_types

    # pulling out named entities (organisations, money, dates etc)
    ent_texts = []
    ent_labels = []
    for title in nlp_trf.pipe(vodafone_reviews.clean_title):
        ent_text = []
        ent_label = []
        for ent in title.ents:
            ent_text.append(ent.text)
            ent_label.append(ent.label_)
        ent_texts.append(ent_text)
        ent_labels.append(ent_label)

    vodafone_reviews['title_ent_text'] = ent_texts
    vodafone_reviews['title_ent_label'] = ent_labels

    # check out spacy's stopword list and modify as necessary
    spacy_stopwords = nlp_sm.Defaults.stop_words # stopwords are the same irrespective of the English language model used

    preproc_titles = []
    preproc_poss = []
    preproc_pos_tags = []
    for title in vodafone_reviews.clean_title:
        titles = []
        pos = []
        pos_tag = []
        t = nlp_trf(title)
        for w in t:
            if not w.is_stop and not w.is_punct and not w.is_digit and not w.is_space:
                titles.append(w.lemma_)
                pos.append(w.pos_)
                pos_tag.append(w.tag_)
        preproc_titles.append(titles)
        preproc_poss.append(pos)
        preproc_pos_tags.append(pos_tag)

    vodafone_reviews['preproc_title'] = preproc_titles
    vodafone_reviews['preproc_pos'] = preproc_poss
    vodafone_reviews['preproc_pos_tag'] = preproc_pos_tags

    title_bigram_model = gensim.models.Phrases(preproc_titles)
    title_bigrams = [title_bigram_model[title] for title in preproc_titles]

    title_trigram_model = gensim.models.Phrases(title_bigrams)
    title_trigrams = [title_trigram_model[title] for title in title_bigrams]

    vodafone_reviews['preproc_title_bigrams'] = title_bigrams
    vodafone_reviews['preproc_title_trigrams'] = title_trigrams

    return vodafone_reviews

def preprocess_review_text():
    # convert the text to lower-case
    vodafone_reviews['lower_review'] = vodafone_reviews.review.str.lower()

    # correct curly apostrophes
    vodafone_reviews.lower_review = vodafone_reviews.lower_review.str.replace("’", "'", regex=False)

    # create a dictionary of common expansions in the english language
    contractions_dict = {"can't": "can not",
                         "won't": "will not",
                         "don't": "do not",
                         "n't":" not",
                         "'m":" am",
                         "'ll":" will",
                         "'d":" would",
                         "'ve":" have",
                         "'re":" are",
                         "'s": ""} # 's could be 'is' or could be possessive: it has no expansion

    # expand the contractions and add to dataframe as new variable
    exp_reviews = []
    for review in vodafone_reviews.lower_review:
        t = []
        for key, value in contractions_dict.items():
            if key in review:
                review = review.replace(key, value)
                t.append(review)
        exp_reviews.append(t)

    vodafone_reviews['clean_review'] = exp_reviews
    vodafone_reviews.clean_review = vodafone_reviews.apply(lambda x: x.clean_review[0] if len(x.clean_review) > 0 else x.lower_review, axis = 1)

    # the same approach could be applied to common miss-spellings, colloqualisms etc
    # part-of-speech tagging
    words = []
    poss = []
    pos_tags = []
    ner_types = []
    for review in vodafone_reviews.clean_review:
        word = []
        pos = []
        pos_tag = []
        ner_type = []
        t = nlp_trf(review)
        for w in t:
            word.append(w.text)
            pos.append(w.pos_)
            pos_tag.append(w.tag_)
            ner_type.append(w.ent_type_)
        words.append(word)
        poss.append(pos)
        pos_tags.append(pos_tag)
        ner_types.append(ner_type)

    vodafone_reviews['review_words'] = words
    vodafone_reviews['review_pos'] = poss
    vodafone_reviews['review_pos_tags'] = pos_tags
    vodafone_reviews['review_ner_types'] = ner_types

    # pulling out named entities (organisations, money, dates etc)
    ent_texts = []
    ent_labels = []
    for review in nlp_trf.pipe(vodafone_reviews.clean_review):
        ent_text = []
        ent_label = []
        for ent in review.ents:
            ent_text.append(ent.text)
            ent_label.append(ent.label_)
        ent_texts.append(ent_text)
        ent_labels.append(ent_label)

    vodafone_reviews['review_ent_text'] = ent_texts
    vodafone_reviews['review_ent_label'] = ent_labels

    # check out spacy's stopword list and modify as necessary
    spacy_stopwords = nlp_sm.Defaults.stop_words # stopwords are the same irrespective of the English language model used

    preproc_reviews = []
    preproc_poss = []
    preproc_pos_tags = []
    for review in vodafone_reviews.clean_review:
        reviews = []
        pos = []
        pos_tag = []
        t = nlp_trf(review)
        for w in t:
            if not w.is_stop and not w.is_punct and not w.is_digit and not w.is_space:
                reviews.append(w.lemma_)
                pos.append(w.pos_)
                pos_tag.append(w.tag_)
        preproc_reviews.append(reviews)
        preproc_poss.append(pos)
        preproc_pos_tags.append(pos_tag)

    vodafone_reviews['preproc_review'] = preproc_reviews
    vodafone_reviews['preproc_pos'] = preproc_poss
    vodafone_reviews['preproc_pos_tag'] = preproc_pos_tags

    review_bigram_model = gensim.models.Phrases(preproc_reviews)
    review_bigrams = [review_bigram_model[review] for review in preproc_reviews]

    review_trigram_model = gensim.models.Phrases(review_bigrams)
    review_trigrams = [review_trigram_model[review] for review in review_bigrams]

    vodafone_reviews['preproc_review_bigrams'] = review_bigrams
    vodafone_reviews['preproc_review_trigrams'] = review_trigrams

    return vodafone_reviews

# caution, these two functions can take a little while to run
preprocess_review_titles()
preprocess_review_text()


### More EDA - wordclouds ###
def create_title_corpus():
    title_corpus = vodafone_reviews.loc[:, ['nps_group', 'preproc_title_bigrams']]
    title_corpus['title_strings'] = title_corpus.preproc_title_bigrams.apply(lambda x: ' '.join(x) if x != '' else x)
    title_text = ' '.join('' if pd.isna(title) else title for title in title_corpus.title_strings)

    promoter_title_corpus = title_corpus[title_corpus.nps_group == 'Promoter']
    promoter_title_text = ' '.join('' if pd.isna(title) else title for title in promoter_title_corpus.title_strings)

    detractor_title_corpus = title_corpus[title_corpus.nps_group == 'Detractor']
    detractor_title_text = ' '.join('' if pd.isna(title) else title for title in detractor_title_corpus.title_strings)

    return title_text, promoter_title_text, detractor_title_text

# overall word cloud
wordcloud_titles = WordCloud(max_font_size=30, max_words=100000, random_state=2021, scale=2,
                             background_color='white', contour_width=3, colormap='RdGy').generate(title_text)
plt.imshow(wordcloud_titles, interpolation='bilinear')
plt.axis("off")

# promoter word cloud
wordcloud_promoter_titles = WordCloud(max_font_size=30, max_words=100000, random_state=2021, scale=2,
                             background_color='white', contour_width=3, colormap='Reds').generate(promoter_title_text)
plt.imshow(wordcloud_promoter_titles, interpolation='bilinear')
plt.axis("off")

# detractor word cloud
wordcloud_detractor_titles = WordCloud(max_font_size=30, max_words=100000, random_state=2021, scale=2,
                             background_color='white', contour_width=3, colormap='Greys').generate(detractor_title_text)
plt.imshow(wordcloud_detractor_titles, interpolation='bilinear')
plt.axis("off")


# to do:
- examine term frequencies using a Count Vectorizer (TFIDF not required as the texts are relatively short)
- examine the correlation between rating and review length (hypothesis is that complaints are longer than prais)




# create dictionaries and corpuses from the pre-processed text
title_dictionary = Dictionary(title_bigrams)
review_dictionary = Dictionary(review_bigrams)

title_corpus = [title_dictionary.doc2bow(text) for text in title_bigrams]
review_corpus = [review_dictionary.doc2bow(text) for text in review_bigrams]

# to do, create dictionaries
# word clouds of promoters vs detractors
# display most common words/phrases (in both pure count and tf-idf forms)

https://www.kdnuggets.com/2019/05/complete-exploratory-data-analysis-visualization-text-data.html
https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools
https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
# https: // medium.com / bigpanda - engineering / exploratory - data - analysis - for -text - data - 29cf7dd54eb8

# use count and tfidf vectorisers to get top words and phrases in corpus - see hashed link (saved as pdf on desktop)