# import libraries
import gensim
import matplotlib.pyplot as plt
import pandas as pd

import en_core_web_sm, en_core_web_trf

from gensim.corpora import Dictionary

# load the language models needed
nlp_sm = en_core_web_sm.load()
nlp_trf = en_core_web_trf.load()

# load the data
vodafone_reviews = pd.read_csv('vodafone_reviews.csv')

# plot a simple bar chart showing the reviews distribution
reviews = pd.value_counts(vodafone_reviews.score.values).sort_index()
plt.bar(x=reviews.index, height=reviews.values, color='#990000')
plt.xlabel('customer rating')
plt.ylabel('number of reviews')

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.show()

# alternatively... NPS-like
# define a function to classify each review into an NPS group
def create_NPS_group(row):
    if row.score <= 3:
        group = 'Detractor'
    elif row.score == 4:
        group = 'Passive'
    else:
        group = 'Promoter'
    return group

vodafone_reviews['nps_group'] = vodafone_reviews.apply(create_NPS_group, axis=1)

reviews2 = pd.value_counts(vodafone_reviews.nps_group.values).sort_index()
plt.bar(x=reviews2.index, height=reviews2.values, color='#990000')
plt.xlabel('NPS group')
plt.ylabel('number of reviews')

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.show()

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
            if not w.is_stop and not w.is_punct and not w.is_digit:
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

    return title_bigrams

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
            if not w.is_stop and not w.is_punct and not w.is_digit:
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

    return review_bigrams


preprocess_review_titles()
preprocess_review_text()

# create dictionaries and corpuses from the pre-processed text
title_dictionary = Dictionary(title_bigrams)
review_dictionary = Dictionary(review_bigrams)

title_corpus = [title_dictionary.doc2bow(text) for text in title_bigrams]
review_corpus = [review_dictionary.doc2bow(text) for text in review_bigrams]

# to do, create dictionaries
# calculate string statistics (average length etc)
# word clouds of promoters vs detractors
# display most common words/phrases (in both pure count and tf-idf forms)
