import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')
nltk.download('wordnet')

from nltk import word_tokenize
from nltk.corpus import cmudict, wordnet

import spacy

# Load Spacy for Part of Speech
en_nlp = spacy.load("en_core_web_sm")

# Load CMU Pronouncing Dictionary for syllable counting
d = cmudict.dict()

def nsyl(word):

    """ 
    Returns the number of syllables in a word using the CMU Pronouncing Dictionary. 
    
    Args:
      word: the word to process
    
    Return:
      int: the number of the syllabels
    """
    
    if word.lower() in d:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    else:
        return None  # If the word is not found, return 0


def morphological_pos(word):

    """
    Returns the POS and POS form for a word. Used for linguistic analysis.

    (TODO) More form of difference part of speech could be added 

    Args:
      word: the word to process
    
    Return:
      str: {POS}_{special form}

    """
    
    doc  = en_nlp(word)
    results = []
    for token in doc:

        pos = token.pos_
        morph = token.morph
        if pos == "NOUN":
            pos = pos + "_" + morph.get("Number")[0]
        elif pos == "VERB":
            pos = pos + "_" + morph.get("VerbForm")[0]
    
    return pos

def orthographic_complexity(word):

    """
    To calculate orthographic_complexity that could affect pronunciation
    for a word. Used for linguistic analysis.

    (TODO) More rules could be added. Calculation method to be adjusted.

    Args:
      word: the word to process
    
    Return:
      float: complexcity score

    """

    # silent letters (example)
    silent_letters = ['kn', 'w', 'b', 'pn', 'ps']

    # multigraphs (example)
    multigraphs = ['sh', 'ch', 'th', 'ph', 'gh']

    complexity_score = 0

    for letter in silent_letters:
        if word.startswith(letter):
            complexity_score += 1

    for mg in multigraphs:
        complexity_score += word.count(mg)

    return complexity_score

def get_lexical_features(word):

    """
    Obtain all lexical features for a word.

    Args:
      word: the word to process
    
    Return:
      dict: key-pairs of multiple lexical features


    """

    word_length = len(word)
    syllables_counts = nsyl(word)
    pos_tags = morphological_pos(word)
    ortho = orthographic_complexity(word)

    return {'word_length': word_length, 'syllables_counts': syllables_counts,\
            'pos_tags': pos_tags, 'ortho_complexity': ortho
            }
