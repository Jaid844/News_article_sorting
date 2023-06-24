import spacy
import string


class spa_:

 @staticmethod
 def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    nlp = spacy.load('en_core_web_sm-3.5.0')
    stop_words = nlp.Defaults.stop_words
    punctuations = string.punctuation
    doc = nlp(sentence)

    # print(doc)
    # print(type(doc))

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [word.lemma_.lower().strip() for word in doc]

    # print(mytokens)

    # Removing stop words
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

    sentence = " ".join(mytokens)
    # return preprocessed list of tokens
    return sentence