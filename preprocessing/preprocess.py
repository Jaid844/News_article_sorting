from sentence_transformers import SentenceTransformer
from log.Applogger import Applogger
import pandas as pd
from spacy_file.spacy_dir import spa_





class pre:
    def __init__(self):
        self.log=Applogger()
        self.spacy_dir=spa_()


    def processing(self):
        try:
             self.file = open("Training_Logs/preprocess.txt", 'w')
             self.log.log(self.file,"Preprocessing the file for training purpose")
             model = SentenceTransformer('all-MiniLM-L6-v2')
             data = pd.read_csv('BBC_NewsTrain.csv')
             data = data[:100]
             possible_label = data["Category"].unique()
             label_dict = {}
             for index, label in enumerate(possible_label):
                 label_dict[label] = index
             data['label'] = data["Category"].replace(label_dict)
             data['tokenize']=data['Text'].apply(self.spacy_dir.spacy_tokenizer)
             data['embeding']=data['tokenize'].apply(model.encode)
             X = data['embeding'].to_list()
             y = data['label'].to_list()
             return X,y
        except Exception as e:
            raise e
        self.file = open("Training_Logs/preprocess.txt", 'w')
        self.log.log(self.file,str(e))








c=pre()
c.processing()



