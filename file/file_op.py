import pickle
import os
import shutil
from log.Applogger import Applogger



class file_op:
    def __init__(self):
        self.model='models/'
        self.log=Applogger()

    def save_model(self,model, filename):
        self.file = open("Training_Logs/file.txt", 'w')
        self.log.log("Making trainingfolder", self.file)
        try:
            path=os.path.join(self.model,filename)
            if os.path.isdir(path):
                shutil.rmtree(path)
                os.makedirs(path)
            else:
                os.makedirs(path)
            with open(path+'/'+filename+'.sav','wb') as f:
                pickle.dump(model,f)
            self.log.log("Model have been sent",self.file)

        except Exception as e:
            raise e

    @staticmethod
    def load_model(filename):
        model="models/"
        try:
            with open(model + filename + '/' + filename + '.sav',
                      'rb') as f:

                return pickle.load(f)
        except Exception as e:

            raise Exception()
