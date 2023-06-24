from sklearn.model_selection import train_test_split
from log.Applogger import Applogger
from preprocessing.preprocess import pre
from sklearn.svm import SVC
from file.file_op import file_op



class train:
    def __init__(self):
        self.log=Applogger()
        self.preprcess=pre()
        self.file=file_op()
    def data(self):
        try:
             X,y=self.preprcess.processing()
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
             sv=SVC()
             sv.fit(X_train,y_train)
             self.file.save_model(sv,"SVM_Model")
        except Exception as e:
            raise e





c=train()
c.data()




