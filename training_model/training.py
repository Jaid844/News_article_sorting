from file.file_op import file_op


class module:

    def __init__(self):
        self.file=file_op()

    @staticmethod
    def training_module():
        model=file_op().load_model('SVM_Model')
        return model
