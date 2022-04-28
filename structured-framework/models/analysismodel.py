# This is an abstract class for handling a specific model on a specific task
class analysismodel:
    # returns the model
    def getmodel(self):
        return self.model

    # returns the modality names
    def getmodalitynames(self):
        return self.modalitynames

    def getmodalitytypes(self):
        return self.modalitytypes

    # given a data instance and a modality name, return the unimodal data of that specific modality in this instance
    def getunimodaldata(self, datainstance, modality):
        raise NotImplementedError

    # compute the output of the model with the given datainstance, returning a result object
    def forward(self, datainstance):
        raise NotImplementedError

    # compute the output of the model with the given list of datainstance, returning a list of result objects (this may allow batching)
    def forwardbatch(self, datainstances):
        raise NotImplementedError

    # get the size of the pre-softmax logit
    def getlogitsize(self):
        raise NotImplementedError

    # returns the pre-softmax logits of the result
    def getlogit(self, resultobj):
        raise NotImplementedError

    # returns the prediction labels of the result
    def getpredlabel(self, resultobj):
        raise NotImplementedError

    # returns the pre-linear layer result of a forward computation (used for sparse linear encoding)
    def getprelinear(self, resultobj):
        raise NotImplementedError

    # returns the correct label for an instance
    def getcorrectlabel(self, datainstance):
        raise NotImplementedError

    # replace data in one modality in an instance with a different data
    def replaceunimodaldata(self, datainstance, modality, newinput):
        raise NotImplementedError

    # get size of prelinear
    def getprelinearsize(self):
        raise NotImplementedError

    # get grad w.r.t. one modality input
    def getgrad(self, datainstance, target):
        raise NotImplementedError
