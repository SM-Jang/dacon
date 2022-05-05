import torch
import torch.nn as nn
import pretrainedmodels
import pytorch_model_summary


class Model(nn.Module):
    def __init__(self, model_name, output_shape):
        super(Model, self).__init__()
        self.model = self._get_pretrainedmodel(model_name, output_shape)
        
    def _get_pretrainedmodel(self, model_name, output_shape):
        try:
            model = pretrainedmodels.__dict__[model_name](num_classes=1000,pretrained='imagenet')
            model.last_linear = nn.Linear(in_features=512, out_features=output_shape, bias=True)
            print("Feature extractor:", model_name)
            print(pytorch_model_summary.summary(model, torch.zeros(1,3,256,256), show_input=True))
            return model
        except:
            raise ("Invalid model name. Check the config file.")

    def forward(self, x):
        x = self.model(x)
        return x

