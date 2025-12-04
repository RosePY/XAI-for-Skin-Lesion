"""
To run install:
-pip install onnx==1.8.0
-pip install pytorch2keras 
It gave error so, name policy was changed
"""

import numpy as np
import torch
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision.models as models

iv4 = 'results-comet-iv4/iv4_i18nf_5runs_4/checkpoints/model_best.pth'

rn50 = 'results-comet-iv4/rn50_i18nf_5runs_5/checkpoints/model_best.pth'

def main():
	model_path = rn50#'results-comet-iv4/iv4_isic2018_t3/checkpoints/model_best.pth'

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
	print(input_np.shape)
	input_var = Variable(torch.FloatTensor(input_np)).to(device)
	model = torch.load(model_path)

	model.to(device)
	model.eval()
	k_model = \
	    pytorch_to_keras(model, input_var, [(3, 224, 224,)], verbose=True, change_ordering=True,name_policy='renumerate')

	for i in range(3):
	    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
	    input_var = Variable(torch.FloatTensor(input_np)).to(device)
	    input_var.cpu()
	    output = model(input_var)
	    pytorch_output = output.cpu().data.numpy()
	    keras_output = k_model.predict(np.transpose(input_np, [0, 2, 3, 1]))
	    error = np.max(pytorch_output - keras_output)
	    print('error -- ', error)  # Around zero :)

	#k_model.save('keras_model')
	k_model.save('keras_h5/rn50_t3i2018_nf.h5')#iv4_t3i2018_nf.h5'

def to_onnx():
	model_path = 'results-comet-iv4/iv4_isic2018_t3/checkpoints/model_best.pth'

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	#input_np = np.random.uniform(0, 1, (1, 3, 299, 299))
	dummy_input = torch.randn(1, 3, 299, 299)
	input_var = dummy_input.to(device)
	model = torch.load(model_path)

	model.to(device)
	model.eval()
	input_names = ["actual_input"]
	output_names = ['output']
	torch.onnx.export(model, 
                  input_var,
                  "onnx/iv4t32018.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  )
                  
main()
