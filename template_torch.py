import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

#### First step : Dataloader ####									#
## your dataset											#
													#
class MyData(Dataset):											#
													#
	def __init__(self , data1_file_path , data2_file_dir): # For example: data1 is label/ 	#	first step : build your dataset
		self.data1 = pd.csv_read(data1_file_path)	 # and save in file path,data2/ 	#			
		self.data2_dir = data2_file_dir		 # is save under dir	       /	#			
													#			
	def __getitem__(self , index):	## get values in Dataset given the array index	#			↓
		label = self.data1[index]								#			
		data2_path = os.path.join(self.dara2_dir , data2_file_name#(get from other ways)#)	#			
		feature(AKA:data2) = pd.read_csv(data2_path)						#			
		return feature , label									#			↓
													#
	def __len__(self):			## shape info of dataset				#
		return self.len									#			↓
													#
dataset = MyData('mydata/mydata.csv' , '/home/xinqi/mydata')						#
													#			↓
train_dataloader = DataLoader(dataset , batch_size=64 , shuffle=True)				#
test_dataloader = DataLoader( ##same method as traindata## )						#
													#			↓
													#################################################
 													#	
class Net(torch.nn.Moudle):										#	second step: design your model
													#
	def __init__(self):										#		model sequential (structure)
		super(Net , self).__init__()								#			
		self.linear_relu_stack = nn.Sequential(						#			↓
            						nn.Linear(28*28, 512),				#	
            						nn.ReLU(),					#			↓
            						nn.Linear(512, 512),				#
            						nn.ReLU(),					#			↓
            						nn.Linear(512, 10),				#
        						)						#			↓
													#
	def forward(self , x):										#		   forward loop
		x = self.flatten(x)									#
		pred_y = self.linear_relu_stack(x)							#			↓
		return pred_y										#
													#			↓
model = Net()												#		instantiate your model
													#
####################################									#			↓
# move to GPU version if you want									#
####################################									#			↓
#													#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")				#		move your model to gpu
# model.to(device)											#
#													#			↓
####################################									######################################################
													#
													# third step : define the hyperparam, loss and optimizer
													#			↓
learning_rate = 1e-3											#		 hyperparameter
batch_size = 64											#
epochs = 5												#			↓
													#
loss_fn = nn.CrossEntropyLoss()									#		  loss function
													#			↓
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)					#		     optimizer
													#####################################################
def train_loop(dataloader , model , loss_fn , optimizer):						#		   training loop
	size = len(dataloader.dataset)								#
	for batch , (X ,y) in enumerate(dataloader):							#			↓
		X , y = X.to(device) , y.to(device)							#		    out to gpu
													#
		pred = model(X)									#			↓
		loss = loss_fn(pred , y)								#		      compute
													#
		optimizer.zero_grad()									#			↓			
		loss.backward()									#		     back propagation
		optimizer.step()									#
													#
		#print frequency : report result every 300 times300					#
        	if batch_idx % 300 == 299:								#			↓
            	print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, loss / 2000))		#		   report result
            												#
													#########################################################
def test_loop(dataloader, model, loss_fn):								#		    test loop
    size = len(dataloader.dataset)									#
    num_batches = len(dataloader)									#
    test_loss, correct = 0, 0										#
													#
    with torch.no_grad():										#
        for X, y in dataloader:									#
            pred = model(X)										#
            test_loss += loss_fn(pred, y).item()							#
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()				#
													#
    test_loss /= num_batches										#
    correct /= size											#
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")		#
													##########################################################
epochs = 10												#
for t in range(epochs):										# 			start training
    print(f"Epoch {t+1}\n-------------------------------")						#
    train_loop(train_dataloader, model, loss_fn, optimizer)						#
    test_loop(test_dataloader, model, loss_fn)							#
print("Done!")												#
					













