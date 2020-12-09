import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model import LSTM

# split training set& test set
def split_data(df, numDay):
	# create seq of data
	raw = df.to_numpy()
	data = []
	for i in range(len(raw)-numDay):
		data.append(raw[i : i+numDay])

	data = np.array(data)
	train_size = int(np.round(0.8*data.shape[0]))
	test_size = data.shape[0] - train_size

	# last data serve as label
	x_train = data[:train_size, :-1, :]
	y_train = data[:train_size, -1:, :]

	x_test = data[train_size:, :-1, :]
	y_test = data[train_size:, -1:, :]

	return x_train,y_train,x_test,y_test

def train(company):
	df = pd.read_csv('./stock_details/'+company+'.csv')
	df = df.sort_values('Date')
	df.reset_index()
	price = df[['Close']]
	# print(price)
	# normalize data to better prediction
	scaler = MinMaxScaler(feature_range=(-1,1))
	price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

	# number of rolling day
	numDay = 20
	x_train, y_train, x_test, y_test = split_data(price,numDay)

	x_train = torch.from_numpy(x_train).type(torch.Tensor)
	x_test = torch.from_numpy(x_test).type(torch.Tensor)
	y_train = torch.from_numpy(y_train).type(torch.Tensor)
	y_test = torch.from_numpy(y_test).type(torch.Tensor)

	input_dim = 1
	hidden_dim = 32
	num_layers = 2
	output_dim = 1
	num_epochs = 50

	model = LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, output_dim=output_dim)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters())

	hist = np.zeros(num_epochs)
	for epoch in range(num_epochs):
		y_train_pred = model(x_train)
		loss = criterion(y_train_pred, y_train)
		print("Epoch ", epoch, "MSE: ", loss.item())
		hist[epoch] = loss.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
	# set dropout and batch normalization layers
	model.eval()
	# torch.save(model.state_dict(),'./saved_model/'+company+'.pth')
	y_test_pred = model(x_test)
	# print(y_test.shape)
	y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
	y_test = scaler.inverse_transform(y_test[:,-1,:].detach().numpy())
	# print(y_test_pred.shape)
	# print(y_test_pred)
	# print(y_test_pred[-1,:])
	return y_test_pred[-1,:]
	
if __name__ == "__main__":
	train('AMZN')