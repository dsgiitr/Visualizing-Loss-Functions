require 'torch';
require 'nn';
require 'gnuplot':
require 'optim':

--loading Data
DataPath = '/home/apoorva/Documents/AutoEncoder/mnist/'

training_data= torch.Diskfile(DataPath .. "train-images.idx3-ubyte")
test_data = torch.Diskfile(DataPath .. "t10k-images.idx3-ubyte")

TR_data = torch.ByteTensor(60000,784)
training_data:readByte(16)
training_data:readByte(Tr_data:storage())
training_data = Tr_data:double():div(255)

Ts_data = torch.ByteTensor(60000,784)
test_data:readByte(16)
test_data:readByte(Ts_data:storage())
test_data = Ts_data:double():div(255)

training_label = torch.Diskfile(DataPath .. "train-labels.idx1-ubyte")
test_label = torch.Diskfile(DataPath .. "t10k-labels.idx1-ubyte")


train_label_set = torch.ByteTensor(60000)
training_label:readByte(8)
training_label:readByte(train_label_set:storage():double())
training_label = torch.Tensor(60000,10)

for n=1,60000 do
	x= torch.zeros(10)
	x[train_label_set[i] + 1]= 1
	training_label[i]= x


test_label_set = torch.ByteTensor(60000)
test_label:readByte(8)
test_label:readByte(test_label_set:storage():double())
test_label = torch.Tensor(60000,10)

for n=1,60000 do
	x= torch.zeros(10)
	x[test_label_set[i] + 1]= 1
	test_label[i]= x

--Data loaded 

--building an autoencoder
net = nn:Sequential()

net:add(nn.Linear(784,128))
net:add(nn.ReLu())
net:add(nn.Linear(128,32))
net:add(nn.ReLu())

net:add(nn.Linear(32,128))
net:add(nn.ReLu())
net:add(nn.Linear(128,784))
net:add(nn.ReLu())

loss= nn.MSECriterion()


--training an autoencoder
for epochs=1,20 do

	print("no. of epochs	".. epochs)
	Weight, Gradients = net:getParamters()
	total_loss= 0

	for n= 1,input_nodes do
		input= train_data
			output= net:forward(input)
		current_loss = loss:forward(output, actual_data)
		y = loss:backward(output, actual_data)
		net:backward(input,y)

		total_loss= total_loss + cur_loss
	end	

	print("loss for this epoch" ..  total_loss)

end
