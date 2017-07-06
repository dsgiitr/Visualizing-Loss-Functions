require 'torch';
require 'nn';
require 'gnuplot';
require 'optim';
require 'image'

--loading Data
DataPath = '/home/apoorva/Documents/AutoEncoders/mnist/'

training_data= torch.DiskFile(DataPath .. "train-images.idx3-ubyte")
test_data = torch.DiskFile(DataPath .. "t10k-images.idx3-ubyte")

Tr_data = torch.ByteTensor(60000,784)
training_data:readByte(16)
training_data:readByte(Tr_data:storage())
training_data = Tr_data:double():div(255)

Ts_data = torch.ByteTensor(10000,784)
test_data:readByte(16)
test_data:readByte(Ts_data:storage())
test_data = Ts_data:double():div(255)

training_label = torch.DiskFile(DataPath .. "train-labels.idx1-ubyte")
test_label = torch.DiskFile(DataPath .. "t10k-labels.idx1-ubyte")


train_label_set = torch.ByteTensor(60000)
training_label:readByte(8)
training_label:readByte(train_label_set:storage())
training_label_temp = train_label_set:double()
training_label = torch.Tensor(60000,10)

for j=1,60000 do
	x= torch.zeros(10)
	x[training_label_temp[j] + 1]= 1
	training_label[j]= x
end

test_label_set = torch.ByteTensor(10000)
test_label:readByte(8)
test_label:readByte(test_label_set:storage())
test_label_temp = test_label_set:double()
test_label = torch.Tensor(10000,10)

for j=1,10000 do
	x= torch.zeros(10)
	x[test_label_temp[j] + 1]= 1
	test_label[j]= x
end

--Data loaded 

--building an autoencoder
net = nn:Sequential()

net:add(nn.Linear(784,128))
net:add(nn.ReLU())
net:add(nn.Linear(128,32))
net:add(nn.ReLU())

net:add(nn.Linear(32,128))
net:add(nn.ReLU())
net:add(nn.Linear(128,784))
net:add(nn.ReLU())

loss= nn.MSECriterion()


--training an autoencoder
for epochs=1,30 do

	print("no. of epochs	".. epochs)
	Weight, Gradients = net:getParameters()
	total_loss= 0

	for n= 1,60000 do

		function feval(Weight)

			Gradients:zero()

			input= training_data[n]
			output= net:forward(input)
			current_loss = loss:forward(output, training_data[n])
			local d_loss = loss:backward(output, training_data[n])
			net:backward(input,d_loss)

			return current_loss, Gradients
		end

		optimState = {
    		learningRate = 0.01,
    	}
		
		optim.sgd(feval, Weight, optim_state)			

		total_loss= total_loss + current_loss

	end	

	total_loss = total_loss/60000

	input_image = torch.reshape(input,28,28)
	output_image = torch.reshape(output,28,28)

	image.save('Results_sgd/'.. epochs ..'_sdg.png', output_image)
	print("loss for this epochs		" ..  total_loss)

end
image.save('actual.png', input_image)
