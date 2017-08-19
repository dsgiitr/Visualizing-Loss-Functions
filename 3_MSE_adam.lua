require 'torch';
require 'optim';
require 'nn';
require 'image';

local optimState = {    learningRate = 1e-4}
--[[
torch.manualSeed(0) 
--epoch = 15






---------------------------------
---------------------------------



trainset = torch.DiskFile('zip/train-images.idx3-ubyte')
testset = torch.DiskFile('zip/t10k-images.idx3-ubyte')

Tr_Set = torch.ByteTensor(60000, 784)
Test_Set = torch.ByteTensor(10000, 784)

trainset:readByte(16)
trainset:readByte(Tr_Set:storage())
Tr_Set = Tr_Set:float():div(255) 
--Tr_Set = Tr_Set[{{1,100},{}}]:float():div(255) 
testset:readByte(16)
testset:readByte(Test_Set:storage())
Test_Set = Test_Set:float():div(255) 


print('Dataset Loaded and Ready..')
]]--
---------------------------------
---------------------------------


function network(Data,Net,Out)
  Data = Data
  Out = Out
  function feval(Weights)
    Gradients:zero()
    y = Net:forward(Data)
    currLoss =  loss:forward(y,Out)
    local dE_dy = loss:backward(y, Out)
    Net:backward(Data, dE_dy)
    return currLoss, Gradients
 end
 optim.adagrad(feval, Weights, optimState)
 return currLoss
end




function Train_Net(Net,Data_Set,Data_out)

  for Tepoch = 1,epoch do
    print('epoch ' .. Tepoch .. '/' .. epoch)
    Total_Loss=0
    for loopno=1,60000 do
      input = Data_Set[loopno]
      output = Data_out[loopno]
      local LossTrain = network(input,Net,output)    
      Total_Loss = Total_Loss + LossTrain
    end
    Total_Loss = Total_Loss/Data_out:size(1)
    print('Training Loss = ' .. Total_Loss)
    Train_Error[{{Tepoch},{3}}] = Total_Loss
  end
end




------------------------------------------------------------------
------------------------------------------------------------------
--      Training       -------------------------------------------
------------------------------------------------------------------
------------------------------------------------------------------

print('Training...')

local net

net = nn:Sequential()

net:add(nn.Linear(784,128))
net:add(nn.ReLU())
net:add(nn.Linear(128,32))
net:add(nn.ReLU())

net:add(nn.Linear(32,128))
net:add(nn.ReLU())
net:add(nn.Linear(128,784))
net:add(nn.ReLU())
loss = nn.MSECriterion()


-- STEP-1
print('At Ladder-1 MSE loss + adam  ................................')
Weights,Gradients = net:getParameters()
Train_Net(net,Tr_Set,Tr_Set)


--[[
------------------------------------------------------------------
------------------------------------------------------------------
--      Validation    --------------------------------------------
------------------------------------------------------------------
------------------------------------------------------------------

net:evaluate()
Total_Loss = 0
for loopno=1,10000 do
  input = Test_Set[loopno]
  input = input
  currLoss = nn.MSECriterion():forward(net:forward(input),input)
  Total_Loss = Total_Loss + currLoss
end

ValLoss = Total_Loss/10000
print('Total Testing/Validation Loss : '.. ValLoss)

input = Test_Set[1]
inputimg = torch.reshape(input,28,28)
input = input
out = net:forward(input)
outputimg = torch.reshape(out,28,28)

image.save('reconsImage_smooth_mse.png', outputimg)
--image.save('ActualImage.png', inputimg)

print(net)
]]--
