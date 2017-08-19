require 'torch';
require 'optim';
require 'nn';
require 'image';

local optimState = {    learningRate = 1e-4}


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

