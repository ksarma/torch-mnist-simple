--
-- User: ksarma
--


require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'optim'
local mnist = require 'mnist'

require 'model.lua'

local trainset = mnist.traindataset()
local testset = mnist.testdataset()

local traindata = trainset.data
local trainlabels = trainset.label + 1

-- traindata:cuda()
-- trainlabels:cuda()


local model, criterion = createModel()

print(model)


function train()

    model:training()

    local epoch = 1

    local time = sys.clock()

    print("Training epoch " .. epoch)

    while (epoch <= 3) do

        local params, grads = model:getParameters()


        local err, outputs

        local feval = function(x)

            model:zeroGradParameters()
            outputs = model:forward(traindata)
            err = criterion:forward(outputs, trainlabels)
            local gradOutputs = criterion:backward(outputs, trainlabels)
            model:backward(traindata, gradOutputs)
            return err, grads

        end

        optim.adadelta(feval, params, {})


        local top1 = 0
        do
           local _,prediction_sorted = outputs:float():sort(2, true) -- descending
           for i=1,60000 do
               if prediction_sorted[i][1] == trainlabels[i] then
                   top1 = top1 + 1
               end
           end
           top1 = top1 * 100 / 60000;
        end

        print(("Epoch: " .. epoch .. " Top1-%%: %.2f"):format(top1))

        epoch = epoch + 1
    end



end



train()