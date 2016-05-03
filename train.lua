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

-- The model expects input to be of the size nSamples x nChannels x dim x dim
-- but since we have only one channel the input tensor is missing the 2nd dimension.
-- We can add it here using resize
traindata = traindata:resize(60000, 1, 28, 28)

-- Here we either convert to cuda or to float; we need one or the other because
-- most operations are only valid on doubles or floats (cuda is a type of float)
-- traindata:cuda()
-- trainlabels:cuda()
traindata = traindata:float()
trainlabels = trainlabels:float()


local model, criterion = createModel()

print(model)

NUM_SAMPLES = 60000
BATCH_SIZE = 1000

function train()

    model:training()

    local epoch = 1

    local time = sys.clock()

    print("Training epoch " .. epoch)

    while (epoch <= 3) do

        -- Shuffle
        local r = torch.randperm(NUM_SAMPLES):long()
        local rand_data = traindata:index(1, r)
        local rand_labels = trainlabels:index(1, r)

        for i=1,NUM_SAMPLES,BATCH_SIZE do
            print("Training minibatch starting at " .. i)
	    
	    local params, grads = model:getParameters()

            local err, outputs

            local batch_data = rand_data[{{i,i+BATCH_SIZE-1},{}}]
            local batch_labels = rand_labels[{{i,i+BATCH_SIZE-1}}]

            local feval = function(x)

                model:zeroGradParameters()
                outputs = model:forward(batch_data)
                err = criterion:forward(outputs, batch_labels)
                local gradOutputs = criterion:backward(outputs, batch_labels)
                model:backward(batch_data, gradOutputs)
                return err, grads

            end

            optim.adadelta(feval, params, {})
        end

        local top1 = 0
        do
           local _,prediction_sorted = outputs:float():sort(2, true) -- descending
           for i=1,100 do
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