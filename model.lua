--
-- User: ksarma
--

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'optim'

-- mnist inputs are 28 x 28

function createModel()

    local model = nn.Sequential()

    -- input 1 x 28 x 28

    model:add(nn.SpatialConvolution(1, 10, 9, 9))

    -- output 10 x 19 x 19

    model:add(nn.Sigmoid())

    model:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- output 10 x 10 x 10

    model:add(nn.View(10 * 10 * 10))

    model:add(nn.Linear(10 * 10 * 10, 10))

    model:add(nn.LogSoftMax())

    -- We need to be either cuda or float since that what the input data is shaped as
    model:cuda()
--    model:float()

    -- important: classes are expected to labeled starting at 1, not zero as mnist is by default
    local criterion = nn.ClassNLLCriterion()

    -- We need to be either cuda or float since that what the input data is shaped as
    criterion:cuda()
--    criterion:float()

    return model, criterion

end