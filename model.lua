--
-- User: ksarma
--

require 'torch'
require 'nn'
require 'cunn'

-- mnist inputs are 28 x 28

function createModel()

    local model = nn.Sequential()

    -- input 1 x 28 x 28

    model:add(nn.SpatialConvolution(1, 10, 9, 9))

    -- output 10 x 19 x 19

    model:add(nn.Sigmoid())

    model:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    -- output 10 x 9 x 9

    model:add(nn.View(10 * 9 * 9))

    model:add(nn.Linear(10 * 9 * 9, 10))

    model:add(nn.LogSoftMax())

    model:cuda()

    -- important: classes are expected to labeled starting at 1, not zero as mnist is by default
    local criterion = nn.ClassNLLCriterion()
    criterion:cuda()

    return model, criterion

end