This is Lua bindings for the NeuralNet library.
It's designed to have an API that matches the `neuralnet` lua project.
That means 1-based indexes.  yes.

- `ANN = require 'NeuralNetLua'` = get the ctor.
- `ANN` = function (table? class?) that constructs an ANN object.
- `ann.dt`
- `ann.x[]`
- `ann.xErr[]`
- `ann.w[]`
- `ann.net[]`
- `ann.netErr[]`
- `ann.useBias[]`
- `ann.dw[]`
- `ann.useBatch`
- `ann.batchCounter`
- `ann.totalBatchCounter`
- `ann:feedForward()`
- `ann:calcError()`
- `ann:backPropagate([dt])`
- `ann:updateBatch()`
- `ann:clearBatch()`
