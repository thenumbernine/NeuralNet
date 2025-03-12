This is Lua bindings for the NeuralNet library.
It's designed to have an API that matches the `neuralnet` lua project.
That means 1-based indexes.  yes.

- `ANN = require 'NeuralNetLua' 'NeuralNet::ANN<float>'` = get the float ctor.
- `ANN = require 'NeuralNetLua' 'NeuralNet::ANN<double>'` = get the double ctor.
- `ann = ANN(layer1, ..., layerN)` = function (table? class?) that constructs an ANN object.
- `ann.dt`
- `ann.layers[]`
- `ann.output`
- `ann.desired`
- `ann.outputError`
- `layer.x[]`
- `layer.xErr[]`
- `layer.w[]`
- `layer.net[]`
- `layer.netErr[]`
- `layer.useBias`
- `layer.dw[]`
- `ann.useBatch`
- `ann.batchCounter`
- `ann.totalBatchCounter`
- `ann:feedForward()`
- `ann:calcError()`
- `ann:backPropagate([dt])`
- `ann:updateBatch()`
- `ann:clearBatch()`

Driven by some Lua C++ automatic binding / member object and method wrapper generation that is pretty concise (500 loc or so).
