This is Lua bindings for the NeuralNet library.
It's designed to have an API that matches the `neuralnet` lua project.
That means 1-based indexes.  yes.

- `ANN = require 'NeuralNetLua'` = get the ctor.
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
TODO's for that:
- needs member-method-returns-ref to work
- needs pass-by-value methods to push copies of full userdata instead of light userdata
- ipairs for IndexAccess classes, pairs for everyone
- expose C++ static members in the obj metatables (esp so the Lua metatable instances can access them, but so can the outside world via the metatable)
