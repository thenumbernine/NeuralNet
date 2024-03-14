### Neural Net

[![Donate via Stripe](https://img.shields.io/badge/Donate-Stripe-green.svg)](https://buy.stripe.com/00gbJZ0OdcNs9zi288)<br>
[![Donate via Bitcoin](https://img.shields.io/badge/Donate-Bitcoin-green.svg)](bitcoin:37fsp7qQKU8XoHZGRQvVzQVP8FrEJ73cSJ)<br>

This is just a simple back-propagataion neural network class I made.

Bias value is baked into the vector memory (debug asserts make sure it's not overwritten).

Vectors are 4-float aligned for future/smart-compiler SIMD optimizations.

Everything is in templates.

There's also some auto C++ Lua binding code that is in the `luabinding` folder.
