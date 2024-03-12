#pragma once
/*
runtime-sized ANN, runtime-sized matrix 
*/
#include "Tensor/Tensor.h"
#include "Common/String.h"	// std::ostream << std::vector<>
#include <vector>
#include <functional>
#include <cassert>
#include <cstring>

namespace NeuralNet {

inline int roundup4(int a) {
	return (a + 3) & (-4);
}

template<typename Real>
struct Vector {
	
	// size = layer size
	// so 'size+1' padding is always present and assigned to the bias
	int size = {};
	
	// storageSize = (size+1) roundup 4
	//		== v.size() when v is a std::vector
	int storageSize = {};
	
	std::vector<Real> v;
	
	Vector() {}
	
	Vector(int size_)
	:	size(size_),
		storageSize(roundup4(size+1)),
		v(storageSize)
	{}
	
	Real normL1() const {
		auto vi = v.data();
		auto vend = vi + size;
		Real sum = fabs(*vi);
		++vi;
		for (; vi < vend; ++vi) {
			sum += fabs(*vi);
		}
		return sum;
	}
	
	decltype(auto) operator[](int i) { return v[i]; }
	decltype(auto) operator[](int i) const { return v[i]; }
};

template<typename T>
std::ostream & operator<<(std::ostream & o, Vector<T> const & v) {
	return o << v.v;
}

template<typename Real>
struct ThinVec {
	Real * v = {};
	int size = {};
	int storageSize = {};
	ThinVec(Real * v_, int size_, int storageSize_)
	:	v(v_),
		size(size_),
		storageSize(storageSize_)
	{}
	decltype(auto) operator[](int i) { return v[i]; }
	decltype(auto) operator[](int i) const { return v[i]; }
};

template<typename Real>
struct Matrix {
	using ThinVec = NeuralNet::ThinVec<Real>;
	using ThinVecConst = NeuralNet::ThinVec<Real const>;
	
	Tensor::int2 size = {};
	
	Tensor::int2 storageSize = {};
	
	std::vector<Real> v;	// values stored row-major, size is 'storageSize'

	int height() const { return size.x; }
	int width() const { return size.y; }
	int storageWidth() const { return storageSize.y; }

	Matrix() {}
	
	Matrix(int h, int w)
	:	size(h, w),
		storageSize(h, roundup4(w)),
		v(storageSize.product())
	{}
	
	Real normL1() const {
		auto [height, width] = size;
		auto [storageHeight, storageWidth] = storageSize;
		assert(height == storageHeight);
		Real sum = {};
		int ij = 0;
		for (int i = 0; i < height; ++i) {
			int j = 0;
			for (; j < width; ++j, ++ij) {
				sum += fabs(v[ij]);
			}
			j += storageWidth - width;
			ij += storageWidth - width;
		}
		return sum;
	}

	ThinVec operator[](int const i) { return ThinVec(v.data() + storageWidth() * i, width(), storageWidth()); }
	ThinVecConst operator[](int const i) const { return ThinVecConst(v.data() + storageWidth() * i, width(), storageWidth()); }
};

template<typename T>
std::ostream & operator<<(std::ostream & o, Matrix<T> const & v) {
	return o << v.v;
}
double tanhDeriv(double x, double y) {
	return 1. - y * y;
}

template<typename Real>
struct Layer {
	using Vector = NeuralNet::Vector<Real>;
	using Matrix = NeuralNet::Matrix<Real>;
	Vector x, net;			// feed-forward
	Matrix w;				// weights
	Vector xErr, netErr;	// back-propagation
	Matrix dw;				// batch training accumulation
	std::function<Real(Real)> activation;				//y(x)
	std::function<Real(Real, Real)> activationDeriv;	//dy/dx(x,y)
private:
	bool useBias = true;
public:
	bool getBias() const { return useBias; }

	Layer(int sizeIn, int sizeOut)
	: 	x(sizeIn),
		net(sizeOut),
		w(sizeOut, sizeIn+1),
		xErr(sizeIn),
		netErr(sizeOut),
		dw(sizeOut, sizeIn+1),
		activation(tanh),
		activationDeriv(tanhDeriv)
	{
		// welp TODO gonna need a setter for that now 
		x.v[sizeIn] = useBias ? 1 : 0;
	}
};

//TODO something from stl 
#include <stdlib.h>	//rand()
template<typename Real = double>
Real random() { return (Real)rand() / (Real)RAND_MAX; }

template<typename Real = double>
struct ANN {
	using Vector = NeuralNet::Vector<Real>;
	using Matrix = NeuralNet::Matrix<Real>;
	using Layer = NeuralNet::Layer<Real>;
	
	std::vector<Layer> layers;
	// last-layer feed-forward components
	Vector output, outputError;
	// used for training:
	Vector desired;

	Real dt = 1;

	int useBatch = 0;	// set to a positive value to accumulate batch weight updates into the dw array
	int batchCounter = 0;
	int totalBatchCounter = 0;	//for entertainment

	//would be nice to just initialize a member-ref to layers[0].x
	// but to od that, i'd need to initialize layers[] in the ctor member list
	// and to do that I'd need t initialize layers[] alongside output, outputError, desired
	// and to do that I'd need something like ctor-member-initialization structure-binding
	// and that's not allowed yet afaik ...
	Vector & input() { return layers[0].x; }
	Vector & inputError() { return layers[0].xErr; }

	ANN(std::initializer_list<int> layerSizes) {
		auto layerSizeIter = layerSizes.begin();
		auto prevLayerSize = *layerSizeIter;
		for (++layerSizeIter; layerSizeIter != layerSizes.end(); ++layerSizeIter) {
			auto & layer = layers.emplace_back(prevLayerSize, *layerSizeIter);
			prevLayerSize = *layerSizeIter;
		
			// default weight initialization ...
			for (int i = 0; i < layer.w.height(); ++i) {
				for (int j = 0; j < layer.w.width(); ++j) {
					layer.w[i][j] = random<Real>() * 2. - 1.;
				}
			}

		}
		// final layer
		output = Vector(prevLayerSize);
		outputError = Vector(prevLayerSize);
		desired = Vector(prevLayerSize);
	
	}

	void feedForward() {
		int const numLayers = (int)layers.size();
		for (int k = 0; k < numLayers; ++k) {
			auto & layer = layers[k];
			
			const auto & w = layer.w;
			auto [height, width] = w.size;
			assert(width > 0);
			assert(height > 0);
			auto [storageHeight, storageWidth] = w.storageSize;
			assert(height == storageHeight);
			
			const auto & x = layer.x;
			assert(x.storageSize == storageWidth);
			
			assert(x.v[x.size] == layer.getBias() ? 1 : 0);

			auto & net = layer.net;
			assert(net.size == height);
			
			assert(width == x.size+1);
			assert(height == net.size);

			auto & y = k == numLayers-1 ? output : layers[k+1].x;
			assert(y.storageSize == net.storageSize);
			assert(y.storageSize == roundup4(storageHeight+1));
		
			auto const & activation = layer.activation;
			auto wij = w.v.data();
			auto xptr = x.v.data();
			auto xendptr = xptr + storageWidth;
			auto neti = net.v.data();	//net.v.begin() ? which is faster?
			auto netiend = neti + height;
			auto yi = y.v.data();
			
			for (; neti < netiend; 
				++neti, ++yi
			) {
				auto xj = xptr;
				*neti = 0;
				for (; xj < xendptr; 
					xj += 4, wij += 4
				) {
					*neti += wij[0] * xj[0]
						+ wij[1] * xj[1]
						+ wij[2] * xj[2]
						+ wij[3] * xj[3];
				}
				*yi = activation(*neti);
			}
			assert(neti == net.v.data() + height);
			assert(yi == y.v.data() + height);
			assert(wij == w.v.data() + storageWidth * height);
		}
	}

	Real calcError() {
		assert(desired.size == outputError.size);
		Real s = {};
		for (int i = 0; i < outputError.size; ++i) {
			auto delta = desired[i] - output[i];
			outputError[i] = delta;
			s += delta * delta;
		}
		return .5 * s;
	}

	void backPropagate() { backPropagate(dt); }
	void backPropagate(Real dt) {
		int const numLayers = (int)layers.size();
		for (int k = (int)numLayers-1; k >= 0; --k) {
			auto & layer = layers[k];
			auto & y = k == numLayers-1 ? output : layers[k+1].x;
			auto & yErr = k == numLayers-1 ? outputError : layers[k+1].xErr;
			auto const & activationDeriv = layer.activationDeriv;
			assert(layer.netErr.size == y.size);
			for (int i = 0; i < y.size; ++i) {
				layer.netErr[i] = yErr[i] * activationDeriv(layer.net[i], y[i]);
			}
			// back-propagate error
			for (int j = 0; j < layer.x.size; ++j) {
				Real s = {};
				for (int i = 0; i < layer.netErr.size; ++i) {
					s += layer.w[i][j] * layer.netErr[i];
				}
				layer.xErr[j] = s;
			}

			// adjust new weights
			assert(layer.x[layer.x.size] == layer.getBias() ? 1 : 0);
			if (!useBatch) {
				// ... directly/immediately
				for (int i = 0; i < layer.w.height(); ++i) {
					auto const l = layer.x.size;
					for (int j = 0; j <= l; ++j) {
						layer.w[i][j] += dt * layer.netErr[i] * layer.x[j];
					}
				}
			} else {
				// ... accumulate into dw
				for (int i = 0; i < layer.w.height(); ++i) {
					auto const l = layer.x.size;
					for (int j = 0; j <= l; ++j) {
						layer.dw[i][j] += dt * layer.netErr[i] * layer.x[j];
					}
				}
			}
		}
		if (useBatch) {
			++totalBatchCounter;
			++batchCounter;
			if (batchCounter >= useBatch) {
				updateBatch();
				batchCounter = 0;
			}
		}
	}

	// update weights by batch ... and then clear the batch
	void updateBatch() {
		if (!useBatch) return;
		for (int k = (int)layers.size()-1; k >= 0; --k) {
			auto & layer = layers[k];
			for (int i = 0; i < layer.w.height(); ++i) {
				auto const l = layer.x.size;
				for (int j = 0; j <= l; ++j) {
					layer.w[i][j] += layer.dw[i][j];
				}
			}
		}
		clearBatch();
	}

	void clearBatch() {
		if (!useBatch) return;
		for (int k = (int)layers.size()-1; k >= 0; --k) {
			auto & layer = layers[k];
			std::memset(layer.dw.v.data(), 0, sizeof(Real) * layer.dw.v.size());
		}
	}
};

}
