#pragma once
/*
runtime-sized ANN, runtime-sized matrix 
*/
#include "Tensor/Tensor.h"
#include "Common/String.h"	// std::ostream << std::vector<>
#include <vector>
#include <functional>
#include <cassert>

namespace NeuralNet {

template<typename Real>
struct Vector {
	int size = {};
	std::vector<Real> v;
	Vector() {}
	Vector(int size_) : size(size_), v(size) {}
	Real normL1() const {
		Real sum = {};
		for (auto vi : v) {
			sum += fabs(vi);
		}
		return sum;
	}
};

template<typename T>
std::ostream & operator<<(std::ostream & o, Vector<T> const & v) {
	return o << v.v;
}

template<typename Real>
struct Matrix {
	Tensor::int2 size = {};
	std::vector<Real> v;	// values stored row-major
	Matrix() {}
	Matrix(Tensor::int2 size_) : size(size_), v(size.product()) {}
	Matrix(int h, int w) : size(h, w), v(h * w) {}
	Real normL1() const {
		Real sum = {};
		for (auto vi : v) {
			sum += fabs(vi);
		}
		return sum;
	}
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
	bool useBias = true;
	std::function<Real(Real)> activation;				//y(x)
	std::function<Real(Real, Real)> activationDeriv;	//dy/dx(x,y)
	Layer(int sizeIn, int sizeOut)
	: 	x(sizeIn),
		net(sizeOut),
		w(sizeOut, sizeIn+1),
		xErr(sizeIn),
		netErr(sizeOut),
		dw(sizeOut, sizeIn+1),
		activation(tanh),
		activationDeriv(tanhDeriv)
	{}
};

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

	ANN(std::initializer_list<int> layerSizes) {
		auto layerSizeIter = layerSizes.begin();
		auto prevLayerSize = *layerSizeIter;
		for (++layerSizeIter; layerSizeIter != layerSizes.end(); ++layerSizeIter) {
			layers.emplace_back(prevLayerSize, *layerSizeIter);
			prevLayerSize = *layerSizeIter;
		}
		// final layer
		output = Vector(prevLayerSize);
		outputError = Vector(prevLayerSize);
		desired = Vector(prevLayerSize);
	}

	// TODO can this default into the initializer_list ctor?
	ANN(std::vector<int> layerSizes) {
		for (int i = 0; i < (int)layerSizes.size()-1; ++i) {
			layers.emplace_back(layerSizes[i], layerSizes[i+1]);
		}
		output = Vector(layerSizes.back());
		outputError = Vector(layerSizes.back());
		desired = Vector(layerSizes.back());
	}

	Vector & input() { return layers[0].x; }

	void feedForward() {
		int numLayers = (int)layers.size();
		for (int k = 0; k < numLayers; ++k) {
			auto & layer = layers[k];
			const auto & w = layer.w;
			auto [height, width] = w.size;
			const auto & x = layer.x;
			auto & net = layer.net;
			auto & y = k == numLayers-1 ? output : layers[k+1].x;
			const auto useBias = layer.useBias;
			assert(width > 0);
			assert(height > 0);
			assert(width == x.size+1);
			assert(height == net.size);
			
			auto & activation = layer.activation;
			auto wij = w.v.data();
			auto xptr = x.v.data();
			auto xendptr = xptr + width - 1;	// minus one to skip bias right-most col
			auto xendptrminus4 = xendptr - 4;
			auto neti = net.v.data();	//net.v.begin() ? which is faster?
			auto netiend = neti + height;
			auto yi = y.v.data();
			for (; neti < netiend; 
				++neti, ++yi
			) {
				if (width == 1) {
					*neti = useBias ? wij[0] : {};
					++wij;
				} else if (width == 2) {
					*neti = wij[0] * xptr[0]
						+ useBias ? wij[1] : {};
					wij += 2;
				} else { 
					auto xj = xptr;
					*neti = *wij * *xj;
					++wij;
					++xj;

#if 1	// runs 3x faster with GCC
					for (; xj <= xendptrminus4; 
						xj += 4, wij += 4
					) {
						*neti += wij[0] * xj[0]
							+ wij[1] * xj[1]
							+ wij[2] * xj[2]
							+ wij[3] * xj[3];
					}
#endif				
					for (; xj < xendptr; 
						++xj, ++wij
					) {
						*neti += wij[0] * xj[0];
					}

					assert(xj == xendptr);
					if (useBias) {
						*neti += *wij;
					}
				}
				++wij;
				*yi = activation(*neti);
			}
			assert(neti == net.v.data() + height);
			assert(yi == y.v.data() + height);
			assert(wij == w.v.data() + width * height);
		}
	}

	void backPropagate(Real dt) {
	}
	void backPropagate() { backPropagate(dt); }
};

}
