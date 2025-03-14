#pragma once
/*
runtime-sized ANN, runtime-sized matrix
*/
#include "Tensor/Tensor.h"
#include "Common/String.h"	// std::ostream << std::vector<>
#include "Common/Exception.h"
#include <vector>
#include <functional>
#include <cassert>
#include <cstring>
#include <cmath>

namespace NeuralNet {

using DefaultReal = double;

constexpr bool ispowerof2(int N) {
	return N > 0 && 
		//N & (N - 1) == 0;	//is this a power-of-2 test?
		(N | (N - 1)) == 2 * N - 1;	// better power-of-2 test?
}

// works for powers of 2
template<int N>
requires (ispowerof2(N))
constexpr inline int roundup(int a) {
	return (a + N) & (-N);
}

template<typename Real>
struct Vector {

	// size = layer size
	// so 'size+1' padding is always present and assigned to the bias
	// TODO but it's not needed in the output layer... so why not provide storageSize in ctor?
	int size = {};

	// storageSize = (size+1) roundup 4
	//		== v.size() when v is a std::vector
	int storageSize = {};

	std::vector<Real> v;

	Vector() {}

	Vector(int size_)
	:	size(size_),
		storageSize(roundup<8>(size+1)),
		v(storageSize)
	{}

	Real normL1() const {
		auto vi = v.data();
		auto vend = vi + size;
		Real sum = std::fabs(*vi);
		++vi;
		for (; vi < vend; ++vi) {
			sum += std::fabs(*vi);
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
struct ThinVector {
	Real * v = {};
	int size = {};
	int storageSize = {};

	ThinVector(Real * v_, int size_, int storageSize_)
	:	v(v_),
		size(size_),
		storageSize(storageSize_)
	{}

	Real normL1() const {
		auto vi = v;
		auto vend = vi + size;
		Real sum = std::fabs(*vi);
		++vi;
		for (; vi < vend; ++vi) {
			sum += std::fabs(*vi);
		}
		return sum;
	}

	decltype(auto) operator[](int i) { return v[i]; }
	decltype(auto) operator[](int i) const { return v[i]; }
};

template<typename Real>
struct Matrix {
	using ThinVector = NeuralNet::ThinVector<Real>;
	using ThinVecConst = NeuralNet::ThinVector<Real const>;

	Tensor::int2 size = {};

	Tensor::int2 storageSize = {};

	std::vector<Real> v;	// values stored row-major, size is 'storageSize'

	int height() const { return size.x; }
	int width() const { return size.y; }
	int storageHeight() const { return storageSize.x; }
	int storageWidth() const { return storageSize.y; }

	Matrix() {}

	Matrix(int h, int w)
	:	size(h, w),
		storageSize(roundup<8>(h), roundup<8>(w)),
		v(storageSize.product())
	{}

	Real normL1() const {
		auto [height, width] = size;
		auto [storageHeight, storageWidth] = storageSize;
		assert(roundup<8>(height) == storageHeight);
		assert(roundup<8>(width) == storageWidth);
		Real sum = {};
		int ij = 0;
		for (int i = 0; i < height; ++i) {
			int j = 0;
			for (; j < width; ++j, ++ij) {
				sum += std::fabs(v[ij]);
			}
			j += storageWidth - width;
			ij += storageWidth - width;
		}
		return sum;
	}

	ThinVector operator[](int const i) { return ThinVector(v.data() + storageWidth() * i, width(), storageWidth()); }
	ThinVecConst operator[](int const i) const { return ThinVecConst(v.data() + storageWidth() * i, width(), storageWidth()); }
};

template<typename T>
std::ostream & operator<<(std::ostream & o, Matrix<T> const & v) {
	return o << v.v;
}

template<typename Real>
Real tanhDeriv(Real x, Real y) {
	return 1 - y * y;
}

//canned functions
template<typename Real = DefaultReal>
struct Activation {
	std::string name;
	std::function<Real(Real)> f;		//y(x)

	static std::vector<Activation> const & all() {
		static std::vector<Activation> list = {
			{"identity", [](Real x) -> Real { return x; }},
			{"tanh", static_cast<Real(*)(Real)>(std::tanh)},
			{"sigmoid", [](Real x) -> Real { return Real(1) / (Real(1) + std::exp(-x)); }},
			{"poorLinearTanh", [](Real x) -> Real {		// aka 'hard tanh' aka 'clamp ±1'
				return std::clamp<Real>(x, Real(-1), Real(1));
			}},
			{"poorQuadraticTanh", [](Real x) -> Real {
				return x < Real(-2) ? Real(-1)
				: (x < Real(0)) ? x * (Real(1) + Real(.25)*x)
				: (x < Real(2)) ? x * (Real(1) - Real(.25)*x)
				: Real(1);
			}},
			{"poorCubicTanh", [](Real x) -> Real {
				return x < Real(-2.5) ? Real(-1)
				: x < Real(0) ? x * (Real(1) + x * (Real(0.32) + x * Real(0.032)))
				: x < Real(2.5) ? x * (Real(1) + x * (Real(-0.32) + x * Real(0.032)))
				: Real(1);
			}},
			{"ReLU", [](Real x) -> Real { 		// aka 'ramp' aka 'max(x,0)'
				return std::max<Real>(x, Real(0));
			}},
		};
		return list;
	}

	// TODO map for look up or nah?
	static Activation get(std::string const & name) {
		for (auto const & f : all()) {
			if (f.name == name) return f;
		}
		throw Common::Exception() << "I couldn't find what you were looking for";
	}
};

template<typename Real = DefaultReal>
struct ActivationDeriv {
	std::string name;
	std::function<Real(Real, Real)> f;		//dy/dx(x,y)

	static std::vector<ActivationDeriv> const & all() {
		static std::vector<ActivationDeriv> list = {
			{"one", [](Real x, Real y) -> Real { return Real(1); }},
			{"tanhDeriv", tanhDeriv<Real>},
			{"sigmoidDeriv", [](Real x, Real y) -> Real { return y * (Real(1) - y); }},
			{"poorLinearTanhDeriv", [](Real x, Real y) -> Real {	// aka boxcar function
				return (x >= Real(-1) && x <= Real(1)) ? Real(1) : Real(0);
			}},
			{"poorQuadraticTanhDeriv", [](Real x, Real y) -> Real {	// aka triangular function
				return (x < Real(-2)) ? Real(0)
				: x < Real(0) ? Real(1) + Real(.5) * x
				: x < Real(2) ? Real(1) - Real(.5) * x
				: Real(0);
			}},
			{"poorCubicTanhDeriv", [](Real x, Real y) -> Real {
				return x < Real(-2.5) ? Real(0)
				: x < Real(0) ? Real(1) + x * (Real(.64) + x * Real(.096))
				: x < Real(2.5) ? Real(1) + x * (Real(-.64) + x * Real(.096))
				: Real(0);
			}},
			{"ReLUDeriv", [](Real x, Real y) -> Real { 	// aka Heaviside
				return x < Real(0) ? Real(0) : Real(1);
			}},
		};
		return list;
	}

	// TODO map for look up or nah?
	static ActivationDeriv get(std::string const & name) {
		for (auto const & f : all()) {
			if (f.name == name) return f;
		}
		throw Common::Exception() << "I couldn't find what you were looking for";
	}
};

template<typename Real>
struct Layer {
	using Vector = NeuralNet::Vector<Real>;
	using Matrix = NeuralNet::Matrix<Real>;
	using Activation = NeuralNet::Activation<Real>;
	using ActivationDeriv = NeuralNet::ActivationDeriv<Real>;

	Vector x, net;			// feed-forward
	Matrix w;				// weights
	Vector xErr, netErr;	// back-propagation
	Matrix dw;				// batch training accumulation

	Activation activation;
	ActivationDeriv activationDeriv;
	// until I get function read/write working (efficiently) ...
	void setActivation(std::string const & name) {
		activation = Activation::get(name);
	}
	void setActivationDeriv(std::string const & name) {
		activationDeriv = ActivationDeriv::get(name);
	}

private:
	bool useBias = true;
public:
	bool getBias() const { return useBias; }
	void setBias(bool bias) {
		useBias = bias;
		for (int i = 0; i < w.height(); ++i) {
			w[i][w.width()-1] = Real(bias ? 1 : 0);
		}
	}

	Layer(int sizeIn, int sizeOut)
	: 	x(sizeIn),
		net(sizeOut),
		w(sizeOut, sizeIn+1),
		xErr(sizeIn),
		netErr(sizeOut),
		dw(sizeOut, sizeIn+1),
		activation(Activation::get("tanh")),
		activationDeriv(ActivationDeriv::get("tanhDeriv"))
	{
		// welp TODO gonna need a setter for that now
		x.v[sizeIn] = useBias ? 1 : 0;
	}
};

//TODO something from stl
#include <stdlib.h>	//rand()
template<typename Real = DefaultReal>
Real random() { return (Real)rand() / (Real)RAND_MAX; }



// different kinds of multipliers to the weights upon update
// One = normal
// Dropout = filered-per-col (but that means traversing cols first then rows, opposite of memory order)
// Dilution = randomly applies, so orig order is fine

template<typename Real>
struct One {
	static constexpr Real f() { return Real(1); }
};

// same as Dilution, only a dif class for template specializing
template<typename Real>
struct Dropout {
	Real dropout;
	Dropout(Real dropout_) : dropout(dropout_) {}
	Real f() const {
		return random<Real>() < dropout ? Real(1) : Real(0);
	}
};

template<typename Real>
struct Dilution {
	Real dilution;
	Dilution(Real dilution_) : dilution(dilution_) {}
	Real f() const {
		return random<Real>() < dilution ? Real(1) : Real(0);
	}
};

// by default do row major i.e. inner product mul
template<typename Real, typename Mul>
struct BackProp {
	constexpr static void go(
		Mul mul,
		auto const height,
		auto const width,
		auto const storageHeight,
		auto const storageWidth,
		auto destwptr,
		auto xptr,
		auto neterrptr,
		auto dt
	) {
		auto destwij = destwptr;
		auto xendptr = xptr + storageWidth;
		auto neterri = neterrptr;
		auto neterriend = neterri + height;
		for (; neterri < neterriend; ++neterri) {
			auto const neterridt = dt * neterri[0];
			for (auto xj = xptr; xj < xendptr;
				xj += 8,
				destwij += 8
			) {
				destwij[0] += neterridt * xj[0] * mul.f();
				destwij[1] += neterridt * xj[1] * mul.f();
				destwij[2] += neterridt * xj[2] * mul.f();
				destwij[3] += neterridt * xj[3] * mul.f();
				destwij[4] += neterridt * xj[4] * mul.f();
				destwij[5] += neterridt * xj[5] * mul.f();
				destwij[6] += neterridt * xj[6] * mul.f();
				destwij[7] += neterridt * xj[7] * mul.f();
			}
		}
	}
};

// for dropout, skip every randomly masked col, so we gotta reverse our mul loop order
template<typename Real>
struct BackProp<Real, Dropout<Real>> {
	constexpr static void go(
		Dropout<Real> mul,
		auto const height,
		auto const width,
		auto const storageHeight,
		auto const storageWidth,
		auto destwptr,
		auto xptr,
		auto neterrptr,
		auto dt
	) {
		for (int j = 0; j < width; ++j) {
			if (mul.f()) {
				Real const dt_xj = dt * xptr[j];
				for (int i = 0; i < storageHeight; i += 8) {
					destwptr[j + storageWidth * (i + 0)] += dt_xj * neterrptr[i + 0];
					destwptr[j + storageWidth * (i + 1)] += dt_xj * neterrptr[i + 1];
					destwptr[j + storageWidth * (i + 2)] += dt_xj * neterrptr[i + 2];
					destwptr[j + storageWidth * (i + 3)] += dt_xj * neterrptr[i + 3];
					destwptr[j + storageWidth * (i + 4)] += dt_xj * neterrptr[i + 4];
					destwptr[j + storageWidth * (i + 5)] += dt_xj * neterrptr[i + 5];
					destwptr[j + storageWidth * (i + 6)] += dt_xj * neterrptr[i + 6];
					destwptr[j + storageWidth * (i + 7)] += dt_xj * neterrptr[i + 7];
				}
			}
		}
	}
};

template<typename Real, typename Mul>
struct UpdateBatch {
	constexpr static void go(
		Mul mul,
		auto const height,
		auto const width,
		auto const storageHeight,
		auto const storageWidth,
		auto wptr,
		auto dwptr
	) {
		auto wijptr = wptr;
		auto dwijptr = dwptr;
		auto wijendptr = wijptr + storageWidth * height;
		for (; wijptr < wijendptr;
			wijptr += 8,
			dwijptr += 8
		) {
			wijptr[0] += dwijptr[0] * mul.f();
			wijptr[1] += dwijptr[1] * mul.f();
			wijptr[2] += dwijptr[2] * mul.f();
			wijptr[3] += dwijptr[3] * mul.f();
			wijptr[4] += dwijptr[4] * mul.f();
			wijptr[5] += dwijptr[5] * mul.f();
			wijptr[6] += dwijptr[6] * mul.f();
			wijptr[7] += dwijptr[7] * mul.f();
		}
	}
};

template<typename Real>
struct UpdateBatch<Real, Dropout<Real>> {
	constexpr static void go(
		Dropout<Real> mul,
		auto const height,
		auto const width,
		auto const storageHeight,
		auto const storageWidth,
		auto wptr,
		auto dwptr
	) {
		for (int j = 0; j < width; ++j) {
			if (mul.f()) {
				for (int i = 0; i < storageHeight; i += 8) {
					wptr[j + storageWidth * (i + 0)] += dwptr[i + storageWidth * (i + 0)];
					wptr[j + storageWidth * (i + 1)] += dwptr[i + storageWidth * (i + 1)];
					wptr[j + storageWidth * (i + 2)] += dwptr[i + storageWidth * (i + 2)];
					wptr[j + storageWidth * (i + 3)] += dwptr[i + storageWidth * (i + 3)];
					wptr[j + storageWidth * (i + 4)] += dwptr[i + storageWidth * (i + 4)];
					wptr[j + storageWidth * (i + 5)] += dwptr[i + storageWidth * (i + 5)];
					wptr[j + storageWidth * (i + 6)] += dwptr[i + storageWidth * (i + 6)];
					wptr[j + storageWidth * (i + 7)] += dwptr[i + storageWidth * (i + 7)];
				}
			}
		}
	}
};

template<typename Real = DefaultReal>
struct ANN {
	using Vector = NeuralNet::Vector<Real>;
	using Matrix = NeuralNet::Matrix<Real>;
	using Layer = NeuralNet::Layer<Real>;
	using Activation = NeuralNet::Activation<Real>;
	using ActivationDeriv = NeuralNet::ActivationDeriv<Real>;

	std::vector<Layer> layers;
	// last-layer feed-forward components
	Vector output, outputError;
	// used for training:
	Vector desired;

	//timestep of the gradient to forward-Euler integrate along
	Real dt = 1;

	int useBatch = 0;	// set to a positive value to accumulate batch weight updates into the dw array
	int batchCounter = 0;

	// what %age of the weight columns to update per-back-propagation / batch-update
	Real dropout = 1;

	// what %age of the weights to update per-back-propagation / batch-update
	Real dilution = 1;

	//would be nice to just initialize a member-ref to layers[0].x
	// but to od that, i'd need to initialize layers[] in the ctor member list
	// and to do that I'd need t initialize layers[] alongside output, outputError, desired
	// and to do that I'd need something like ctor-member-initialization structure-binding
	// and that's not allowed yet afaik ...
	Vector & input() { return layers[0].x; }
	Vector & inputError() { return layers[0].xErr; }

	ANN(std::initializer_list<int> layerSizes) {
		auto layerSizeIter = layerSizes.begin();
		if (layerSizeIter == layerSizes.end()) throw Common::Exception() << "cannot construct a network with no layers";
		auto prevLayerSize = *layerSizeIter;
		for (++layerSizeIter; layerSizeIter != layerSizes.end(); ++layerSizeIter) {
			auto & layer = layers.emplace_back(prevLayerSize, *layerSizeIter);
			prevLayerSize = *layerSizeIter;

			// default weight initialization ...
			for (int i = 0; i < layer.w.height(); ++i) {
				for (int j = 0; j < layer.w.width(); ++j) {
					layer.w[i][j] = random<Real>() * 2 - 1;
				}
			}

		}
		// final layer
		output = Vector(prevLayerSize);
		outputError = Vector(prevLayerSize);
		desired = Vector(prevLayerSize);
	}

	//identical to intializer_list ctor ... maybe make a templated consolidation
	ANN(std::vector<int> const & layerSizes) {
		auto layerSizeIter = layerSizes.begin();
		if (layerSizeIter == layerSizes.end()) throw Common::Exception() << "cannot construct a network with no layers";
		auto prevLayerSize = *layerSizeIter;
		for (++layerSizeIter; layerSizeIter != layerSizes.end(); ++layerSizeIter) {
			auto & layer = layers.emplace_back(prevLayerSize, *layerSizeIter);
			prevLayerSize = *layerSizeIter;

			// default weight initialization ...
			for (int i = 0; i < layer.w.height(); ++i) {
				for (int j = 0; j < layer.w.width(); ++j) {
					layer.w[i][j] = random<Real>() * 2 - 1;
				}
			}

		}
		// final layer
		output = Vector(prevLayerSize);
		outputError = Vector(prevLayerSize);
		desired = Vector(prevLayerSize);
	}



	void feedForward() {
		auto const numLayers = layers.size();
		for (size_t k = 0; k < numLayers; ++k) {
			auto & layer = layers[k];

			auto const & w = layer.w;
			auto const height = w.size.x;
			assert(w.size.y/*width*/ > 0);
			assert(height > 0);
			auto const storageWidth = w.storageSize.y;
			assert(roundup<8>(height) == w.storageSize.x/*storageHeight*/);

			const auto & x = layer.x;
			assert(x.storageSize == storageWidth);

			// not try necessarily, the weight will be zero, the input can be anything
			//assert(x.v[x.size] == (layer.getBias() ? 1 : 0));

			auto & net = layer.net;
			assert(net.size == height);

			assert(w.size.y/*width*/ == x.size+1);
			assert(height == net.size);

			auto & y = k == numLayers-1 ? output : layers[k+1].x;
			assert(y.storageSize == net.storageSize);
			assert(y.storageSize == roundup<8>(w.size.x/*height*/));

			auto const & activation = layer.activation.f;
			auto wij = w.v.data();
			auto xptr = x.v.data();
			auto const xendptr = xptr + storageWidth;
			auto neti = net.v.data();	//net.v.begin() ? which is faster?
			auto const netiend = neti + height;
			auto yi = y.v.data();

			for (; neti < netiend;
				++neti, ++yi
			) {
				auto xj = xptr;
				*neti = 0;
				for (; xj < xendptr;
					xj += 8, wij += 8
				) {
					neti[0] += wij[0] * xj[0]
							+ wij[1] * xj[1]
							+ wij[2] * xj[2]
							+ wij[3] * xj[3]
							+ wij[4] * xj[4]
							+ wij[5] * xj[5]
							+ wij[6] * xj[6]
							+ wij[7] * xj[7]
					;
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
		return Real(.5) * s;
	}

	template<typename Mul>
	void backPropagateWithPerWeightMul(Real dt, Mul mul) {
		int const numLayers = (int)layers.size();
		for (int k = (int)numLayers-1; k >= 0; --k) {
			auto & layer = layers[k];
			auto & y = k == numLayers-1 ? output : layers[k+1].x;
			auto & yErr = k == numLayers-1 ? outputError : layers[k+1].xErr;
			auto const & activationDeriv = layer.activationDeriv.f;
			auto const height = layer.w.height();
			assert(height == y.size);
			assert(height == layer.netErr.size);
			{
				auto neti = layer.net.v.data();
				auto neterri = layer.netErr.v.data();
				auto neterriend = neterri + height;
				auto yerri = yErr.v.data();
				auto yi = y.v.data();
				for (; neterri < neterriend;
					neterri += 8,
					neti += 8,
					yerri += 8,
					yi += 8
				) {
					neterri[0] = yerri[0] * activationDeriv(neti[0], yi[0]);
					neterri[1] = yerri[1] * activationDeriv(neti[1], yi[1]);
					neterri[2] = yerri[2] * activationDeriv(neti[2], yi[2]);
					neterri[3] = yerri[3] * activationDeriv(neti[3], yi[3]);
					neterri[4] = yerri[4] * activationDeriv(neti[4], yi[4]);
					neterri[5] = yerri[5] * activationDeriv(neti[5], yi[5]);
					neterri[6] = yerri[6] * activationDeriv(neti[6], yi[6]);
					neterri[7] = yerri[7] * activationDeriv(neti[7], yi[7]);
				}
			}
			// back-propagate error
#if 1
			{
				auto xerrj = layer.xErr.v.data();
				auto xerrjend = xerrj + layer.xErr.size;
				assert(layer.x.size == layer.xErr.size);
				assert(layer.x.size == layer.w.width()-1);
				auto neterrptr = layer.netErr.v.data();
				auto neterriend = neterrptr + height;
				auto wi = layer.w.v.data();
				for (; xerrj < xerrjend;
					++xerrj, ++wi
				) {
					*xerrj = {};
					auto neterri = neterrptr;
					auto wij = wi;
					for (; neterri < neterriend;
						neterri += 8,
						wij += 8 * layer.w.storageWidth()
					) {
						*xerrj += wij[0 * layer.w.storageWidth()] * neterri[0]
								+ wij[1 * layer.w.storageWidth()] * neterri[1]
								+ wij[2 * layer.w.storageWidth()] * neterri[2]
								+ wij[3 * layer.w.storageWidth()] * neterri[3]
								+ wij[4 * layer.w.storageWidth()] * neterri[4]
								+ wij[5 * layer.w.storageWidth()] * neterri[5]
								+ wij[6 * layer.w.storageWidth()] * neterri[6]
								+ wij[7 * layer.w.storageWidth()] * neterri[7]
						;
					}
				}
			}
#else
			for (int j = 0; j < layer.xErr.size; ++j) {
				Real sum = {};
				for (int i = 0; i < layer.netErr.size; ++i) {
					sum += layer.netErr[i] * layer.w[i][j];
				}
				layer.xErr[j] = sum;
			}
#endif

			// adjust new weights
			// not try necessarily, the weight will be zero, the input can be anything
			//assert(layer.x[layer.x.size] == (layer.getBias() ? 1 : 0));
			{
				BackProp<Real, Mul>::go(
					mul,
					layer.w.height(),
					layer.w.width(),
					layer.w.storageHeight(),
					layer.w.storageWidth(),
					useBatch 				//destwptr
						? layer.dw.v.data() 	// ... accumulate into dw
						: layer.w.v.data(),		// ... directly/immediately
					layer.x.v.data(),		//xptr
					layer.netErr.v.data(),	//neterrptr
					dt
				);
			}
		}

		if (useBatch) {
			++batchCounter;
			if (batchCounter >= useBatch) {
				updateBatch();
				batchCounter = 0;
			}
		}
	}
	void backPropagate(Real dt) {
		if (dropout == Real(1) && dilution == Real(1)) {
			backPropagateWithPerWeightMul<One<Real>>(dt, One<Real>());
		} else if (dropout != Real(1)) {
			backPropagateWithPerWeightMul<Dropout<Real>>(dt, Dropout<Real>(dropout));
		} else {	// with dilution
			backPropagateWithPerWeightMul<Dilution<Real>>(dt, Dilution<Real>(dilution));
		}
	}
	void backPropagate() {
		backPropagate(dt);
	}

	// update weights by batch ... and then clear the batch
	template<typename Mul>
	void updateBatchWithPerWeightMul(Mul mul) {
		if (!useBatch) return;
		for (int k = (int)layers.size()-1; k >= 0; --k) {
			auto & layer = layers[k];
			UpdateBatch<Real, Mul>::go(
				mul,
				layer.w.height(),
				layer.w.width(),
				layer.w.storageHeight(),
				layer.w.storageWidth(),
				layer.w.v.data(),	//wptr
				layer.dw.v.data()	//dwptr
			);
		}
		clearBatch();
	}
	void updateBatch() {
		if (dropout == Real(1) && dilution == Real(1)) {
			updateBatchWithPerWeightMul<One<Real>>(One<Real>());
		} else if (dropout != Real(1)) {
			updateBatchWithPerWeightMul<Dropout<Real>>(Dropout<Real>(dropout));
		} else {
			updateBatchWithPerWeightMul<Dilution<Real>>(Dilution<Real>(dilution));
		}
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
