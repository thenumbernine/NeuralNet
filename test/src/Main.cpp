#include "NeuralNet/ANN.h"
#include <iostream>

void accuracy() {
	//NeuralNet::ANN nn{222, 80, 40, 2};
	NeuralNet::ANN nn{5, 4, 3, 2};
	for (auto & layer : nn.layers) {
		layer.activation = NeuralNet::Activation<>::get("identity");
		layer.activationDeriv = NeuralNet::ActivationDeriv<>::get("one");
	}

	double x = 0;
	auto src = [&]() -> double {
		++x;
		return sin(1./x);
	};

	for (auto & layer : nn.layers) {
		auto & w = layer.w;
		for (int i = 0; i < w.height(); ++i) {
			for (int j = 0; j < w.width(); ++j) {
				w[i][j] = src();
			}
		}
	}
	for (int i = 0; i < nn.input().size; ++i) {
		nn.input()[i] = src();
	}
	nn.feedForward();
	
	//std::cout << "input " << nn.input() << std::endl;
	std::cout << "input L1 norm " << nn.input().normL1() << std::endl;
	//std::cout << "w[0] " << nn.layers[0].w << std::endl;
	std::cout << "w[0] L1 norm " << nn.layers[0].w.normL1() << std::endl;
	//std::cout << "x[1] " << nn.layers[1].x << std::endl;
	std::cout << "x[1] L1 norm " << nn.layers[1].x.normL1() << std::endl;
	//std::cout << "w[1] " << nn.layers[1].w << std::endl;
	std::cout << "w[1] L1 norm " << nn.layers[1].w.normL1() << std::endl;
	//std::cout << "x[2] " << nn.layers[2].x << std::endl;
	std::cout << "x[2] L1 norm " << nn.layers[2].x.normL1() << std::endl;
	//std::cout << "w[2] " << nn.layers[2].w << std::endl;
	std::cout << "w[2] L1 norm " << nn.layers[2].w.normL1() << std::endl;
	//std::cout << "output " << nn.output << std::endl;
	std::cout << "output L1 norm " << nn.output.normL1() << std::endl;

	nn.desired.v[0] = src();
	nn.desired.v[1] = src();
	//std::cout << "desired " << nn.desired << std::endl;
	std::cout << "desired L1 norm " << nn.desired.normL1() << std::endl;
	auto totalError = nn.calcError();
	std::cout << "total error " << totalError << std::endl;
	//std::cout << "outputError " << nn.outputError << std::endl;
	std::cout << "outputError L1 norm " << nn.outputError.normL1() << std::endl;
	nn.backPropagate();
	//std::cout << "netErr[2] " << nn.layers[2].netErr << std::endl;
	std::cout << "netErr[2] L1 norm " << nn.layers[2].netErr.normL1() << std::endl;
	//std::cout << "xErr[2] " << nn.layers[2].xErr << std::endl;
	std::cout << "xErr[2] L1 norm " << nn.layers[2].xErr.normL1() << std::endl;
	//std::cout << "w[2] " << nn.layers[2].w << std::endl;
	std::cout << "w[2] L1 norm " << nn.layers[2].w.normL1() << std::endl;
	//std::cout << "netErr[1] " << nn.layers[1].netErr << std::endl;
	std::cout << "netErr[1] L1 norm " << nn.layers[1].netErr.normL1() << std::endl;
	//std::cout << "xErr[1] " << nn.layers[1].xErr << std::endl;
	std::cout << "xErr[1] L1 norm " << nn.layers[1].xErr.normL1() << std::endl;
	//std::cout << "w[1] " << nn.layers[1].w << std::endl;
	std::cout << "w[1] L1 norm " << nn.layers[1].w.normL1() << std::endl;
	//std::cout << "netErr[0] " << nn.layers[0].netErr << std::endl;
	std::cout << "netErr[0] L1 norm " << nn.layers[0].netErr.normL1() << std::endl;
	//std::cout << "xErr[0] " << nn.layers[0].xErr << std::endl;
	std::cout << "xErr[0] L1 norm " << nn.layers[0].xErr.normL1() << std::endl;
	//std::cout << "w[0] " << nn.layers[0].w << std::endl;
	std::cout << "w[0] L1 norm " << nn.layers[0].w.normL1() << std::endl;
}

#include "Common/Profile.h"
void performance() {
	auto nn = NeuralNet::ANN{222, 80, 40, 2};

	for (int i = 0; i < nn.input().size; ++i) {
		nn.input().v[i] = NeuralNet::random();
	}

	int numIter = 10000;
	Common::timeFunc("feedForward + backPropagate", [&](){
		for (int i = 0; i < numIter; ++i) {
			nn.feedForward();
			nn.desired[0] = random();
			nn.desired[0] = random();
			nn.calcError();
			nn.backPropagate();
		}
	});
	Common::timeFunc("feedForward only", [&](){
		for (int i = 0; i < numIter; ++i) {
			nn.feedForward();
		}
	});
}

#if 0  // unit tests
	{
		static constexpr myint m = 15;
		static constexpr myint n = 15;
		auto nn = NeuralNet::ANN<real>{m,n};
		nn.layers[0].setBias(false);

		// test that w[j][i] alone works
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				// y_j = w_ji x_i

				// set only w[i][j], i.e. w[k][l] = δ_ik δ_jl
				for (int k = 0; k < m; ++k) {
					for (int l = 0; l < n; ++l) {
						nn.layers[0].w[l][k] = i == k && j == l ? 1 : 0;
					}
				}

				// make sure for all inputs k, it only produces output[j] for when k==i and zero otherwise
				for (int k = 0; k < m; ++k) {
					// set only x_k, i.e. x_l = δ_kl
					for (int l = 0; l < m; ++l) {
						nn.input()[l] = k == l ? 1 : 0;
					}

					// for the weight w_ij being set (and all others zero)
					// we expect x_i to be set => y_j to be set (and all other input/output's zero)

					nn.feedForward();
				
					for (int l = 0; l < n; ++l) {
						real expected = i == k && j == l ? 1 : 0;
						if (nn.layers[0].net[l] != expected) {
							std::cerr
								<< " i=" << i
								<< " j=" << j
								<< " k=" << k
								<< " l=" << l
								<< " ...";
							std::cerr << " got bad value"
								<< " expected=" << expected
								<< " got=" << nn.layers[0].net[l]
								<< std::endl;
							std::cerr << "x=" << nn.input() << std::endl;
							std::cerr << "w=" << nn.layers[0].w << std::endl;
							std::cerr << "y=" << nn.layers[0].net << std::endl;
							std::cerr << std::endl;
						}
					}
				}
			}
		}
		return 0;
	}
#endif

int main() {
	accuracy();
	performance();
}
