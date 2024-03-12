#include "NeuralNet/ANN.h"
#include <iostream>

void accuracy() {
	//NeuralNet::ANN nn{222, 80, 40, 2};
	NeuralNet::ANN nn{5, 4, 3, 2};
	for (auto & layer : nn.layers) {
		layer.activation = [](double x) -> double { return x; };
		layer.activationDeriv = [](double x, double y) -> double { return 1.; };
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

int main() {
	accuracy();
//	performance();
}
