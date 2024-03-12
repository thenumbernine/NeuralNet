#include "NeuralNet/ANN.h"
#include <iostream>

void accuracy() {
	//NeuralNet::ANN nn(std::vector<int>{222, 80, 40, 2});
	NeuralNet::ANN nn{222, 80, 40, 2};
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
	
	std::cout << "input " << nn.input() << std::endl;
	std::cout << "input L1 norm " << nn.input().normL1() << std::endl;
	std::cout << "hidden " << nn.layers[1].x << std::endl;
	std::cout << "hidden L1 norm " << nn.layers[1].x.normL1() << std::endl;
	std::cout << "hidden " << nn.layers[2].x << std::endl;
	std::cout << "hidden L1 norm " << nn.layers[2].x.normL1() << std::endl;
	std::cout << "output " << nn.output << std::endl;
	std::cout << "output L1 norm " << nn.output.normL1() << std::endl;

	nn.desired.v[0] = src();
	nn.desired.v[1] = src();
	std::cout << "desired " << nn.desired << std::endl;
	std::cout << "desired L1 norm " << nn.desired.normL1() << std::endl;
	nn.calcError();
	nn.backPropagate();
	std::cout << "input error " << nn.inputError() << std::endl;
	std::cout << "input error L1 norm " << nn.inputError().normL1() << std::endl;
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
//	accuracy();
	performance();
}
