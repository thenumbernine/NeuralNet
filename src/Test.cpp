#include "NeuralNet/ANN.h"
#include <iostream>

int main() {
	// TODO list ctor plz
	NeuralNet::ANN nn(std::vector<int>{
		222, 80, 40, 2
	});

	double x = 0;
	auto src = [&]() -> double {
		++x;
		return sin(1./x);
	};

	for (auto & layer : nn.layers) {
		for (auto & wij : layer.w.v) {
			wij = src();
		}
	}
	for (auto & xi : nn.input().v) {
		xi = src();
	}
	nn.feedForward();
	
	std::cout << "input " << nn.input() << std::endl;
	std::cout << "input L1 norm " << nn.input().normL1() << std::endl;
	std::cout << "hidden " << nn.layers[1].x << std::endl;
	std::cout << "hidden L1 norm " << nn.layers[1].x.normL1() << std::endl;
	std::cout << "output " << nn.output << std::endl;
	std::cout << "output L1 norm " << nn.output.normL1() << std::endl;
}
