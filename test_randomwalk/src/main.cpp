#include "NeuralNet/QNNEnv.h"
#include "NeuralNet/ANN.h"	// QNNEnv incl this?
#include <algorithm>

using real = double;
using NN = NeuralNet::ANN<>;

constexpr auto rad(auto const x) { return x * M_PI / 180.; }

constexpr int size = 11;

enum {
	ACTION_LEFT,	//putting IDLE first makes it choose idle too often and fail more .. hmm....
	ACTION_RIGHT,
	NUM_ACTIONS,
};

struct State {
	int index = {};
};

std::ostream & operator<<(std::ostream & o, State const & s) {
	return o << s.index;
}

struct Problem {
	using Real = real;
	using State = State;

	static auto createNeuralNet() {
		static constexpr int inputSize = size;
		static constexpr int outputSize = 2;
		auto nn = NN{inputSize, outputSize};
		nn.layers[0].setBias(false);
		nn.dilution = 1;
		nn.dropout = 1;
		nn.useBatch = 0;
		nn.layers[0].setActivation("identity");
		for (int i = 0; i < outputSize; ++i) {
			for (int j = 0; j < inputSize; ++j) {
				nn.layers[0].w[i][j] = 0;
			}
		}
		return nn;
	}

	static State initState() {
		State state;
		state.index = size / 2;
//std::cout << "initState " << state << std::endl;
		return state;
	}

	static State performAction(
		State const & state,
		int action,
		Real actionQ
	) {
		State newState = state;
		if (action == ACTION_LEFT) {
			newState.index--;
		} else if (action == ACTION_RIGHT) {
			newState.index++;
		}
		return newState;
	}

	static std::pair<real, bool> getReward(
		State const & state
	) {
		real reward = 0;
		real reset = false;
		if (state.index <= -1) {
			reward = -1;
			reset = true;
		} else if (state.index >= size) {
			reward = 1;
			reset = true;
		}
		return std::make_pair(reward, reset);
	}

	static void observe(
		State const & state,
		NN & nn
	) {
		for (int i = 0; i < nn.input().size; ++i) {
			nn.input()[i] = 0;
		}
		if (state.index >= 0 && state.index < nn.input().size) {
			nn.input()[state.index] = 1;
		}
	}
};

int main(int argc, char** argv) {
	srand(time(nullptr));
	QNNEnv<Problem> env;
	env.lambda = .1;
	env.historySize = 100;

	//env.run(100000, 10000);
	//env.runForever();
	for (int i = 0; i < 100; ++i) {
		env.step();
		std::cout << env.nn.layers[0].w << std::endl;
	}
}
