#include "NeuralNet/QNNEnv.h"
#include "NeuralNet/ANN.h"	// QNNEnv incl this?
#include <algorithm>

using real = double;
using NN = NeuralNet::ANN<>;

constexpr auto rad(auto const x) { return x * M_PI / 180.; }

enum {
	ACTION_LEFT,	//putting IDLE first makes it choose idle too often and fail more .. hmm....
	ACTION_IDLE,
	ACTION_RIGHT,
	NUM_ACTIONS,
};

struct State {
	real x = {};
	real dt_x = {};
	real theta = {};
	real dt_theta = {};
	int itersUpright = {};
};

std::ostream & operator<<(std::ostream & o, State const & s) {
	return o << "{"
		<< "x=" << s.x
		<< ", θ=" << s.theta
		<< ", ∂x/∂t=" << s.dt_x
		<< ", ∂θ/∂t=" << s.dt_theta
		<< ", t=" << s.itersUpright 
		<< "}";
}

static constexpr int xBins = 3;
static constexpr int dtxBins = 3;
static constexpr int thetaBins = 6;
static constexpr int dtthetaBins = 3;

struct Problem {
	using Real = real;
	using State = State;

	static auto createNeuralNet() {
		//auto nn = NN{3, 64, 3};	// nan weights?
		static constexpr int inputSize = xBins * dtxBins * thetaBins * dtthetaBins;
		static constexpr int outputSize = 3;
		auto nn = NN{inputSize, outputSize};
		nn.layers[0].setBias(false);
		nn.dilution = 1;
		nn.dropout = 1;
		nn.useBatch = 0;
		nn.layers[0].setActivation("identity");
		for (int i = 0; i < outputSize; ++i) {
			for (int j = 0; j < inputSize; ++j) {
				nn.layers[0].w[i][j] = 0;//(NeuralNet::random<real>() * 2. - 1.);
			}
		}
		return nn;
	}

	static State initState() {
		State state;
		state.theta = (NeuralNet::random<real>() * 2. - 1.) * rad(6.);	// +- 6 radians
//std::cout << "initState " << state << std::endl;
		return state;
	}

	static State performAction(
		State const & state,
		int action,
		Real actionQ
	) {
		constexpr real gravity = 9.8;
		constexpr real massCart = 1;
		constexpr real massPole = .1;
		constexpr real totalMass = massPole + massCart;
		constexpr real length = .5;
		constexpr real poleMassLength = massPole * length;
		constexpr real forceMag = 20;
		constexpr real dt = .02;

//std::cout << "action=" << action << std::endl;
		real force = 0;
		if (action == ACTION_LEFT) {
			force = -forceMag;
		} else if (action == ACTION_RIGHT) {
			force = forceMag;
		}

		State newState = state;
		
		real cosTheta = std::cos(newState.theta);
		real sinTheta = std::sin(newState.theta);
		real temp = (force + poleMassLength * newState.dt_theta * newState.dt_theta * sinTheta) / totalMass;
		real dt2_theta = (gravity * sinTheta - cosTheta * temp) / (length * (4./3. - massPole * cosTheta * cosTheta / totalMass));
		real dt2_x = temp - poleMassLength * dt2_theta * cosTheta / totalMass;

		newState.itersUpright++;
		newState.x += dt * newState.dt_x;
		newState.theta += dt * newState.dt_theta;
		newState.dt_x += dt * dt2_x;
		newState.dt_theta += dt * dt2_theta;

//std::cout << "step " << state << " => " << newState << std::endl;

		return newState;
	}

	static std::pair<real, bool> getReward(
		State const & state
	) {
		bool fail = state.x < -2.4 
			|| state.x > 2.4 
			|| state.theta < rad(-12.)
			|| state.theta > rad(12.);
		static constexpr int successIterations = 100000;
		bool success = state.itersUpright > successIterations;
		bool reset = fail || success;
		if (reset) {
			std::cout << state.itersUpright 
				//<< " w=" << env.nn.layers[0].w
				<< std::endl;
		}
		return std::make_pair(fail ? -1. : .001, reset);
	}

	static void observe(
		State const & state,
		NN & nn
	) {
		for (int i = 0; i < nn.input().size; ++i) {
			nn.input()[i] = 0;
		}

		if (state.x < -2.4 
			|| state.x > 2.4 
			|| state.theta < rad(-12.)
			|| state.theta > rad(12.)
		) {
			return;// 0; //invalid state means we've failed
		}

		auto bin = [](real x, real min, real max, int n) {
			return std::clamp(
				(int)((x - min) / (max - min) * (n - 2)),
				-1, 
				n-2
			) + 1;
		};

		int xIndex = bin(state.x, -.8, .8, xBins);
		int dtxIndex = bin(state.dt_x, -.5, .5, dtxBins);

		// state.theta is nonlinear...
		int thetaIndex = {};
		if (state.theta < rad(-6.)) {
			thetaIndex = 0;
		} else if (state.theta < rad(-1.)) {
			thetaIndex = 1;
		} else if (state.theta < 0.) {
			thetaIndex = 2;
		} else if (state.theta < rad(1.)) {
			thetaIndex = 3;
		} else if (state.theta < rad(6.)) {
			thetaIndex = 4;
		} else {
			thetaIndex = 5;
		}

		int dtthetaIndex = bin(state.dt_theta, rad(-50.), rad(50.), dtthetaBins);

		assert(xIndex >= 0 && xIndex < xBins);
		assert(dtxIndex >= 0 && dtxIndex < dtxBins);
		assert(thetaIndex >= 0 && thetaIndex < thetaBins);
		assert(dtthetaIndex >= 0 && dtthetaIndex < dtthetaBins);
		
		int stateIndex = xIndex + xBins * (
			dtxIndex + dtxBins * (
				thetaIndex + thetaBins * dtthetaIndex
			)
		);
		assert(stateIndex >= 0 && stateIndex < nn.input().size);
		nn.input()[stateIndex] = 1.;
	}
};

int main(int argc, char** argv) {
	srand(time(nullptr));
	QNNEnv<Problem> env;
	env.alpha = .1;
	env.gamma = .9;
	env.lambda = .7;
	env.historySize = 10;

	//env.run();
	env.runForever();
}
