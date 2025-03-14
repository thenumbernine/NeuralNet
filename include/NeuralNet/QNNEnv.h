#pragma once

#include <vector>
#include <iostream>

#include "NeuralNet/ANN.h"	//only for NeuralNet::random() right now

/*
Controller needs:
	Real
	State
	nn = createNeuralNet()
	state = initState()	// but why not just put it in the State ctor?
	observe(State, nn)	// fill in the inputs of the NN
	State newState = performAction(state, action, actionQ)
	[reward, reset] = getReward(newState)

... why not combine Controller and State?
*/
template<typename Controller>
struct QNNEnv {
	using Real = typename Controller::Real;
	using State = typename Controller::State;
	using NN = decltype(Controller::createNeuralNet());

	State state;
	NN nn;

	Real alpha = .1;
	Real gamma = .99;
	Real lambda = .7;
	Real noise = 0;

	std::vector<std::tuple<State, int, Real>> history;
	int historySize = 10;

	std::vector<int> actionCount;
	QNNEnv()
	: nn(Controller::createNeuralNet())
	{
		state = Controller::initState();
		resetActionCount();
	}

	void resetActionCount() {
		actionCount = std::vector<int>(nn.output.size);
	}

	void feedForwardForState(State const & state) {
		Controller::observe(state, nn);
		nn.feedForward();
	}

	std::pair<int, Real> determineAction(State const & state, Real noise = 0) {
		// propagate input -> output = actions
		feedForwardForState(state);

		// softmax with random choice from the best
		Real bestValue = nn.output[0];
		if (noise) bestValue += noise * NeuralNet::random<Real>();
		int bestAction = 0;
		for (int i = 1; i < nn.output.size; ++i) {
			Real checkValue = nn.output[i];
			if (noise) checkValue += noise * NeuralNet::random<Real>();	// is it get(M, row, col) or get(M, col, row) ?
			if (bestValue < checkValue) {
				bestValue = checkValue;
				bestAction = i;
			}
		}

		// don't return bestValue since it could have noise in it
		return std::make_pair(bestAction, nn.output[bestAction]);
	}

	Real applyReward(
		State const & newState,
		Real reward,
		State const & lastState,
		int lastAction,
		Real lastActionQ
	) {
		//feed-forward from the new state to get the max next Q?
		// btw is this what 'critic' is? it's a separate nn for evaluating this?
		// but then how do the weights of critic and actor relate?  they don't?  wtf?
		Real maxNextQ = std::get<1>(determineAction(newState, 0));
		// maxNextQ = max(Q(S[t+1], *))

		// restore the inputs & weights to the state before action for backprop's sake
		feedForwardForState(lastState);
		// fill in 'outputError' based on state, newstate, reward, etc
#if 1	// reward only the action taken
		for (int i = 0; i < nn.output.size; ++i) {
			nn.outputError[i] = 0;
		}
		Real err = reward + gamma * maxNextQ - lastActionQ;
		nn.outputError[lastAction] = err;
#endif
#if 0	// reward all action signals according to their output?
		for (int i = 0; i < nn.output.size; ++i) {
			nn.outputError[i] = nn.output[i] * reward + gamma * maxNextQ
				- histActionQs[i]
			;
		}
#endif
		// backprop reward -> outputError -> weights
		nn.backPropagate(alpha);

		if (history.size() > 0) {
			// can I TD-lambda by accumulating outputError or should I backProp for each history individually?  batch or no batch?
			for (int i = 0; i < history.size(); ++i) {
				err *= lambda;
				auto [histState, histAction, histActionQ] = history[i];
				feedForwardForState(histState);
				for (int j = 0; j < nn.output.size; ++j) {
					nn.outputError[j] = 0;
				}
				nn.outputError[histAction] = err;
				nn.backPropagate(alpha);
			}
		}

		return err;
	}

	std::pair<Real, bool> step() {
//std::cout << "step():" << std::endl;
//std::cout << "state=" << state << std::endl;
//std::cout << "nn.layers[0].w =" << nn.layers[0].w << std::endl;
//std::cout << "nn.layers[1].w =" << nn.layers[1].w << std::endl;

		// fill inputs
//std::cout << "nn.input = " << nn.input() << std::endl;
//std::cout << "nn.output = " << nn.output << std::endl;
		// state = S[t]
		auto [action, actionQ] = determineAction(state, noise);
		// action = A[t]
		// actionQ = Q(S[t], A[t])
		++actionCount[action];
		//auto actionQs = std::vector(nn.output.v);

		// act on env state
		State newState = Controller::performAction(state, action, actionQ);
		// newState = S[t+1]

		// determine reward of action
		auto [reward, reset] = Controller::getReward(newState);
		// reward = R[t+1]

		applyReward(newState, reward, state, action, actionQ);

		//TD-lambda: add to history after applyReward (so it doesn't get considered by applyReward)
		if (historySize > 0) {
			history.insert(history.begin(), std::make_tuple(state, action, actionQ));
			if (history.size() > historySize) {
				history.resize(historySize);
			}
		}

		// update env state
		state = newState;

		// reset condition TODO where to put this ...
		if (reset) {
			state = Controller::initState();
		}

		return std::make_pair(reward, reset);
	}

	void runForever() {
		for (;;) {
			step();
		}
	}

	void run(int maxsteps) {
		run(maxsteps, maxsteps + 1);
	}

	void run(int maxsteps, int numEval) {
		Real avgReward = {};
		for (int stepIndex = 0, eval = 0; stepIndex < maxsteps; ++stepIndex, ++eval) {
			auto [reward, reset] = step();
			avgReward += reward;
			if (eval >= numEval) {
				std::cout << "stepIndex="
					<< stepIndex
					<< " avgReward=" << (avgReward/(Real)numEval)
					<< " actionCount=" << actionCount
					//<< " output=" << nn.output
					<< std::endl;
				resetActionCount();
				eval = 0;
				avgReward = 0;
			}
		}
	}
};
