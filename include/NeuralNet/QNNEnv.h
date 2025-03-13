#pragma once

#include <vector>
#include <iostream>

/*
Problem needs:
	Real
	State
	nn = createNeuralNet()
	state = initState()	// but why not just put it in the State ctor?
	observe(State, nn)	// fill in the inputs of the NN
	State newState = step(state, action)
	[reward, reset] = reward(newState)
*/
template<typename Problem>
struct QNNEnv {
	using Real = typename Problem::Real;
	using State = typename Problem::State;

	State state;
	decltype(Problem::createNeuralNet()) nn;
	Real lastStateActionQ = {};
	std::vector<Real> lastStateActionQs;

	std::vector<int> actionCount;
	QNNEnv() 
	: nn(Problem::createNeuralNet())
	{
		resetState();
		resetActionCount();
	}

	void resetState() {
		state = Problem::initState();
		lastStateActionQ = {};
		lastStateActionQs.clear();
		lastStateActionQs.resize(nn.output.size);
	}

	void resetActionCount() {
		actionCount = std::vector<int>(nn.output.size);
	}

	std::pair<int, Real> getOutputMaxQ() {

// TODO epsilon to randomness for choosing actions, but only for choosing the action,not for getting next state max Q
#if 0
		auto rand = [&]() {
			constexpr static Real actionChoiceEpsilon = 1e-2;
			return random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (Real)0, (Real)actionChoiceEpsilon, rng);
		};
#else
		auto rand = []() { return 0; };
#endif

		// softmax
		int bestAction = 0;
		Real bestValue = nn.output[0] + rand();
		for (int i = 1; i < nn.output.size; ++i) {
			Real checkValue = nn.output[i] + rand();	// is it get(M, row, col) or get(M, col, row) ?
			if (bestValue < checkValue) {
				bestValue = checkValue;
				bestAction = i;
			}
		}

		return std::make_pair(bestAction, bestValue);
	}

	static constexpr Real alpha = .1;
	static constexpr Real gamma = .99;
	Real step() {
//std::cout << "step():" << std::endl;
//std::cout << "state=" << state << std::endl;
//std::cout << "nn.layers[0].w =" << nn.layers[0].w << std::endl;
//std::cout << "nn.layers[1].w =" << nn.layers[1].w << std::endl;

		// fill inputs
		Problem::observe(state, nn);
//std::cout << "nn.input = " << nn.input() << std::endl;	
		// propagate input -> output = actions
		nn.feedForward(); 			
//std::cout << "nn.output = " << nn.output << std::endl;	
		auto [action, actionQ] = getOutputMaxQ();
		++actionCount[action];
		
		// act on env state
		State newState = Problem::step(state, action);	

		// determine reward of action
		auto [reward, reset] = Problem::reward(newState);
		
		//feed-forward from the new state to get the max next Q?
		// btw is this what 'critic' is? it's a separate nn for evaluating this?
		// but then how do the weights of critic and actor relate?  they don't?  wtf?
		Problem::observe(newState, nn);
		nn.feedForward();
		Real maxNextQ = std::get<1>(getOutputMaxQ());
		// restore the inputs & weights to the state before action for backprop's sake
		Problem::observe(state, nn);
		nn.feedForward();
		
		// fill in 'outputError' based on state, newstate, reward, etc
#if 1	// reward only the action taken
		for (int i = 0; i < nn.output.size; ++i) {
			nn.outputError[i] = 0;
		}
		nn.outputError[action] = reward + gamma * maxNextQ
			- lastStateActionQ
		;
		lastStateActionQ = nn.output[action];
#endif
#if 0	// reward all action signals according to their output?
		for (int i = 0; i < nn.output.size; ++i) {
			nn.outputError[i] = nn.output[i] * reward + gamma * maxNextQ
				- lastStateActionQs[i]
			;
		}
		lastStateActionQs = nn.output.v;
#endif
		// backprop reward -> outputError -> weights
		nn.backPropagate(alpha);
		
		// update env state
		state = newState;

		// reset condition TODO where to put this ...
		if (reset) {
			resetState();
		}
	
		return reward;
	}

	void run(int maxsteps) {
		run(maxsteps, maxsteps + 1);
	}

	void run(int maxsteps, int numEval) {
		Real avgReward = {};
		for (int stepIndex = 0, eval = 0; stepIndex < maxsteps; ++stepIndex, ++eval) {
			avgReward += step();
			if (eval >= numEval) {
				std::cout << "stepIndex=" 
					<< stepIndex 
					<< " avgReward=" << (avgReward/(Real)numEval) 
					<< " actionCount=" << actionCount 
					<< " output=" << nn.output
					<< std::endl;
				resetActionCount();
				eval = 0;
				avgReward = 0;
			
				// reset here too?
				resetState();
			}
		}
	}
};
