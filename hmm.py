from dataset import Dataset
from collections import defaultdict, Counter
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution



def get_transition_count(data):
	transition_pair = {}

	for sentence in data:
		for i in sentence:
			if i[1] in transition_pair:
				transition_pair[i[1]] +=1
			else:
				transition_pair[i[1]] = 1

	return transition_pair

def get_biagrams(data):
	biagram_pairs = {}
	for sentence in data:
		for i in range(len(sentence) - 1):
			if (sentence[i][1], sentence[i+1][1]) in biagram_pairs:
				biagram_pairs[(sentence[i][1], sentence[i+1][1])] += 1
			else:
				biagram_pairs[(sentence[i][1], sentence[i+1][1])] = 1

	return biagram_pairs

def sequence_start_times(data):
	sequence_start_count = {}
	for sentence in data:
		if sentence[0][1] in sequence_start_count:
			sequence_start_count[sentence[0][1]] += 1
		else:
			sequence_start_count[sentence[0][1]] = 1

	return sequence_start_count	
	
def sequence_end_times(data):
	sequence_end_count = {}
	for sentence in data:
		if sentence[-1][1] in sequence_end_count:
			sequence_end_count[sentence[-1][1]] += 1
		else:
			sequence_end_count[sentence[-1][1]] = 1

	return sequence_end_count	

def pair_tag_count(data):
	pair_count = {}
	for sentence in data:
		for token in sentence:
			if token[1] in pair_count:
				if token[0] in pair_count[token[1]]:
					pair_count[token[1]][token[0]] += 1
				else:
					pair_count[token[1]][token[0]] = 1
			else:
				pair_count[token[1]] = {}
				pair_count[token[1]][token[0]] = 1

	return pair_count




class HMMTagger:
	def __init__(self, split = 0.9):
		data  = Dataset(split = split)

		print("generating probabilities")

		transition_pair = get_transition_count(data.train_dataset)
		biagram_pairs = get_biagrams(data.train_dataset)
		pair_count = pair_tag_count(data.train_dataset)
		seq_start = sequence_start_times(data.train_dataset)
		seq_end = sequence_end_times(data.train_dataset)
		self.basic_model = HiddenMarkovModel(name="base-hmm-tagger")


		# Add State
		state = {}
		print("Adding state transitions")
		for tag in pair_count:
			total = transition_pair[tag]
			prob_dist = {key: value/total for key,value in pair_count[tag].items()}
			tag_distribution = DiscreteDistribution(prob_dist)
			temp_state = State(tag_distribution, name=tag)
			self.basic_model.add_state(temp_state)
			state[tag] = temp_state


		# Add Transition State
		print("Adding transitions")
		for pair in biagram_pairs:
			
			s_1 = pair[0]
			s_2 = pair[1]

			try:
				prob = biagram_pairs[s_2]/transition_pair[s_1]
			except:
				prob = 0
			self.basic_model.add_transition(state[s_1], state[s_2], prob)


		# Add starting transition
		print("Adding starting transitions")
		for pair in biagram_pairs:
			
			s_1 = pair[0]
			try:
				prob = seq_start[s_1]/len(data.train_dataset)
			except:
				prob =  0
			self.basic_model.add_transition(self.basic_model.start, state[s_1], prob)

		# Add ending transition
		print("Adding ending transitions")
		for pair in biagram_pairs:
			
			s_1 = pair[0]
			try:
				prob = seq_end[s_1]/len(data.train_dataset)
			except:
				prob = 0
			self.basic_model.add_transition(self.basic_model.end, state[s_1], prob)

		self.basic_model.bake()


model = HMMTagger()

# the DT B-NP
# pound NN I-NP
# is VBZ B-VP
# widely RB I-VP
# expected

x = model.basic_model.viterbi(["pound", "is", "widely","expected"])
print(x)