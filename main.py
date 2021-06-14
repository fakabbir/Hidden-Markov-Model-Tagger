from dataset import Dataset
from collections import defaultdict, Counter

class CountBasedTagger:
	def __init__(self):
		data  = Dataset(split = 0.9)
		self.count_table = {}
		token = []
		for sentence in data.train_dataset:
			for pair in sentence:
				if pair[0] in self.count_table:
					self.count_table[pair[0]].append(pair[1])
				else:
					self.count_table[pair[0]] = [pair[1]]
				token.append(pair[1])
		self.missing_token = Counter(token).most_common()[0][0]
		self.lookup_table = {}
		for i in self.count_table:
			self.lookup_table[i] = Counter(self.count_table[i]).most_common()[0][0]


c = CountBasedTagger()

evaluate_data = Dataset().test_dataset


# s1 = test_data[0]

wrong_classification = 0
total_words = 0 
for i in evaluate_data:
	for j in i :
		total_words = total_words + 1

for s1 in evaluate_data:
	for i in s1:
		token = i[0]
		tag = c.missing_token
		if token in c.lookup_table:
			tag = c.lookup_table[token]
		if i[1] != tag:
			print("Token -> {} | Predicted -> {} | Actual -> {}".format(i[0], tag, i[1]))
			wrong_classification = wrong_classification + 1

print("Accuracy - > {}".format((total_words-wrong_classification)/total_words))



def get_biagrams(data):
	biagram_pairs = {}
	for sentence in data:
		for i in range(len(sentence) - 1):
			if (sentence[i][1], sentence[i+1][1]) in biagram_pairs:
				biagram_pairs[(sentence[i][1], sentence[i+1][1])] += 1
			else:
				biagram_pairs[(sentence[i][1], sentence[i+1][1])] = 1

	return biagram_pairs



class HMMTagger:
	def __init__(self, split):
		data  = Dataset(split = 0.9)

		biagram_pairs = get_biagrams(data.train_dataset)
		print(biagram_pairs)