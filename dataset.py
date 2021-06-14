from random import randint

def prepare_words_tags(dataset):	
	length = len(dataset)

	tags = []
	words = []
	for sentance in dataset:
		for word, tag in sentance:
			tags.append(tag)
			words.append(word)



	tags = list(set(tags))
	tokens = list(set(words))
	dataset = dataset

	return dataset, tokens, tags

def splitter( dataset, split = 0.8):
	train_length = int(split * len(dataset))
	train_index = []
	while train_length:
		x = randint(0, len(dataset))
		if x not in train_index:
			train_index.append(x)
			train_length = train_length - 1

	train_dataset = []
	test_dataset = []

	for i in range(len(dataset)):
		if i in train_index:
			train_dataset.append(dataset[i])
		else:
			test_dataset.append(dataset[i])

	return test_dataset, train_dataset


class Dataset:
	def __init__(self, split = 0.8):
		with open('train.txt') as f:
			data = f.readlines()
		dataset = []
		sentance = []
		for i in data:
			row = i.strip()
			if len(row) == 0:
				dataset.append(sentance)
				sentance = []
			else:
				word, token, _ = row.split(" ")
				sentance.append((word,token))

		test_dataset, train_dataset = splitter(dataset,split)
		self.train_dataset, self.train_tokens, self.train_tags = prepare_words_tags(train_dataset)
		self.test_dataset, self.test_tokens, self.test_tags = prepare_words_tags(test_dataset)