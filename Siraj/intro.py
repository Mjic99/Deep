from sklearn import tree, svm, neural_network, discriminant_analysis

models = [
	tree.DecisionTreeClassifier(),
	svm.SVC(),
	neural_network.MLPClassifier(hidden_layer_sizes=(5, 7, 9, 2, 8)),
	discriminant_analysis.LinearDiscriminantAnalysis(),
	neural_network.MLPClassifier()
]


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data

for model in models:
	model.fit(X,Y)
	prediction = model.predict(X)

	correct = 0
	for i in range(len(Y)):
		if Y[i] == prediction[i]:
			correct += 1

	accu = correct*100/len(Y)

	print(str(model)+"\naccuracy on training set: "+str(accu)+"%\n")