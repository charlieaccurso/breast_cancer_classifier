from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# load data
breast_cancer_data= load_breast_cancer()

# print the first datapoint in the set, print feature names
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)

# print the targets and target names
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)
# results: 0 corresponds to malignant, 1 to benign

# Splitting the data into Training and Validation (Test) Sets
training_data, validation_data, training_labels, validation_labels= train_test_split(
  breast_cancer_data.data,
  breast_cancer_data.target,
  test_size=0.2,
  random_state=100
)
# check to make sure there are an equal number of labels and datapoints
print(len(training_data))
print(len(training_labels))

# Running the Classifier
best_k= 0
best_score= 0
accuracies= [] # y-values for graph later
for i in range(1, 101):
  classifier= KNeighborsClassifier(n_neighbors=i)
  classifier.fit(training_data, training_labels)
  # get classifier score
  score= classifier.score(validation_data, validation_labels)
  accuracies.append(score)
  if score > best_score:
    best_k= i
    best_score= score
  print("For k= {}: ".format(i) + str(score))
print(best_k)

# Graphing the results
k_list= range(1, 101)

# plot accuracies by k
plt.plot(k_list, accuracies)
plt.axvline(x=best_k, color='r')
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()
plt.clf()
