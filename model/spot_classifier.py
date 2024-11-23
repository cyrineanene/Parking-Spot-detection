import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

#Step 1: Preparing the data
input_dir = 'data\\train_model'
categories = ['empty', 'not_empty']
data = []
labels = []

#Loading, reading and resizing the image to fit the classifier
for category_idx, category in enumerate(categories):
    print(f'Working on the {category}')
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15,15))
        data.append(img.flatten())
        labels.append(category_idx)

date = np.asarray(data)
labels = np.asarray(labels)

#Step 2: Prepare train/test Data
print('Working on splitting the data')
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) #stratify is keeping the proportions of the original dataset

#Step 3: Train the model
print('Working on initializing the model')
classifier = SVC()

parameters = [{'gamma':[0.01, 0.001, 0.0001], 'C':[1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)
print('Finished working on the model')

#Step 4: Evaluate the model
best_estimator = grid_search.best_estimator_

y_pred = best_estimator.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print(f'{str(acc*100)}% of samples were correctly classified')

#Step 5: Saving the model
pickle.dump(best_estimator, open('model/classifier.p', 'wb'))
print('Saved the model')