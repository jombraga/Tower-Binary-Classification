from load_data import load_data
import matplotlib.pyplot as plt
import time
import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


categories = ['positive', 'negative']
train_dir = '/home/admin1/CSD/ALaM/Jom/SVM/train'
valid_dir = '/home/admin1/CSD/ALaM/Jom/SVM/test'

print('Loading training data.')
x_train, y_train = load_data(
    dir=train_dir, categories=categories, img_size=(150, 150, 3))

print('Loading test data.')
x_valid, y_valid = load_data(
    dir=valid_dir, categories=categories, img_size=(150, 150, 3))

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [
    0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}

svc = svm.SVC(probability=True)
model = GridSearchCV(svc, param_grid)

start = time.time()
print('Training....')
model.fit(x_train, y_train)
print('Training completed')
print(model.best_params_)
print(f'Training took {(time.time() - start)/60} minutes.')

y_pred = model.predict(x_test)
print(f'The model is {accuracy_score(y_pred, y_valid)*100}% accurate.')

filename = time.strftime('%Y%m%d-%H%M%S') + '.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
