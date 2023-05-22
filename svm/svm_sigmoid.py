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


svc = svm.SVC(C=1, kernel='sigmoid', degree=3, gamma='scale', probability=True)


start = time.time()
print('Training....')
svc.fit(x_train, y_train)
print('Training completed')
print(f'Training took {(time.time() - start)/60} minutes.')

y_pred = svc.predict(x_valid)
print(f'The model is {accuracy_score(y_pred, y_valid)*100}% accurate.')

filename = 'svm_sigmoid' + time.strftime('%Y%m%d-%H%M%S') + '.pkl'
with open(filename, 'wb') as file:
    pickle.dump(svc, file)
