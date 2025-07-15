#importing necessary libraries
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def read_images(folder): # reading images from corresponding folder
    df = pd.read_csv(folder + '.csv')
    image_files = df['image_id'].astype('string').to_list()
    features_vectors = []
    for file_name in image_files:
      img = cv2.imread(folder + '/' + file_name + '.png')
      hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]) # one channel for each: R, G, B
      hist = cv2.normalize(hist, hist).flatten() # normalization and flattening
      hist = np.array(hist)
      features_vectors.append(hist)
    return features_vectors

def read_labels(file_name): # reading labels from .csv file
    df = pd.read_csv(file_name + '.csv')
    labels = df['label'].astype(int).to_list()
    return labels


class KNN:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels
    
    def classify_image(self, test_image, no_neigh, distance):
        if distance == 'l1': #manhattan distance
           distances = np.sum((np.abs(self.train_images - test_image)), axis=1)
           
        elif distance == 'l2': # euclidean distance
            distances = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis=1))
        
        sorted_indexes = np.argsort(distances)[:no_neigh] # sort and take the closest no_neigh neighbors
        closest_labels = self.train_labels[sorted_indexes]
        
        #counting frequency for each label
        histogram =  np.bincount(closest_labels.astype(int))

        # returning label with maximum frequency
        return np.argmax(histogram)
    
    def classify_images(self, test_images, no_neigh, distance):

        num_predictions = test_images.shape[0] 
        predictions = np.zeros(num_predictions)
    
        for i in range(num_predictions): # making predictions for each image
            predictions[i] = self.classify_image(test_images[i], no_neigh, distance)
    
        return predictions
    
def accuracy(labels, predicted): #measuring accuracy: no of right answers / total no of answers
        ct = 0
        no_labels = len(labels)
        for i in range(no_labels):
                ct += (labels[i] == predicted[i])
        return 1.0 * ct / no_labels

def plottingAccuracy(no_neigh, acc, distance):
    plt.figure(figsize=(8, 5))
    plt.plot(no_neigh, acc, marker='o', linestyle='-', color='blue', label='Acuratețe')

    for x, y in zip(no_neigh, acc):
        plt.text(x, y + 0.002, f"{y:.2f}", ha='center', fontsize=9)

    plt.title('Metrica ' + distance)
    plt.xlabel('Număr de vecini (k)')
    plt.ylabel('Acuratețe')
    plt.xticks(no_neigh)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
        
# loading images and labels
train_images = read_images("train")
validation_images = read_images("validation")
test_images = read_images("test")
train_labels = read_labels("train")
validation_labels = read_labels("validation")

# converting them to np arrays
train_images = np.array(train_images)
validation_images = np.array(validation_images)
test_images = np.array(test_images)
train_labels = np.array(train_labels)
validation_labels = np.array(validation_labels)

classifier = KNN(train_images, train_labels)
vecini = [x for x in range(3, 50, 2)]

#accuracy for l1 distance 
acc1 = []
for v in vecini:
    predicted_labels = classifier.classify_images(validation_images, no_neigh=v, distance='l1')
    acc = accuracy(validation_labels, predicted_labels)
    print(f"Acuratetea pentru {v} vecini este {acc}")
    acc1.append(acc)

# plotting the results
plottingAccuracy(vecini, acc1, "l1")

#accuracy for l2 distance
acc2 = []
for v in vecini:
    predicted_labels = classifier.classify_images(validation_images, no_neigh=v, distance='l2')
    acc = accuracy(validation_labels, predicted_labels)
    print(f"Acuratetea pentru {v} vecini este {acc}")
    acc2.append(acc)

# plotting the results
plottingAccuracy(vecini, acc2, "l2")


#predicting result on validation set
predicted_labels = classifier.classify_images(validation_images, no_neigh=7, distance='l1') # with best parameters found
#displaying confusion matrix
cm = confusion_matrix(validation_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d') 
plt.title("Matricea de confuzie")
plt.xlabel("Etichete prezise")
plt.ylabel("Etichete reale")
plt.tight_layout()
plt.show()

#predicting labels for test dataset
predictions_test = classifier.classify_images(test_images, no_neigh=7, distance="l1")
#saving them to corresponding file
df_test = pd.read_csv('test.csv')                   
df_test['label'] = predictions_test                   

df_test[['image_id', 'label']].to_csv('predictionsKNN.csv', index=False)
