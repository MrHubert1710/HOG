import cv2
import numpy as np

hog = cv2.HOGDescriptor()

def hoggify(a,b):

    data=[]
    labels=[]
    for i in range(a,int(b)):
        image = cv2.imread("./TrainImages/pos-"+str(i)+".pgm", 0)
        dim = 128
        img = cv2.resize(image, (dim,dim), interpolation = cv2.INTER_AREA)
        img = hog.compute(img)
        img = np.squeeze(img)
        data.append(img)
        labels.append(np.int_(1))

        image = cv2.imread("./TrainImages/neg-"+str(i)+".pgm", 0)
        dim = 128
        img = cv2.resize(image, (dim,dim), interpolation = cv2.INTER_AREA)
        img = hog.compute(img)
        img = np.squeeze(img)
        data.append(img)
        labels.append(np.int_(-1))

    return data, labels

def svmTrain(data,labels):
    svm=cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_NU_SVR)
    svm.setNu(0.5)
    svm.setC(0.01)
    svm.setP(0.5)
    svm.setDegree(0.1);
    svm.trainAuto(data, cv2.ml.ROW_SAMPLE,labels)
    return svm

def list_to_matrix(lst):
    return np.stack(lst)

print(hog.getDescriptorSize())
lst, labels = hoggify(0,100)
data=list_to_matrix(lst)
#print(data)
labels=np.int_(labels)
model=svmTrain(data,labels)
model.save("cars.xml")

lst, labels = hoggify(100,110)
data=list_to_matrix(lst)
data=np.float32(data)
lab_out=model.predict(data)

print(lab_out)