import cv2
import numpy as np
from skimage import exposure
from skimage import feature

def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

def transpose(vec):
    ret_vec = []
    for i in vec[0]:
        ret_vec.append([i])
    return np.stack(ret_vec)

woman="woman.jpeg"
car="car.jpeg"
oryg = cv2.imread(car, cv2.IMREAD_REDUCED_COLOR_8)
img = cv2.imread(car, cv2.IMREAD_REDUCED_COLOR_8)
hog = cv2.HOGDescriptor()
svm=cv2.ml.SVM_load("cars.xml")
print(len(cv2.ml_SVM.getSupportVectors(svm)[0]))
print(len(cv2.HOGDescriptor_getDefaultPeopleDetector()))


hog.setSVMDetector(transpose(cv2.ml_SVM.getSupportVectors(svm)))
found, w = hog.detectMultiScale(img)
found_filtered = []
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        if ri != qi and is_inside(r, q):
            break
        else:
            found_filtered.append(r)
for person in found:
    draw_person(img, person)
cv2.imshow("people detection", img)
(H, hogImage) = feature.hog(oryg, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True,
                            block_norm="L1", visualize=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")

cv2.imshow("HOG Image", hogImage)

print(w)
cv2.waitKey(0)
cv2.destroyAllWindows()
