import numpy as np
import cv2
from matplotlib import pyplot as plt

"---Question 1---"
"""
Create and train the PCA model 
"""
# create dataset
face_vector = []
counter = 1
for i in range(6):
    path = "Faces/Training/face" + str(counter) + ".jpg"
    face_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2GRAY)
    counter += 1
    face_image = face_image.reshape(50625)
    face_vector.append(face_image)
# compute face vectors
face_vector = np.asarray(face_vector)
face_vector = face_vector.transpose()
print("Face vectors: \n", face_vector)

# noramlize face vectors
avg_face_vector = face_vector.mean(axis=1)
avg_face_vector = avg_face_vector.reshape(face_vector.shape[0], 1)
normalized_face_vector = face_vector - avg_face_vector
print("Normalized face vectors:\n", normalized_face_vector)

# calculate covariance matrix
covariance_matrix = np.cov(np.transpose(normalized_face_vector))

# Calculate the Eigen Values and Eigen Vectors
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# Select K-Best Eigen Vectors
eigen_vectors = cv2.sort(eigen_vectors, 2)
k_eigen_vectors = eigen_vectors[0:10, :]
print("------Eigen-vectors------")
print(eigen_vectors)
print("------Eigen-values------")
print(eigen_values)
print("------K-Eigen-vectors------")
print(k_eigen_vectors)

#  Convert Lower Dimensional K Eigen Vectors to Original Dimensional
eigen_faces = k_eigen_vectors.dot(normalized_face_vector.T)
print("------Eigen-faces------")
print(eigen_faces)

# Represent Each Eigen Face as combination of K-Eigen Vectors
weights = normalized_face_vector.T.dot(eigen_faces.T)

"""
Test the model
"""
print("---------------------Testing------------------------------")


def is_face(img):

    # convert image into face vector
    img = img.reshape(50625, 1)
    # print("Face vectors : \n", img)

    # normalize face vector
    # this allows to eliminate all the features that are common to all images
    normalized_face_vector = img - avg_face_vector
    # print("Normalized face vectors : \n", normalized_face_vector)

    # projection of the vector onto an eigenspace
    weight = normalized_face_vector.T.dot(eigen_faces.T)
    # print("Weights : \n", weight)

    # compute index with minimum distance
    distances = np.linalg.norm(weight - weights, axis=1)
    print("Distances to dataset : \n", distances)
    index = np.argmin(np.linalg.norm(weight - weights, axis=1))
    # print("Index of closest match : ", index)

    if distances.min() < 5*10**7:
        return print(True, "\n")
    else:
        return print(False, "\n")


"---Question 2---"
# Test with 3 faces in different Lighting
print("---Testing light---")
light_face1 = cv2.cvtColor(cv2.imread("Faces/Testing/face4.jpg"), cv2.COLOR_RGB2GRAY)
light_face2 = cv2.cvtColor(cv2.imread("Faces/Testing/face6.jpg"), cv2.COLOR_RGB2GRAY)
light_face3 = cv2.cvtColor(cv2.imread("Faces/Testing/face7.jpg"), cv2.COLOR_RGB2GRAY)
is_face(light_face1)
is_face(light_face2)
is_face(light_face3)

"---Question 3---"
# Test with 2 different skin tones
print("---Testing skin tone---")
tone_face1 = cv2.cvtColor(cv2.imread("Faces/Testing/face1.jpg"), cv2.COLOR_RGB2GRAY)
tone_face2 = cv2.cvtColor(cv2.imread("Faces/Testing/face4.jpg"), cv2.COLOR_RGB2GRAY)
is_face(tone_face1)
is_face(tone_face2)

"---Question 4---"
# Test with rotated images
print("---Testing rotation---")
rotated_face1 = cv2.cvtColor(cv2.imread("Faces/Testing/face2.jpg"), cv2.COLOR_RGB2GRAY)
rotated_face2 = cv2.cvtColor(cv2.imread("Faces/Testing/face3.jpg"), cv2.COLOR_RGB2GRAY)
is_face(rotated_face1)
is_face(rotated_face2)

"---Question 5---"
# test with similar object to faces
print("---Testing with objects---")
cookie_face1 = cv2.cvtColor(cv2.imread("Faces/Testing/face8.jpg"), cv2.COLOR_RGB2GRAY)
skulls_face2 = cv2.cvtColor(cv2.imread("Faces/Testing/face9.jpg"), cv2.COLOR_RGB2GRAY)
christmas_face3 = cv2.cvtColor(cv2.imread("Faces/Testing/face10.jpg"), cv2.COLOR_RGB2GRAY)
is_face(cookie_face1)
is_face(skulls_face2)
is_face(christmas_face3)

"---Question 6---"
print("---Fail testing---")
fail_face1 = cv2.cvtColor(cv2.imread("Faces/Testing/face11.jpg"), cv2.COLOR_RGB2GRAY)
fail_face2 = cv2.cvtColor(cv2.imread("Faces/Testing/face12.jpg"), cv2.COLOR_RGB2GRAY)
fail_face3 = cv2.cvtColor(cv2.imread("Faces/Testing/face13.jpg"), cv2.COLOR_RGB2GRAY)
is_face(fail_face1)
is_face(fail_face2)
is_face(fail_face3)
