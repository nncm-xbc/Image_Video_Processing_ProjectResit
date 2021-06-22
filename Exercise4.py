import numpy as np
import cv2

"""
Create and train PCA model 

"""

# create dataset
face_vector = []
counter = 1
for i in range(6):
    path = "Faces/face" + str(counter) + ".jpg"
    face_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2GRAY)
    counter += 1
    face_image = face_image.reshape(50625)
    face_vector.append(face_image)
face_vector = np.asarray(face_vector)
face_vector = face_vector.transpose()

# noramlize face vectors
avg_face_vector = face_vector.mean(axis=1)
avg_face_vector = avg_face_vector.reshape(face_vector.shape[0], 1)
normalized_face_vector = face_vector - avg_face_vector

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

#  Convert Lower Dimensional K Eigen Vectors to Original Dimensional
eigen_faces = k_eigen_vectors.dot(normalized_face_vector.T)
print("------Eigen-faces------")
print(eigen_faces)

# Represent Each Eigen Face as combination of the K-Eigen Vectors
weights = normalized_face_vector.T.dot(eigen_faces.T)

"""
Test the model
"""
print("---------------------------------------------------")
test_img = cv2.cvtColor(cv2.imread("Faces/test_face.jpg"), cv2.COLOR_RGB2GRAY)

# convert image into face vector
test_img = test_img.reshape(50625, 1)
print("-----face vector------")
print(test_img)

# normalize face vector
# this allows to eliminate all the features that are common to all images
test_normalized_face_vector = test_img - avg_face_vector
print("-----Normalized vector------")
print(test_normalized_face_vector)

# projection of the vector onto an eigenspace
test_weight = test_normalized_face_vector.T.dot(eigen_faces.T)
print("-----Weights------")
print(test_weight)

# compute index with minimum distance
distances = np.linalg.norm(test_weight - weights, axis=1)
print("-----Distance to training data set instances------")
print(distances)
index = np.argmin(np.linalg.norm(test_weight - weights, axis=1))
print("-----Index of closest match------")
print(index)
