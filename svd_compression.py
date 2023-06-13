import numpy as np
from scipy import datasets
from PIL import Image


def compression_rate(original_image, compressed_image):
    original_image_array = np.array(original_image)
    compressed_image_array = np.array(compressed_image)
    return (original_image_array.size * original_image_array.itemsize) / (compressed_image_array.size
                                                                          * compressed_image_array.itemsize)


def approximation_error(original_image, compressed_image):
    original_image_array = np.array(original_image)
    compressed_image_array = np.array(compressed_image)
    return np.linalg.norm(original_image_array - compressed_image_array, ord='fro')


def rank_k_approx(matrix, k):
    u, s, vh = np.linalg.svd(matrix)
    rank = np.linalg.matrix_rank(matrix)

    if k >= rank:
        raise IndexError

    s_k = np.zeros(rank)
    s_k[:k] = s[:k]
    return u.dot(np.diag(s_k)).dot(vh)


image = datasets.ascent()
A_5 = rank_k_approx(image, 5)
A_20 = rank_k_approx(image, 20)
A_75 = rank_k_approx(image, 75)

compressed_image_k5 = Image.fromarray(A_5)
compressed_image_k20 = Image.fromarray(A_20)
compressed_image_k75 = Image.fromarray(A_75)

print("Kompressionsrate k = 5:", compression_rate(image, compressed_image_k5))
print("Kompressionsrate k = 50:", compression_rate(image, compressed_image_k20))
print("Kompressionsrate k = 75:", compression_rate(image, compressed_image_k75))

print("Approximationsfehler k = 5:", approximation_error(image, compressed_image_k5))
print("Approximationsfehler k = 20:", approximation_error(image, compressed_image_k20))
print("Approximationsfehler k = 75:", approximation_error(image, compressed_image_k75))
