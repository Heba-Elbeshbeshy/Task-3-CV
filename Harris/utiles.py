import matplotlib.pyplot as plt
import numpy as np


def gradient_x(imggray , k_size):

    if k_size == 5:
        kernel_x = np.array([
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2],
            [-4, -2, 0, 2, 4],
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2]])

    elif k_size == 3:
        kernel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]])

    return Convolution( imggray, kernel_x )

def gradient_y(imggray, k_size):
    if k_size == 5:
        kernel_y = np.array([
            [-2, -2, -4, -2, -2],
            [-1, -1, -2, -1, -1],
            [ 0,  0,  0,  0,  0],
            [ 1,  1,  2,  1,  1],
            [ 2,  2,  4,  2,  2]])

    elif k_size == 3:
        kernel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]])

    return Convolution( imggray, kernel_y )


def gaussian_filter(img, m, n, sigma):
    gaussian = np.zeros((m, n))
    m = m//2
    n = n//2
    for i in range(-m, m+1):
        for j in range(-n, n+1):
            x1 = sigma*(2*np.pi)**2
            x2 = np.exp(-(i**2+j**2)/(2*sigma**2))
            gaussian[i+m, j+n] = (1/x1)*x2
    
    return Convolution( img, gaussian )


def Calculate_Size_After_Applying_Kernel( img_size: int, kernel_size: int) -> int:
    num_pixels = 0

    for i in range(img_size+1):
        added = i + kernel_size
        if added <= img_size:
            num_pixels += 1

    return num_pixels


def Convolution( img: np.array, kernel: np.array) -> np.array:

    Image_size = Calculate_Size_After_Applying_Kernel(
        img_size=img.shape[0],
        kernel_size=kernel.shape[0])

    k = kernel.shape[0]

    convolved_img = np.zeros(shape=(Image_size, Image_size))

    for i in range(Image_size):
        for j in range(Image_size):
            mat = img[i:i+k, j:j+k]
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))

    return convolved_img


def plot_image( img ):
    # plt.figure( figsize = ( 6, 6 ) )
    plt.imshow( img, cmap='gray'  )
    plt.axis( 'off' )
    plt.show( )


def from_RGB_to_GS( image ):
    R, G, B = image[ :, :, 0], image[ :, :, 1], image[ :, :, 2]
    imgGray = ( 0.2989 * R + 0.5870 * G + 0.1140 * B ).astype( np.uint8 )
    return imgGray


# height, width = source.shape
# harris_response = []
# offset = int(window_size / 2)

# # Loop over each column in the image
# for y in range(offset, height - offset):
#     # Loop over each row in the image
#     for x in range(offset, width - offset):
#         Sxx = np.sum(Ixx[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
#         Syy = np.sum(Iyy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
#         Sxy = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])

#         # Find determinant and trace, use to get corner response
#         det = (Sxx * Syy) - (Sxy ** 2)
#         trace = Sxx + Syy
#         r = det - k * (trace ** 2)

#         harris_response.append(r)

# Convert response from list to numpy array
# new_width = source.shape[0] - (window_size - offset)
# new_height = source.shape[1] - (window_size - offset)
# harris_response = np.array(harris_response).reshape((new_width, new_height))



    # for rowindex, response in enumerate(harris_matrix ):
#     for colindex, r in enumerate(response):
#         if r > (max_corner_response * threshold):
#             # this is a corner
#             img_copy_for_corners[rowindex, colindex] = [255,0,0]
#         elif r < 0:
#             # this is an edge
#             img_copy_for_edges[rowindex, colindex] = [0,255,0]
#         elif r == 0:
#             # this is flat
#             img_copy_for_flat[rowindex, colindex] = [0,0,255]


# return  img_copy_for_corners, img_copy_for_edges, img_copy_for_flat 

# Dilate the points to be more clear
# harris_matrix = cv2.dilate(harris_matrix, None)

# img_corners = map_indices_to_image( img, corners )

# plot_image(img_corners)

# def map_indices_to_image(source: np.ndarray, indices: np.ndarray):

#     src = np.copy(source)

#     # Make sure that the original source shape == indices shape
#     src = src[:indices.shape[0], :indices.shape[1]]

#     # Mark each index with dot
#     src[indices == 1] = [255,0,0]

#     return src