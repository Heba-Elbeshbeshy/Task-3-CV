import matplotlib.image as mpimg
from harris import * 

img = mpimg.imread( 'Harris/Images/chess.jpeg' ) 
# img = mpimg.imread( 'Harris/Images/download.jpeg' )

imggray = from_RGB_to_GS( img )

# apply Harris Corner Detection
# k : Sensitivity factor chess k = 0.1 - 0.04
harris_response = f_harris( imggray, k = 0.1)

# categorize Harris response Edge, Corner, Flat chess threshold = 0.5 - 0.1
corners, edges = categorize_harris_response( img, harris_response , threshold = 0.5)

plot_image(corners)