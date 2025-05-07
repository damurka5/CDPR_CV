import cv2
import cv2.aruco as aruco

# Create a dictionary of 6x6 markers with 250 possible IDs
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Generate a marker with ID 23, size 200x200 pixels
id = 3
marker_image = aruco.generateImageMarker(dictionary, id=id, sidePixels=200, borderBits=1)

# Save the marker image
cv2.imwrite(f"cv/marke{id}.png", marker_image)

# Optionally display the marker
# cv2.imshow("Marker 23", marker_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()