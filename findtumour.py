import cv2

# Mouse callback function
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse click
        b, g, r = img[y, x]  # note: OpenCV uses BGR order
        print(f"Coordinates: x={x}, y={y} | Color (BGR): {b}, {g}, {r}")

# Load your JPG image
img = cv2.imread("/Users/ernaeliasdottir/Desktop/UoM 2024-2025/Önn 3/Research Project/Finale/FixNoAugjustgenmask_noaugtrain_predictions_3d_output_25epochs/flippedright_100019_1_C_edited/slice_118.jpg")

cv2.imshow("image", img)
cv2.setMouseCallback("image", get_coordinates)

cv2.waitKey(0)
cv2.destroyAllWindows()
