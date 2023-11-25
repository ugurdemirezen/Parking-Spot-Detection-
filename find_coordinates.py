import cv2
import numpy as np

cap=cv2.VideoCapture("easy.mp4")
i=0
coordinates = []
def draw(event, x, y, flags, param):
    global coordinates

    if event == cv2.EVENT_LBUTTONDOWN:

        coordinates.append([x, y])
        if len(coordinates) == 4:
            with open("coordinates.txt", "a") as file:
                file.write("np.array("+f"{coordinates})\n")


            # Koordinat listesini sıfırla
            coordinates = []


while True:
    ret, frame=cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        continue
    winname = "video"
    cv2.imshow(winname,frame)
    cv2.setMouseCallback(winname, draw)
    key=cv2.waitKey(1) & 0xFF


cap.release()
cv2.destroyAllWindows()