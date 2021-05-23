import numpy as np
import cv2
import matplotlib.pyplot as plt


is_drawing = False
point_x, point_y = None, None


def show(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def draw(ret_size, dp_size=1024, thickness=30, title='Drawing Window'):
    def callback(event, x, y, flags, param):
        global is_drawing, point_x, point_y

        if event == cv2.EVENT_LBUTTONDOWN:
            is_drawing = True
            point_x, point_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_drawing:
                cv2.line(img, (point_x, point_y), (x, y), color=(255, 255, 255), thickness=thickness)
                point_x, point_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            is_drawing = False
            cv2.line(img, (point_x, point_y), (x, y), color=(255, 255, 255), thickness=thickness)

    img = np.zeros((dp_size, dp_size, 3), np.uint8)
    cv2.namedWindow(title)
    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(title, callback)

    while True:
        cv2.imshow(title, img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    img = cv2.resize(img, dsize=(ret_size, ret_size), interpolation=cv2.INTER_AREA).sum(axis=2)
    return img


if __name__ == '__main__':
    show(draw(512))
