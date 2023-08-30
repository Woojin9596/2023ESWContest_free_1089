import os
import cv2
import matplotlib.pyplot as plt


def show_images(img1, img2, location_info):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.set_title('RGB')
    ax1.axis("off")
    ax2.imshow(img2, cmap='gray')
    ax2.set_title('Gray Scale')
    ax2.axis("off")
    fig.text(0.5, 0.92, "POTHOLE DETECTED", size=12, ha="center")
    fig.text(0.5, 0.86, location_info, size=8, ha="center")
    plt.subplots_adjust(hspace=0.5)


def save_captured_image(frame, capNum):
    img_captured_path = './capture_/captured_%d.png' % capNum
    cv2.imwrite(img_captured_path, frame)


def process_captured_image(img_captured_path, capNum):
    img1 = cv2.imread(img_captured_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_captured_path, cv2.IMREAD_GRAYSCALE)

    location_info = "Location: (123.456, 789.012)\nLat: 37.1234, Long: -122.5678"
    show_images(img1, img2, location_info)


def main():
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    save_path = os.path.join(desktop_path, "result.jpg")

    capture = cv2.VideoCapture(1)

    if capture.isOpened() == False:
        print("Camera open failed")
        exit()

    capNum = 0

    while True:
        ret, frame = capture.read()

        if not ret:
            print("Can't read camera")
            break

        cv2.imshow("ex01", frame)

        key = cv2.waitKey(1)

        if key == ord('c'):
            save_captured_image(frame, capNum)
            process_captured_image('./capture_/captured_%d.png' % capNum, capNum)
            capNum += 1

        elif key == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()