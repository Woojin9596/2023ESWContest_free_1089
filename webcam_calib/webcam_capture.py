import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img

capture = cv2.VideoCapture(1)  # 외부 카메라 연결

if capture.isOpened() == False:
    print("Camera open failed")
    exit()

capNum = 0

while True:  # 무한 반복
    ret, frame = capture.read()  # 카메라 영상 받기

    if not ret:
        print("Can't read camera")
        break

    cv2.imshow("ex01", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):  # c를 누르면 화면 캡쳐 후 파일경로에 저장
        img_captured_path = './capture_/captured_%d.png' % capNum
        cv2.imwrite(img_captured_path, frame)
        img_test = img.imread(img_captured_path)
        plt.imshow(img_test)
        plt.show()
        capNum += 1  # 캡쳐시마다 번호증가

    elif key == ord('q'):  # q를 누르면 while문 탈출
        break

capture.release()
cv2.destroyAllWindows()