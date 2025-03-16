import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def using_cv2():

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        isOpen, frame = cap.read()

        if not isOpen:
            print("Couldn't receive frame...")
            break
        
        cv2.imshow('Macbook camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()


def using_matplot():
    cap = cv2.VideoCapture(0)
    plt.ion()  # Interactive mode on

    fig, ax = plt.subplots()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        plt.pause(0.001)
        ax.clear()

    cap.release()
    plt.ioff()
    plt.close()
    


if __name__ == "__main__":
    print("Starting camera...")
    # using_cv2()
    using_matplot()
    print("Ending camera...")
