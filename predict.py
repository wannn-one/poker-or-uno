import cv2
import numpy as np
import copy

from utils.train_utils import LoadModel
from utils.constants import *

def drawRectangleForCardDetectionFrame(frame):
    # frame size 1280 x 720
    cv2.rectangle(img=frame, pt1=(x_player, y), pt2=(x_player + distance_x, y + distance_y), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.rectangle(img=frame, pt1=(x_deck, y), pt2=(x_deck + distance_x, y + distance_y), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.rectangle(img=frame, pt1=(x_opponent, y), pt2=(x_opponent + distance_x, y + distance_y), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    

def draw_text(img, s_text, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = pos
    font_scale = 0.7
    font_color = (0, 0, 255)
    thickness = 2
    line_type = 2
    cv2.putText(img, s_text, position, font, font_scale, font_color, thickness, line_type)
    return copy.deepcopy(img)

def process_card_image(card, model, class_names, top_left):
    card = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)

    # feed card to model
    arr = []

    img = cv2.resize(card, (128, 128))
    img = np.asarray(img) / 255
    img = img.astype('float32')
    arr.append(img)

    arr = np.array(arr)
    arr = arr.astype('float32')

    hs = model.predict(arr)
    n = np.max(np.where(hs == hs.max()))

    text_coord = top_left + [0, -10]
    return f'{class_names[n]} {"{:.2f}".format(hs[0, n])}', text_coord - [95, 0]

def main():
    cap = cv2.VideoCapture(0)
    ip = 'http://192.168.0.101:8080/video'
    cap.open(ip)

    class_names = [
        "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "10C", "JC", "QC", "KC", "AC",
        "2H", "3H", "4H", "5H", "6H", "7H", "8H", "9H", "10H", "JH", "QH", "KH", "AH",
        "2S", "3S", "4S", "5S", "6S", "7S", "8S", "9S", "10S", "JS", "QS", "KS", "AS",
        "2D", "3D", "4D", "5D", "6D", "7D", "8D", "9D", "10D", "JD", "QD", "KD", "AD",
        "Joker"
    ]

    model = LoadModel("trained/trained-card-epoch-10.h5")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        drawRectangleForCardDetectionFrame(frame)

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img_thres = cv2.adaptiveThreshold(img_gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          thresholdType=cv2.THRESH_BINARY, blockSize=101, C=10)

        total_labels, _, values, _ = cv2.connectedComponentsWithStats(img_thres, connectivity=4, ltype=cv2.CV_32S)

        for i in range(1, total_labels):  # Start from 1 to exclude background label (0)
            width, height = values[i, 2:4]

            if 100 < width < 900 and 300 < height < 1000:
                top_left = values[i, 0:2]
                bottom_right = top_left + values[i, 2:4]

                frame = cv2.rectangle(img=frame, pt1=tuple(top_left), pt2=tuple(bottom_right), color=(0, 0, 0), thickness=2)

                # Check if the card is within one of the specified rectangles
                player_rect = (x_player, x_player + distance_x, y, y + distance_y)
                deck_rect = (x_deck, x_deck + distance_x, y, y + distance_y)
                opponent_rect = (x_opponent, x_opponent + distance_x, y, y + distance_y)

                if any(rect[0] <= top_left[0] <= rect[1] and rect[2] <= top_left[1] <= rect[3] for rect in [player_rect, deck_rect, opponent_rect]):
                    card = img_thres[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                    cv2.imshow("card", card)
                    text, text_coord = process_card_image(card, model, class_names, top_left)
                    draw_text(frame, text, text_coord)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
