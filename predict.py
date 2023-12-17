import cv2
import numpy as np
import time

from utils.train_utils import LoadModel
from utils.constants import *

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('5024211048_Ikhwan.mp4', fourcc, 20.0, (1280, 720))

def drawRectangleForCardDetectionFrame(frame):
    # frame size 1280 x 720
    # player area
    cv2.rectangle(img=frame, pt1=(x_player, y), pt2=(x_player + distance_x, y + distance_y), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(img=frame, text="Player", org=(x_player, y - 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(25, 25, 25), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(img=frame, text="Player", org=(x_player, y - 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # opponent area
    cv2.rectangle(img=frame, pt1=(x_opponent, y), pt2=(x_opponent + distance_x, y + distance_y), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(img=frame, text="Opponent", org=(x_opponent, y - 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(25, 25, 25), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(img=frame, text="Opponent", org=(x_opponent, y - 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)    

def isCardValid(card, last_card):
    if last_card is None:
        return True  # Any card can be played as the first card

    # Extract rank and suit from the card names
    rank_card, suit_card = card[:-1], card[-1]
    rank_last, suit_last = last_card[:-1], last_card[-1]

    # Check if the ranks or suits match
    return rank_card == rank_last or suit_card == suit_last

def drawText(img, s_text, pos):
    font = cv2.FONT_HERSHEY_PLAIN
    position = pos
    font_scale = 2
    font_color = (0, 0, 255)
    thickness = 2
    line_type = 2
    return cv2.putText(img, s_text, position, font, font_scale, font_color, thickness, line_type)

def predictCard(card, model, class_names, top_left):
    card = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)

    # feed card to model
    arr = []

    img = cv2.resize(card, (128, 128)) # resize to 128x128
    img = np.asarray(img) / 255 # normalize
    img = img.astype('float32') # convert to float32
    arr.append(img) # add to array

    arr = np.array(arr) # convert to numpy array
    arr = arr.astype('float32') # convert to float32

    hs = model.predict(arr) # predict

    # get the highest score
    v = hs[0, :]
    idx = np.max(np.where(v == v.max()))

    # get the class name
    text = class_names[idx]

    # get the position
    x, y = top_left
    text_coord = (x, y - 10)

    return text, text_coord

def waitForKeyPress(timeout_in_ms=0):
    start_time = time.time()
    key = -1

    while True:
        elapsed_time = int((time.time() - start_time) * 1000)
        key = cv2.waitKey(1)

        if key != -1 or (timeout_in_ms > 0 and elapsed_time >= timeout_in_ms):
            break

    return key

def main():
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    player_turn = True
    last_card = None
    game_over = False
    player_cards = 5
    computer_cards = 5
    total_deck_cards = 52 - player_cards - computer_cards

    model = LoadModel("trained/trained-card-epoch-10.h5")

    print(f"Player has {player_cards} cards left, computer has {computer_cards} cards left, {total_deck_cards} cards left in deck")

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

        for i in range(1, total_labels):
            width, height = values[i, 2:4]

            if 200 < width < 900 and 400 < height < 1000:
                top_left = values[i, 0:2]
                bottom_right = top_left + values[i, 2:4]

                frame = cv2.rectangle(img=frame, pt1=tuple(top_left), pt2=tuple(bottom_right), color=(0, 0, 0), thickness=2)

                player_rect = (x_player, x_player + distance_x, y, y + distance_y)
                opponent_rect = (x_opponent, x_opponent + distance_x, y, y + distance_y)

                # print(f"Player has {player_cards} cards left, computer has {computer_cards} cards left, {total_deck_cards} cards left in deck")

                if player_turn and player_cards > 0 and player_rect[0] <= top_left[0] <= player_rect[1] and player_rect[2] <= top_left[1] <= player_rect[3]:
                    card = img_thres[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                    text, text_coord = predictCard(card, model, class_names, top_left)

                    drawText(frame, text, text_coord)

                    if isCardValid(text, last_card):
                        print(f"Player played {text}")
                        print(f"Are you sure you want to play {text} (y/n)?")

                        key = waitForKeyPress(1000)

                        if key == ord('y'):
                            print(f"Player played {text}")
                            last_card = text
                            player_cards -= 1
                            player_turn = not player_turn
                            if player_cards == 0:
                                break
                            else:
                                print(f"Player has {player_cards} cards left, computer turn")
                        elif key == ord('n'):
                            print(f"Player didn't play {text}, changing card...")
                        else:
                            print(f"Invalid input")
                    else:
                        print(f"Invalid card, should be {last_card} or {last_card[:-1]} but was {text}")
                        print(f"Do you want to draw a card? (y/n)")

                        key = waitForKeyPress(1000)

                        if key == ord('y'):
                            print(f"Player drew a card")
                            player_cards += 1
                            total_deck_cards -= 1
                            player_turn = not player_turn
                            print(f"Player has {player_cards} cards left, computer turn")
                        elif key == ord('n'):
                            print(f"Player didn't draw a card, changing card...")
                        else:
                            print(f"Invalid input")

                elif not player_turn and computer_cards > 0 and opponent_rect[0] <= top_left[0] <= opponent_rect[1] and opponent_rect[2] <= top_left[1] <= opponent_rect[3]:
                    card = img_thres[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                    text, text_coord = predictCard(card, model, class_names, top_left)

                    drawText(frame, text, text_coord)

                    if isCardValid(text, last_card):
                        print(f"Computer played {text}")
                        print(f"Are you sure you want to play {text} (y/n)?")

                        key = waitForKeyPress(1000)

                        if key == ord('y'):
                            print(f"Computer played {text}")
                            last_card = text
                            computer_cards -= 1
                            player_turn = not player_turn
                            if computer_cards == 0:
                                break
                            else:
                                print(f"Computer has {computer_cards} cards left, player turn")
                        elif key == ord('n'):
                            print(f"Computer didn't play {text}, changing card...")
                        else:
                            print(f"Invalid input")
                    else:
                        print(f"Invalid card, should be {last_card} or {last_card[:-1]} but was {text}")
                        print(f"Do you want to draw a card? (y/n)")

                        key = waitForKeyPress(1000)

                        if key == ord('y'):
                            print(f"Computer drew a card")
                            computer_cards += 1
                            total_deck_cards -= 1
                            player_turn = not player_turn
                            print(f"Computer has {computer_cards} cards left, player turn")
                        elif key == ord('n'):
                            print(f"Computer didn't draw a card")
                        else:
                            print(f"Invalid input")

        if player_cards == 0:
            print("Player won!")
            game_over = True
        elif computer_cards == 0:
            print("Computer won!")
            game_over = True

        out.write(frame)
        
        cv2.imshow("Poker or uno", frame)

        if cv2.waitKey(1) == ord('q') or game_over:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
