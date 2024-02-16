import cv2
import numpy as np
import os
import Card

""" INIT """
FILEPATH = os.getcwd() + '/Dataset'
cap = cv2.VideoCapture(1)

# Cam Display Dimension
WIDTH  = 960 
HEIGHT = 540

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Load Rank & Suit Data
Card.LoadRank(FILEPATH)
Card.LoadSuit(FILEPATH)

# Flat Card Dimension
CARD_WIDTH  = 200
CARD_HEIGHT = 300

# Rank Section Dimension
RANK_CROP_WIDTH  = 30
RANK_CROP_HEIGHT = 50

# Suit Section Dimension
SUIT_CROP_WIDTH  = 30
SUIT_CROP_HEIGHT = 100

# Rank Data Dimension
RANK_WIDTH  = 70
RANK_HEIGHT = 125

# Suit Data Dimension
SUIT_WIDTH  = 70
SUIT_HEIGHT = 100

cards_detected = []
ranks_detected = []
suits_detected = []

dealer_cards   = []
player_cards   = []

D_index        = 0
P_index        = 0

game_phase     = 0

dealer_score   = 0
player_score   = 0

flag_deal      = True

delay          = 0

""" Functions """
def cardValue(card):
    value = 0
    if card in ['Ace', 'Ace of Clubs', 'Ace of Diamonds', 'Ace of Hearts', 'Ace of Spades']:
        value = 1
    elif card in ['2', '2 of Clubs', '2 of Diamonds', '2 of Hearts', '2 of Spades']:
        value = 2
    elif card in ['3', '3 of Clubs', '3 of Diamonds', '3 of Hearts', '3 of Spades']:
        value = 3
    elif card in ['4', '4 of Clubs', '4 of Diamonds', '4 of Hearts', '4 of Spades']:
        value = 4
    elif card in ['5', '5 of Clubs', '5 of Diamonds', '5 of Hearts', '5 of Spades']:
        value = 5
    elif card in ['6', '6 of Clubs', '6 of Diamonds', '6 of Hearts', '6 of Spades']:
        value = 6
    elif card in ['7', '7 of Clubs', '7 of Diamonds', '7 of Hearts', '7 of Spades']:
        value = 7
    elif card in ['8', '8 of Clubs', '8 of Diamonds', '8 of Hearts', '8 of Spades']:
        value = 8
    elif card in ['9', '9 of Clubs', '9 of Diamonds', '9 of Hearts', '9 of Spades']:
        value = 9
    elif card in ['10', '10 of Clubs', '10 of Diamonds', '10 of Hearts', '10 of Spades']:
        value = 10
    elif card in ['Jack', 'Jack of Clubs', 'Jack of Diamonds', 'Jack of Hearts', 'Jack of Spades']:
        value = 10
    elif card in ['Queen', 'Queen of Clubs', 'Queen of Diamonds', 'Queen of Hearts', 'Queen of Spades']:
        value = 10
    elif card in ['King', 'King of Clubs', 'King of Diamonds', 'King of Hearts', 'King of Spades']:
        value = 10
    return value


def reorder(points):
    points = points.reshape((4, 2))
    pointsNew = np.zeros((4, 1, 2), dtype= np.int32)
    add = points.sum(1)
    
    pointsNew[0] = points[np.argmin(add)]
    pointsNew[3] = points[np.argmax(add)]
    diff = np.diff(points, axis= 1)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[2] = points[np.argmax(diff)]
    
    return pointsNew


def flattenImage(frame, points):
    if points.size != 0:
        points = reorder(points)
        pts1 = np.float32(points)
        pts2 = np.float32([[0,0], [CARD_WIDTH, 0], [0, CARD_HEIGHT], [CARD_WIDTH, CARD_HEIGHT]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(frame, matrix, (CARD_WIDTH, CARD_HEIGHT))

    return imgWarpColored


def preprocessRank(img, index):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    
    retval, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)
    
    dilate = cv2.dilate(thresh, (5, 5), iterations= 5)
    #cv2.imshow(f"process{index}", thresh)
    
    contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contours = sorted(contours, key= cv2.contourArea, reverse= True)
    
    for contour in max_contours:
        if len(contour) != 0:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            #cv2.imshow(f"bounding_rect{index}", img)
            img = thresh[y:y + h, x:x + w]
            img = cv2.resize(img, (RANK_WIDTH, RANK_HEIGHT))
            #cv2.imshow(f"cropped_to_size{index}", img)

            return img
        else:
            return np.zeros(0)
        

def preprocessSuit(img, index):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    
    retval, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)
    
    dilate = cv2.dilate(thresh, (5, 5), iterations= 5)
    #cv2.imshow(f"process{index}", thresh)
    
    contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contours = sorted(contours, key= cv2.contourArea, reverse= True)
    
    for contour in max_contours:
        if len(contour) != 0:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            #cv2.imshow(f"bounding_rect{index}", img)
            img = thresh[y:y + h, x:x + w]
            img = cv2.resize(img, (SUIT_WIDTH, SUIT_HEIGHT))
            #cv2.imshow(f"cropped_to_size{index}", img)

            return img
        else:
            return np.zeros(0)


def matchResult():
    if dealer_score <= 21:
        if player_score <= 21:
            if dealer_score > player_score:
                txt = "DEALER WINS!!"
            elif dealer_score < player_score:
                txt = "PLAYER WINS!!"
            else:
                txt = "DRAW!!"
            #endif
        else:
            txt = "DEALER WINS!!"
        #endif
    elif dealer_score > 21:
        if player_score <= 21:
            txt = "PLAYER WINS!!"
        else:
            txt = "DRAW!!"
    #endif
    
    return txt
  

def textAdjust(width, txt, scale, thickness):
    txt = txt
    text_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    return text_x


def displayUI(display):
    deck_img = np.zeros((HEIGHT, WIDTH//2, 3), np.uint8)
    deck_img[:] = (51, 102, 0)
    display[0:HEIGHT, 0:WIDTH//2] = deck_img

    display = cv2.line(display, (0, HEIGHT//2), (WIDTH//2, HEIGHT//2), (255, 255, 255), 3)
    display = cv2.line(display, (WIDTH//2, 0), (WIDTH//2, HEIGHT), (255, 255, 255), 3)
    
    # Points text
    cv2.putText(display, f"Dealer's Hand : {dealer_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(display, f"Player's Hand : {player_score}", (50, (HEIGHT//2) + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Hands
    if D_index > 0:
        for d in range (D_index):
            img = dealer_cards[d - 1].shape
            img_height, img_width, _ = img
            x = 50 + (50*d)
            display[100:100 + img_height, x:x + img_width] = dealer_cards[d - 1]
    if P_index > 0:
        for p in range (P_index):
            img = player_cards[p - 1].shape
            img_height, img_width, _ = img
            if p < 6:
                x = 50 + (50*p)
                y = HEIGHT//2 + 100
            else:
                x = 50 + (50*(p - 6))
                y = HEIGHT//2 + 120
            display[y:y + img_height, x:x + img_width] = player_cards[p - 1]
    
    return display


""" Main Program """
while True:
    ret, frame = cap.read()
    
    display = frame.copy()
    display = displayUI(display)
    
    # Processing image for contours
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 4500]
    max_contour = sorted(contours, key= cv2.contourArea, reverse= True)[:2]

    #### Dealer's Cards ####
    if game_phase == 0:
        if (flag_deal):
            txt = "Dealer deal his card..."
            x = textAdjust(WIDTH//2, txt, 0.75, 2)
            cv2.putText(display, txt, (WIDTH//2 + x, HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            
            for index, contour in enumerate(max_contour):
                if index < 2:
                    cv2.drawContours(display, max_contour, -1, (0, 255, 0), 2)
                    perim = cv2.arcLength(contour, True)
                    corners = cv2.approxPolyDP(contour, 0.02 * perim, True)
                    if len(corners) == 4:
                        flatCard = flattenImage(frame, corners)
                        
                        rankSect = flatCard[2:RANK_CROP_HEIGHT, 0:RANK_CROP_WIDTH]
                        rankSect = cv2.resize(rankSect, (0, 0), fx= 4, fy= 4)
                        rankImg = preprocessRank(rankSect, index)
                        #cv2.imshow(f"Rank card{index}", rankSect)
                        
                        suitSect = flatCard[52:SUIT_CROP_HEIGHT, 0:SUIT_CROP_WIDTH]
                        suitSect = cv2.resize(suitSect, (0, 0), fx= 4, fy= 4)
                        suitImg = preprocessSuit(suitSect, index)
                        #cv2.imshow(f"Suit card{index}", suitSect)
                        
                        flatCard = cv2.resize(flatCard, (0, 0), fx= 0.4, fy= 0.4)
                        cards_detected = flatCard
                        #cv2.imshow(f"flatten card{D_index}", flatCard)
                        
                        if rankImg != np.zeros(0) and suitImg != np.zeros(0):
                            rankPredictions = Card.CompareRank(rankImg)
                            suitPredictions = Card.CompareSuit(suitImg)
                            #print(rankPredictions, "of", suitPredictions)
                        
                            # Append Value
                            ranks_detected.insert(0, cardValue(rankPredictions))
                            suits_detected.insert(0, suitPredictions)
                        
                            # Predictions Text
                            x, y, w, h = cv2.boundingRect(contour)
                            txt = f"{rankPredictions} of {suitPredictions}"
                            xt = textAdjust(w, txt, 1, 3)
                            cv2.putText(display, txt, (xt + x, y + (h//2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                            delay = delay + 1
                        #endif
                    #endif
                #endif
            #endfor
            
            if delay > 50:
                dealer_score = dealer_score + ranks_detected[0]
                
                dealer_cards.insert(D_index, cards_detected)
                D_index = D_index + 1
                    
                ranks_detected.clear()
                suits_detected.clear()
                
                flag_deal = False
                
                if player_score != 0:
                    game_phase = 2
            #endif
            
        else:
            txt = "Press c to deal player's card..."
            x = textAdjust(WIDTH//2, txt, 0.75, 2)
            cv2.putText(display, txt, (WIDTH//2 + x, HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        #endif
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            game_phase = 1
            flag_deal = True
            delay = 0
            #endif
        #endif

    #### Player's Cards ####
    elif game_phase == 1:
        if (flag_deal):
            txt = "Dealer deal player's card..."
            x = textAdjust(WIDTH//2, txt, 0.75, 2)
            cv2.putText(display, txt, (WIDTH//2 + x, HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                    
            for index, contour in enumerate(max_contour):
                if index < 2:
                    cv2.drawContours(display, max_contour, -1, (0, 255, 0), 2)
                    perim = cv2.arcLength(contour, True)
                    corners = cv2.approxPolyDP(contour, 0.02 * perim, True)
                    if len(corners) == 4:
                        flatCard = flattenImage(frame, corners)
                        
                        rankSect = flatCard[2:RANK_CROP_HEIGHT, 0:RANK_CROP_WIDTH]
                        rankSect = cv2.resize(rankSect, (0, 0), fx= 4, fy= 4)
                        rankImg = preprocessRank(rankSect, index)
                        #cv2.imshow(f"Rank card{index}", rankSect)
                        
                        suitSect = flatCard[52:SUIT_CROP_HEIGHT, 0:SUIT_CROP_WIDTH]
                        suitSect = cv2.resize(suitSect, (0, 0), fx= 4, fy= 4)
                        suitImg = preprocessSuit(suitSect, index)
                        #cv2.imshow(f"Suit card{index}", suitSect)
                        
                        flatCard = cv2.resize(flatCard, (0, 0), fx= 0.4, fy= 0.4)
                        cards_detected = flatCard
                        #cv2.imshow(f"flatten card{index}", flatCard)
                        
                        if rankImg != np.zeros(0) and suitImg != np.zeros(0):
                            rankPredictions = Card.CompareRank(rankImg)
                            suitPredictions = Card.CompareSuit(suitImg)
                            #print(rankPredictions, "of", suitPredictions)
                        
                            # Append Value
                            ranks_detected.insert(0, cardValue(rankPredictions))
                            suits_detected.insert(0, suitPredictions)
                        
                            # Predictions Text
                            x, y, w, h = cv2.boundingRect(contour)
                            txt = f"{rankPredictions} of {suitPredictions}"
                            xt = textAdjust(w, txt, 1, 3)
                            cv2.putText(display, txt, (xt + x, y + (h//2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        
                            delay = delay + 1
                        #endif
                    #endif
                #endif
            #endfor
            
            if delay > 50:
                player_score = player_score + ranks_detected[0]
            
                player_cards.insert(P_index, cards_detected)
                P_index = P_index + 1
                
                ranks_detected.clear()
                suits_detected.clear()
                
                flag_deal = False
            #endif
            
        else:
            if player_score > 21:
                txt = "Press x to deal Dealer's last card"
                x = textAdjust(WIDTH//2, txt, 0.75, 2)
                cv2.putText(display, txt, (WIDTH//2 + x, HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            else:
                txt = "Hit(c) / Stand(x)"
                x = textAdjust(WIDTH//2, txt, 0.75, 2)
                cv2.putText(display, txt, (WIDTH//2 + x, HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            #endif
        #endif
            
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            game_phase = 1
            flag_deal = True
            delay = 0
        elif k == ord('x'):
            game_phase = 0
            flag_deal = True
            delay = 0
        #endif
    
    ### Winner ###
    elif game_phase == 2:
        txt = matchResult()
        x = textAdjust(WIDTH//2, txt, 2, 3)
        cv2.putText(display, txt, (WIDTH//2 + x, HEIGHT//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('r'):
            dealer_score = 0
            player_score = 0
            
            dealer_cards.clear()
            player_cards.clear()
            
            D_index = 0
            P_index = 0
            
            game_phase = 0
        #endif
    #endif
        
    cv2.imshow('Card Detection', display)
    
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
