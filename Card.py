""" Functions for Card Processing """
import cv2
import numpy as np
import os


suit_Images = []
suit_Names  = []

rank_Images = []
rank_Names  = []



def LoadSuit(filepath):
    suit_path = filepath + '/Suit'

    myList = os.listdir(suit_path)

    # Loading every suit data in rank_path
    for suit in myList:
        imgCur = cv2.imread(f'{suit_path}/{suit}', 0)
        suit_Images.append(imgCur)
        suit_Names.append(os.path.splitext(suit)[0])


def LoadRank(filepath):
    rank_path = filepath + '/Rank'

    myList = os.listdir(rank_path)

    # Loading every rank data in rank_path
    for rank in myList:
        imgCur = cv2.imread(f'{rank_path}/{rank}', 0)
        rank_Images.append(imgCur)
        rank_Names.append(os.path.splitext(rank)[0])


def CompareSuit(picked_img):
    # Image to be compared
    best_rank  = 10000
    best_match = []
    i = 0
    
    # Comparing until match
    for img in suit_Images:
        diff_img = cv2.absdiff(picked_img, img)
        rank_diff = int(np.sum(diff_img)/255)
        if rank_diff < best_rank:
            best_rank = rank_diff
            best_match.insert(0, suit_Names[i])
            if best_rank == 0:
                break
        i = i + 1
    
    return best_match[0]
    

def CompareRank(picked_img):
    # Image to be compared
    best_rank  = 10000
    best_match = []
    i = 0
    
    # Comparing until match
    for img in rank_Images:
        diff_img = cv2.absdiff(picked_img, img)
        rank_diff = int(np.sum(diff_img)/255)
        if rank_diff < best_rank:
            best_rank = rank_diff
            best_match.insert(0, rank_Names[i])
            if best_rank == 0:
                break
        i = i + 1
    
    return best_match[0]

   