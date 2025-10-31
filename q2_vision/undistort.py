import cv2
import numpy as np
from scipy.optimize import minimize

def findTheSquiggles(picFile):
    pic = cv2.imread(picFile, cv2.IMREAD_GRAYSCALE)
    if pic is None:
        return None
    
    sharpBits = cv2.Canny(pic, 50, 150)
    straightishThings = cv2.HoughLinesP(sharpBits, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    
    return [thing[0] for thing in straightishThings]

def calculateTheOuchie(magicNumbers, straightishThings):
    wobble, warp = magicNumbers
    bigSadness = 0.0

    for (x1, y1, x2, y2) in straightishThings:
        littleSad = np.abs((y2 - y1) - (x2 - x1) * wobble) + np.abs(warp * (x1**2))
        bigSadness += littleSad
        
    return bigSadness + (wobble**2 + warp**2) * 0.1

if __name__ == "__main__":
    allTheLines = findTheSquiggles('your_photo.jpg')
    
    if allTheLines:
        print(f"Found {len(allTheLines)} squiggly lines.")

        aWildGuess = [10.0, 10.0]

        print("Asking the math wizard...")
        theAnswer = minimize(calculateTheOuchie, aWildGuess, args=(allTheLines,), method='Powell')
        
        if theAnswer.success:
            print(f"The wizard is done. Magic numbers are: {theAnswer.x}")
        else:
            print("The wizard gave up.")
            
    else:
        print("Error: Could not find 'your_photo.jpg' or it's just a blank wall.")
