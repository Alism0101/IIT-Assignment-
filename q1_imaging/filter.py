import cv2
import numpy as np

def guessTheGlow(magicLogPicture, smudgeFactor):
    
    if smudgeFactor % 2 == 0:
        smudgeFactor += 1
        
    picHeight, picWidth = magicLogPicture.shape
    theShadowyTruth = np.zeros_like(magicLogPicture)
    
    cushionSize = smudgeFactor // 2
    fluffyPicture = np.pad(magicLogPicture, cushionSize, mode='reflect')
    
    for row_num in range(picHeight):
        for col_num in range(picWidth):
            peeking_window = fluffyPicture[row_num : row_num + smudgeFactor, col_num : col_num + smudgeFactor]
            
            theShadowyTruth[row_num, col_num] = np.mean(peeking_window)
            
    return theShadowyTruth

def makeItBoring(whereTheFileSleeps, magicWandSize):
    boringOldPic = cv2.imread(whereTheFileSleeps)
    if boringOldPic is None:
        print(f"Error loading {whereTheFileSleeps}")
        return

    fiftyShadesOfGray = cv2.cvtColor(boringOldPic, cv2.COLOR_BGR2GRAY)
    
    logRhythms = np.log(fiftyShadesOfGray.astype(float) + 1.0)

    theBoringPart = guessTheGlow(logRhythms, magicWandSize)
    
    theCoolPart = logRhythms - theBoringPart
    
    actuallyCoolPart = np.exp(theCoolPart)
    
    tadaa = cv2.normalize(actuallyCoolPart, None, 0, 255, cv2.NORM_MINMAX)
    
    cv2.imwrite('reflectance_mono.jpg', tadaa.astype(np.uint8))
    print("Saved reflectance_mono.jpg")

def makeItPretty(whereTheFileSleeps, magicWandSize):
    boringOldPic = cv2.imread(whereTheFileSleeps)
    if boringOldPic is None:
        print(f"Error loading {whereTheFileSleeps}")
        return

    scienceLab = cv2.cvtColor(boringOldPic, cv2.COLOR_BGR2Lab)
    theBrightSide, appleOrAlien, bananaOrSmurf = cv2.split(scienceLab)
    
    logOfBrightness = np.log((theBrightSide.astype(float) / 100.0) + 1.0)

    shadowGuess = guessTheGlow(logOfBrightness, magicWandSize)
    
    sparkleLog = logOfBrightness - shadowGuess
    theRealSparkle = np.exp(sparkleLog)
    
    shinyNewBrightness = cv2.normalize(theRealSparkle, None, 0, 100, cv2.NORM_MINMAX)
    shinyNewBrightness = shinyNewBrightness.astype(np.uint8)
    
    fixedSciencePic = cv2.merge([shinyNewBrightness, appleOrAlien, bananaOrSmurf])
    
    voila = cv2.cvtColor(fixedSciencePic, cv2.COLOR_Lab2BGR)
    
    cv2.imwrite('reflectance_color.jpg', voila)
    print("Saved reflectance_color.jpg")

if __name__ == "__main__":
    
    SMUDGE_O_METER = 5
    
    THE_CHOSEN_ONE = 'your_photo.jpg'
    
    print("Processing grayscale version...")
    makeItBoring(THE_CHOSEN_ONE, SMUDGE_O_METER)
    
    print("Processing color version...")
    makeItPretty(THE_CHOSEN_ONE, SMUDGE_O_METER)
    
    print("Done.")

