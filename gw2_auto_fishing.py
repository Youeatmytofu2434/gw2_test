import numpy as np
import cv2
import pyautogui
import pydirectinput
from mss import mss
from numpy import ndarray


YELLOW_BOX_THRESHOLD = 0.95


def get_screen_to_capture() -> dict:
    screenWidth, screenHeight = pyautogui.size()

    screenToCapture = {
        'top' : int( 1/3 * screenHeight ),
        'left' : int( 1/3 * screenWidth ),
        'width' : 1100,
        'height' : 750
    }

    return screenToCapture

def get_object_dimensions( srcImg: ndarray ):
    srcImgWidth = srcImg.shape[1]
    srcImgHeight = srcImg.shape[0]
    return ( srcImgWidth, srcImgHeight )

def group_rectangles( potentialMatches: ndarray, objectImg: ndarray ):
    ( objectWidth, objectHeight ) = get_object_dimensions(
        objectImg
    ) 

    detectedObjects = [] # Holds all potential rectangles to
                                # group them later on.

    # Grab everything that the program think is the yellow box.
    yLoc, xLoc = np.where( potentialMatches >= YELLOW_BOX_THRESHOLD )

    # Make sure there are at least 2 rectangles for each instance 
    # for grouping to work.
    for( x, y ) in zip( xLoc, yLoc ):
        detectedObjects.append(
            [x, y, objectWidth+x, objectHeight+y]
        )
        detectedObjects.append( 
            [x, y, objectWidth+x, objectHeight+y] 
        )

    detectedObjects, _ = cv2.groupRectangles(
        detectedObjects,
        groupThreshold = 1, 
        eps = 0.2
    )

    return detectedObjects

def is_box_on_left( greenBoxRec, yellowBoxRec ):
    if( ( greenBoxRec[0] - yellowBoxRec[0] ) < 0 ):
        return True
    return False

def move_yellow_box( greenBoxRecs, yellowBoxRecs ):
    if( len( greenBoxRecs ) > 0 and
        len( yellowBoxRecs) > 0 ):
        # Check if greenbox is on the left.
        if( is_box_on_left(
            greenBoxRecs[0], yellowBoxRecs[0]
        ) ):
            pydirectinput.keyUp( 'd' )
            pydirectinput.keyDown( 'a' )
        else:
            pydirectinput.keyUp( 'a' )
            pydirectinput.keyDown( 'd' )

if __name__ == "__main__":

    greenBox = cv2.imread( 'images/greenBox.png' )
    yellowBox = cv2.imread( 'images/yellowBoxTest.png' )

    greenBox = cv2.cvtColor( greenBox, cv2.COLOR_RGB2GRAY )
    yellowBox = cv2.cvtColor( yellowBox, cv2.COLOR_RGB2GRAY )
    
    with mss() as sct:
        while True:
            # Screenshot predefined region.
            screenShot = sct.grab( get_screen_to_capture() )
            screenShot = np.array( screenShot )

            # Convert 2 grayscale to sastify the dimension requirement.
            grayScaleScreenShot = cv2.cvtColor(
                screenShot,
                cv2.COLOR_RGBA2GRAY
            )
            
            # Template matching, grab anything that resembles our objects.
            greenMatches = cv2.matchTemplate(
                grayScaleScreenShot,
                greenBox,
                cv2.TM_CCOEFF_NORMED
            )
            yellowMatches = cv2.matchTemplate(
                grayScaleScreenShot,
                yellowBox,
                cv2.TM_CCOEFF_NORMED
            )


            potentialGreenBoxes  = group_rectangles(
                greenMatches,
                greenBox
            )
            potentialYellowBoxes = group_rectangles(
                yellowMatches,
                yellowBox
            )

            move_yellow_box( potentialGreenBoxes, potentialYellowBoxes )

            # DRAW on screen.
            for ( x1,y1,x2,y2 ) in potentialGreenBoxes:
                cv2.rectangle( screenShot, ( x1,y1 ),
                            ( x2,y2 ), (0,255,255),
                            2
                )
            
            for ( x1,y1,x2,y2 ) in potentialYellowBoxes:
                cv2.rectangle( screenShot, ( x1,y1 ),
                            ( x2,y2 ), (255,0,255),
                            2
                )
            
            cv2.imshow( 'Screen', screenShot )

            # Quit by pressing q.
            if( cv2.waitKey(1) & 0xFF ) == ord( 'q' ):
                cv2.destroyAllWindows()
                break