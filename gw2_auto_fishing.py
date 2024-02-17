import numpy as np
import cv2
from mss import mss
from PIL import Image
import pyautogui

def get_screen_size() -> dict:
    screenWidth, screenHeight = pyautogui.size()

    screenToCapture = {
        'top' : int( 1/3 * screenHeight ),
        'left' : int( 1/3 * screenWidth ),
        'width' : 1100,
        'height' : 650
    }

    return screenToCapture


if __name__ == "__main__":

    # capture current screen.
    # find greenBox and yellowBox.

    greenBox = cv2.imread( 'images/greenBox.png' )
    yellowBox = cv2.imread( 'images/yellowBox.png' )

    greenBox = cv2.cvtColor( greenBox, cv2.COLOR_RGB2GRAY )
    yellowBox = cv2.cvtColor( yellowBox, cv2.COLOR_RGB2GRAY )
    
    with mss() as sct:
        while True:
            # Screenshot predefined region.
            screenShot = sct.grab( get_screen_size() )
            screenShot = np.array( screenShot )

            # Convert 2 grayscale to sastify the dimension requirement.
            grayScaleScreenShot = cv2.cvtColor( screenShot,
                                               cv2.COLOR_RGBA2GRAY )
            
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
            
            greenMinVal, greenMaxVal, greenMinLoc, greenMaxLoc = cv2.minMaxLoc( greenMatches )
            yellowMinVal, yellowMaxVal, yellowMinLoc, yellowMaxLoc = cv2.minMaxLoc( yellowMatches )

            # greenBox & yellowBox' shapes.
            greenBoxWidth   = greenBox.shape[1]
            greenBoxHeight  = greenBox.shape[0]
            yellowBoxWidth  = yellowBox.shape[1]
            yellowBoxHeight = yellowBox.shape[0]

            greenBoxLoc2 = ( 
                greenMaxLoc[0] + greenBoxWidth,
                greenMaxLoc[1] + greenBoxHeight
            )
            yellowBoxLoc2 = (
                yellowMaxLoc[0] + yellowBoxWidth,
                yellowMaxLoc[1] + yellowBoxHeight
            )

            # Draw on to our screen.
            cv2.rectangle( screenShot, greenMaxLoc,
                           greenBoxLoc2,
                           (0,255,255), 2 )
            cv2.rectangle( screenShot, yellowMaxLoc,
                           yellowBoxLoc2,
                           (255,0,255), 2 )

            cv2.imshow( 'Screen', screenShot )


            # Quit by pressing q.
            if( cv2.waitKey(1) & 0xFF ) == ord( 'q' ):
                cv2.destroyAllWindows()
                break