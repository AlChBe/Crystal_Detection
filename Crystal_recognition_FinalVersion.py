import numpy as np
import cv2
import os

multiple_images = True

resultsdir = r"..."
resultsdir = resultsdir.replace("/", "\\")

if not os.path.exists(resultsdir):
    os.mkdir(resultsdir)

path = r"..." # Single image to test parameters
folder = r"..." # Whole folder
path = path.replace("/", "\\") # Only for Windows
folder = folder.replace("/", "\\") # Only for Windows

images = [folder + "/" + file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))] if multiple_images == True else [path]


crystcount = 0
Wellscount = 0

for i in images:

    #Load image, convert to greyscale, apply Otsu's threshold
    image = cv2.imread(i)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)[1] # First free parameter
    ROI_list = []
    coord = []
    

    # Find contours, obtain bounding rectangle, extract and save ROI
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(gray_image, contours, -1, (100), 2)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if (w > 20 and h > 20) and h < 1000: # Only the wells should be recognized as regions of interest
            ROI_list.append(thresh[y:y+h, x:x+w])
            coord.append((x, y))
            if (np.mean(thresh[y:y+h, x:x+w])) < 210 or abs(w-h) > 0.15*((w+h)/2): # Second free parameter
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                crystcount += 1
            else:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)             

    if multiple_images == True: # Differenting between cases because of how Windows names paths
        cv2.imwrite(resultsdir + "/" + i.split("/")[-1], image)
    else:
        cv2.imwrite(resultsdir + "/" + i.split("\\")[-1], image)

    Wellscount += len(ROI_list)

print("Wells found: " + str(Wellscount))
print("Hit rate: " + str(crystcount/Wellscount))
