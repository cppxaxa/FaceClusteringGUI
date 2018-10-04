import argparse
import os
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

def rescale_by_height(image, target_height, method=cv2.INTER_LANCZOS4):
    """Rescale `image` to `target_height` (preserving aspect ratio)."""
    w = int(round(target_height * image.shape[1] / image.shape[0]))
    return cv2.resize(image, (w, target_height), interpolation=method)

def rescale_by_width(image, target_width, method=cv2.INTER_LANCZOS4):
    """Rescale `image` to `target_width` (preserving aspect ratio)."""
    h = int(round(target_width * image.shape[0] / image.shape[1]))
    return cv2.resize(image, (target_width, h), interpolation=method)


parser = argparse.ArgumentParser(description='Sample 1')
parser.add_argument('input', metavar='input_path')
parser.add_argument('--output', dest='output', metavar='output_path')
args = parser.parse_args()

print(str(args.input) + ' ' + str(args.output))

ResizedImagesFolder = os.path.join(args.output, "ResizedImages")
if not os.path.exists(ResizedImagesFolder):
    os.makedirs(ResizedImagesFolder)

InputDataset = args.input
if not os.path.exists(InputDataset):
	print('The input dataset directory does not exists')
	exit()
    
imagePaths = list(paths.list_images(InputDataset))
data = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] Processing image {}/{} - {}".format(i + 1, len(imagePaths), imagePath))
    image = cv2.imread(imagePath)
    resultImage = rescale_by_width(image, 800)

    newFileName = os.path.join(ResizedImagesFolder, os.path.basename(imagePath))
    cv2.imwrite(newFileName, resultImage)

