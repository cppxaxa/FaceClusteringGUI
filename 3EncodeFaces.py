import argparse
import os
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

parser = argparse.ArgumentParser(description='Sample 1')
parser.add_argument('input', metavar='input_path')
parser.add_argument('--output', dest='output', metavar='output_path')
args = parser.parse_args()

print(str(args.input) + ' ' + str(args.output))

EncodingOutput = os.path.join(args.output, "EncodingOutput")
if not os.path.exists(EncodingOutput):
    os.makedirs(EncodingOutput)

InputDataset = args.input
if not os.path.exists(InputDataset):
	print('The input dataset directory does not exists')
	exit()

DetectionMethod = 'cnn'
# DetectionMethod = 'hog'

# grab the paths to the input images in our dataset, then initialize
# out data list (which we'll soon populate)
# print("[INFO] Quantifying faces")
imagePaths = list(paths.list_images(InputDataset))
data = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	print("[INFO] Processing image {}/{} - {}".format(i + 1, len(imagePaths), imagePath))
	image = cv2.imread(imagePath)

	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb, model=DetectionMethod)

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# build a dictionary of the image path, bounding box location,
	# and facial encodings for the current image
	d = [{"imagePath": imagePath, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
	data.extend(d)

# dump the facial encodings data to disk
print("[INFO] serializing encodings...")
OutputFile = os.path.join(EncodingOutput, "Encoding.pickle")
f = open(OutputFile, "wb")
f.write(pickle.dumps(data))
f.close()
