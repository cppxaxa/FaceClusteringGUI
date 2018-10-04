import argparse
import os
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Sample 1')
parser.add_argument('input', metavar='input_path')
parser.add_argument('--output', dest='output', metavar='output_path')
args = parser.parse_args()

print(str(args.input) + ' ' + str(args.output))

FramesOutput = os.path.join(args.output, "Frames")
if not os.path.exists(FramesOutput):
    os.makedirs(FramesOutput)

InputVideoFile = args.input
if not os.path.exists(InputVideoFile):
    print('The input video file does not exists')
    exit()

# Create a capture object
cap = cv2.VideoCapture(InputVideoFile)

VideoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

counter = 1
while (cap.isOpened()):
    if (counter == VideoLength):
        break

    ret, frame = cap.read()

    small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    cv2.imwrite(os.path.join(FramesOutput, 'frame_' + str(counter) + ".jpg"), small)
    counter += 1

    if (counter % 100 == 0):
        print(str(counter * 100 / VideoLength) + '%')


    # if(cv2.waitKey(1) & 0xFF == ord('q')):
    #     break

# cv2.destroyAllWindows()

# input("Press Enter to continue...")


