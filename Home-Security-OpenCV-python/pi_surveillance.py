# Run this in cmd line: python3 pi_surveillance.py --conf conf.json
# python3 Face_Trainer.py
# import the necessary packages
from pyimagesearch.tempimage import TempImage
#------------ Lines 4-5 will allow us to access the raw video stream of the Raspberry Pi camera  ------------ #
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
#------------- Line 10 grabs the Dropbox API ---------------- #
import dropbox
import imutils
import json
import time
import cv2
import numpy as np #For converting Images to Numerical array 
import os #To handle directories 
from PIL import Image #Pillow lib for handling images 

labels = ["Chris", "Daenerys", "Daniel"] 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load("face-trainner.yml")

# ------------ Lines 20-23 handle parsing our command line arguments. ----------- #
# ------------ All we need is a single switch, --conf , which is the path to where our JSON configuration file lives on disk. -------------- # 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="path to the JSON configuration file")
args = vars(ap.parse_args())


# ------------ Line 30 filters warning notifications from Python, specifically ones generated from urllib3  and the dropbox  packages ----------- #
# ------------ lastly, we’ll load our JSON configuration dictionary from disk on Line 31 and initialize our Dropbox client  on Line 32. ----------- #
# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None


# ------------ On Line 38 we make a check to our JSON configuration to see if Dropbox should be used or not. -------------- #
# ------------ If it should, Line 40 authorizes our app with the API key.
# check to see if the Dropbox should be used
if conf["use_dropbox"]:
	# connect to dropbox and start the session authorization process
	client = dropbox.Dropbox(conf["dropbox_access_token"])
	print("[SUCCESS] dropbox account linked")

# ------------ We setup our raw capture to the Raspberry Pi camera on Lines 45-48 -------------- #
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

# ------------ we’ll initialize the average background frame, along with some bookkeeping variables on Lines 52-54. ---------- #
# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

	# ----------------- looping over frames directly from our Raspberry Pi video stream ----------------- #

# ------------------ We pre-process our frame a bit by resizing it to have a width of 500 pixels, followed by converting it to grayscale, and applying a Gaussian blur to remove high frequency noise and allowing us to focus on the “structural” objects of the image. ------------------- #
# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image and initialize
	# the timestamp and occupied/unoccupied text
	frame = f.array		# frame = img
	timestamp = datetime.datetime.now()
	text = "Empty"
 
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) #Recog. faces
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	# if the average frame is None, initialize it
	if avg is None:
		print("[INFO] starting background model...")
		avg = gray.copy().astype("float")
		rawCapture.truncate(0)
		continue
	
# ------------------- take the weighted mean of previous frames along with the current frame. This means that our script can dynamically adjust to the background, even as the time of day changes along with the lighting conditions. -------------- #
# ------------------- Based on the weighted average of frames, we then subtract the weighted average from the current frame, leaving us with what we call a frame delta: ------------------- #
# ------------------- delta = |background_model – current_frame| -------------- #
	# accumulate the weighted average between the current frame and
	# previous frames, then compute the difference between the current
	# frame and running average
	cv2.accumulateWeighted(gray, avg, 0.5)
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

		# ---------------- We can then threshold this delta to find regions of our image that contain substantial difference from the background model — these regions thus correspond to “motion” in our video stream: ------------------ #
	# ---------- To find regions in the image that pass the thresholding test, we simply apply contour detection. ------ #
	# threshold the delta image, dilate the thresholded image to fill
	# in holes, then find contours on thresholded image
	thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
		cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
 
	# ------------ We then loop over each of these contours individually (Line 101) ----------- # 
	# ------------ and see if the pass the min_area  test (Lines 104 and 105), If the regions are sufficiently larger enough, then we can indicate that we have indeed found motion in our current frame ------------ #
	# loop over the contours (Contours can be explained as a curve joining all the continuous points (along the boundary), having same colour or intensity. Useful for shaping and object detection)
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < conf["min_area"]:
			continue
 
		# -------------- Lines 110-112 then compute the bounding box of the contour, draw the box around the motion, and update our text  variable. -------- #
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Movement Detected"
		
		for (x, y, w, h) in faces:
			name = "Unknown"
			roi_gray = gray[y:y+h, x:x+w] #Convert Face to greyscale 

			id_, confi = recognizer.predict(roi_gray) #recognize the Face
			
			print(confi)
		
			if confi>=150:
				font = cv2.FONT_HERSHEY_SIMPLEX #Font style for the name 
				name = labels[id_] #Get the name from the List using ID number 
				cv2.putText(frame, name, (x,y), font, 1, (0,0,255), 2)
			else: 
				font = cv2.FONT_HERSHEY_SIMPLEX #Font style for the name 
				cv2.putText(frame, "Unknown", (x,y), font, 1, (0,0,255), 2)
 
	# ------------- Finally, Lines 116-120 take our current timestamp and status text  and draw them both on our frame. ---------- #
	# draw the text and timestamp on the frame
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)

		# ------------------- Code to hand uploading to Dropbox ------------- #
	# --------- We make a check on Line 126 to see if we have indeed found motion in our frame. ---------- #
	# --------- If so, we make another check on Line 128 to ensure that enough time has passed between now and the previous upload to Dropbox — if enough time has indeed passed, we’ll increment our motion counter -------------- #
	# check to see if the room is occupied
	if text == "Movement Detected":
		# check to see if enough time has passed between uploads
		if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
			# increment the motion counter
			motionCounter += 1
 
			# --------------- If our motion counter reaches a sufficient number of consecutive frames (Line 135), ---------- #
			# --------------- we’ll then write our image to disk using the TempImage class, upload it via the Dropbox API, and then reset our motion counter and last uploaded timestamp. -------------------- #
			# --------------- If motion is not found in the room (Lines 129 and 130), we simply reset our motion counter to 0. -------------- #
			# check to see if the number of frames with consistent motion is
			# high enough
			if motionCounter >= conf["min_motion_frames"]:
				# check to see if dropbox should be used
				if conf["use_dropbox"]:
					# write the image to temporary file
					t = TempImage()
					cv2.imwrite(t.path, frame)
 
					# upload the image to Dropbox and cleanup the tempory image
					print("[UPLOAD] {}".format(ts))
					path = "/{base_path}/{timestamp}.jpg".format(
					    base_path=conf["dropbox_base_path"], timestamp=ts)
					client.files_upload(open(t.path, "rb").read(), path)
					t.cleanup()
 
				# update the last uploaded timestamp and reset the motion
				# counter
				lastUploaded = timestamp
				motionCounter = 0
 
	# otherwise, the room is not occupied
	else:
		motionCounter = 0

	# ----------------  We make a check to see if we are supposed to display the video stream to our screen (based on our JSON configuration), ------------ #
	# check to see if the frames should be displayed to screen
	if conf["show_video"]:
		# display the security feed
		cv2.imshow("Security Feed", frame)
		key = cv2.waitKey(1) & 0xFF
 
		# if the `q` key is pressed, break from the loop
		if key == ord("q"):
			break
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)

# Credits to Adrian at PyimageSearch
