# Security-cam-face-recog

Home Security Camera with Facial Recognition and Dropbox

To run the program in cmd line:
	-> python3 Face_Trainer.py
	-> python3 pi_surveillance.py --conf conf.json


Implementation of the program:
-> Video camera appears via the Raspberry Pi Camera Module

	-> Camera displays the status of the room (initialized as 'Empty')
	-> Displays the current time and date 

-> Green rectangle appears (Bounding box) in which it recognizes an object in motion
	-> Changes the display status from 'Empty' to 'Movement Detected'
	-> When confidence level is less than 150, name display is "Unknown"
	-> When condidence level is greater then or equal 150, name is displayed as the face that is trained to recognize
		-> The faces trained in this case are Daenerys (Game of Thrones), Chris Hemsworth, and me (Daniel)
-> If display status is changed from 'Empty' to 'Movement Detected'
	-> Takes a picture of the frame and sends it to Dropbox via Dropbox API


Problems with the program
-> Face recognition is not accurate
	-> Can sometimes mistake me as Daenerys 
	-> Can also mistake me as anyone depending on the lighting (dark room vs light room)
