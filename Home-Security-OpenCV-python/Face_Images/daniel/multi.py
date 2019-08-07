from picamera import PiCamera

camera = PiCamera()

for i in range(30):
	camera.capture('Adan_image{0:04d}.jpg'.format(i))
