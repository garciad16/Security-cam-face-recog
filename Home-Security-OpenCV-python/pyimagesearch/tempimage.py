# ------------ This class simply constructs a random filename on Lines 9 and 10, followed by providing a cleanup  method to remove the file from disk once we are finished with it. ------------ #
# import the necessary packages
import uuid
import os
 
class TempImage:
	def __init__(self, basePath="./", ext=".jpg"):
		# construct the file path
		self.path = "{base_path}/{rand}{ext}".format(base_path=basePath,
			rand=str(uuid.uuid4()), ext=ext)
 
	def cleanup(self):
		# remove the file
		os.remove(self.path)
