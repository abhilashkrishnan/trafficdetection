import cv2

filename = 'traffic.avi'
cap = cv2.VideoCapture(filename)
reference_frame = None
image_area = None
#fourcc = cv2.cv2.CV_FOURCC(*'MP4V')
out = cv2.VideoWriter('traffic-detect.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 25, (1280,720),True)
#out = cv2.VideoWriter('traffic-detect.mp4',fourcc, 20.0, (1280,720))

while True:
	ret,frame = cap.read()
	#print ret

	if ret == False:
		break
	else:
		if reference_frame is None:
			reference_frame = frame
			reference_frame = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
			image_area = reference_frame.shape[0] * reference_frame.shape[1]
			continue

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		difference = cv2.absdiff(reference_frame, gray)
		blurred = cv2.medianBlur(difference, 31)
		
		f, threshold = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)		
		
		(_,contours,_) = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for i in contours:
			contour_area = cv2.contourArea(i)
			if (contour_area > (0.0005 * image_area)) and (contour_area < (0.2 * image_area)):
				(x,y,w,h) = cv2.boundingRect(i)
				cv2.rectangle(frame,(x,y), (x+w,y+h),(0,0,255),2)

		cv2.imshow("Frames", frame)
		out.write(frame)
		if cv2.waitKey(1) == ord('q'):
			break
cap.release()
out.release
cv2.destroyAllWindows()
