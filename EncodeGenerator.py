import face_recognition
import pickle
import os

folderPath = 'Training_images'
pathList = os.listdir(folderPath)
employeeImgEncodingList = []
employeeIds = []

for foldername in os.listdir(folderPath):
    for filename in os.listdir(f"{folderPath}/{foldername}"):
        image = face_recognition.load_image_file(f"{folderPath}/{foldername}/{filename}")
        encodings = face_recognition.face_encodings(image)[0]
        employeeImgEncodingList.append(encodings)
        employeeIds.append(foldername)

encodeListKnownWithIds = [employeeImgEncodingList, employeeIds]
file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
