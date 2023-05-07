import face_recognition
import numpy as np
import pickle
from flask import Flask, request

app = Flask(__name__)

@app.route('/identify', methods=['POST'])
def identify_face():
    # Read the image from the request
    image_data = request.files['image']
    # Convert the image buffer to a numpy array
    # nparr = np.frombuffer(image_data, np.uint8)
    # Load the image using face_recognition
    image = face_recognition.load_image_file(image_data)

    # Find faces in the image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Load the known encodings from a pickle file
    with open('EncodeFile.p', 'rb') as f:
        known_encodings = pickle.load(f)
    
    encodeListKnown, studentIds = known_encodings

    # Identify the faces
    identified_names = []
    for face_encoding in face_encodings:
        # Compare the face encoding to the known encodings
        matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
        if True in matches:
            # Get the index of the matched encoding and the corresponding name
            match_index = matches.index(True)
            identified_names.append(studentIds[match_index])

    # Return the identified names
    if len(identified_names) == 0:
        return 'No faces identified'
    elif len(identified_names) == 1:
        return f'Identified person: {identified_names[0]}'
    else:
        return f'Identified people: {", ".join(identified_names)}'

if __name__ == '__main__':
    app.run()