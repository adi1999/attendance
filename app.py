import face_recognition
import numpy as np
import pickle
from flask import Flask, request
from flask import jsonify
from flask_cors import CORS
import math




app = Flask(__name__)
cors = CORS(app)


dummy_user = {
    'username': 'admin',
    'password': 'admin123'
}


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return round(linear_val * 100, 2)
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return round(value, 2) 

@app.route('/login', methods=['POST'])
def login():
    # Get the username and password from the request's JSON data
    username = request.json.get('username')
    password = request.json.get('password')

    if not username or not password:
        return jsonify({'error': 'Missing username or password'}), 400

    if username != dummy_user['username'] or password != dummy_user['password']:
        return jsonify({'error': 'Invalid credentials'}), 401

    return jsonify({'message': 'Login successful'})

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
        face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = studentIds[best_match_index]
            confidence = face_confidence(face_distances[best_match_index])
            if(confidence > 90):
                identified_names.append(name)
        
        
        # if True in matches:
        #     # Get the index of the matched encoding and the corresponding name
        #     match_index = matches.index(True)
        #     identified_names.append(studentIds[match_index])
            

    # Return the identified names
    if len(identified_names) == 0:
        return 'No faces identified'
    elif len(identified_names) == 1:
        return f'Identified person: {identified_names[0]}'
    else:
        return f'Identified people: {", ".join(identified_names)}'


@app.route('/register', methods=['POST'])
def encode_images():
    try:
        images = request.files.getlist('images')
        name = request.form.get('name')
        employeeImgEncodingList = []
        employeeIds = []

        if len(images) == 0:
            return 'No images received', 500

        try:
            with open('EncodeFile.p', 'rb') as file:
                encodings = pickle.load(file)
        except FileNotFoundError:
            encodings = []

        for image in images:
            image_data = face_recognition.load_image_file(image)
            encoding = face_recognition.face_encodings(image_data)[0]  # Assuming there is only one face in each image
            encodings[0].append(encoding)
            encodings[1].append(name)

    
        # encodeListKnownWithIds = [employeeImgEncodingList, employeeIds]
        file = open("EncodeFile.p", 'wb')
        pickle.dump(encodings, file)
        file.close()

        return 'Encodings saved successfully.'
    except Exception as e:
        return f'An error occurred: {str(e)}', 500

if __name__ == '__main__':
    app.run()