from flask import Flask, render_template, request, redirect, jsonify
from werkzeug.utils import secure_filename
import os
from chess import image_to_lichess
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Convert the original image to base64
            original_image_base64 = convert_image_to_base64(filepath)

            # Process the image using your functions
            cropped_image_base64, image_with_predictions_base64, lichess_link = image_to_lichess(filepath)

            return render_template('index.html', results=True, original_image_data=original_image_base64,
                                   cropped_image_data=cropped_image_base64,
                                   processed_image_data=image_with_predictions_base64, lichess_link=lichess_link)
    return render_template('index.html', results=False)


@app.route('/get-lichess-link', methods=['POST'])
def get_lichess_link():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image using your functions
        _, _, lichess_link = image_to_lichess(filepath)

        return jsonify({"lichess_link": lichess_link})

    return jsonify({"error": "An error occurred while processing the image"}), 500


if __name__ == '__main__':
    app.run(debug=True)
