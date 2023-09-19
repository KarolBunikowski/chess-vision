import configparser
from ultralytics import YOLO
from IPython.display import display, Image
import webbrowser
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2
from shapely.geometry import Polygon
import base64
from PIL import Image as PILImage
from io import BytesIO

# Constants
CONFIG_FILE_PATH = "config.ini"

def read_configuration(file_path):
    '''Read the configuration from a given file and return the parsed parameters.

    Args:
        file_path (str): Path to the configuration file.

    Returns:
        tuple: Contains two dictionaries, 
               the first for global parameters and 
               the second for trained models.
    '''
    
    config = configparser.ConfigParser()
    config.read(file_path)

    try:
        global_params = {}
        for key in config["GLOBAL_PARAMS"]:
            values = config["GLOBAL_PARAMS"][key].split(", ")
            global_params[int(key)] = (values[0], int(values[1]), float(values[2]))

        model_configs = {
            "chessboard_corners": config["MODELS"]["chessboard_corners_prediction_model"],
            "chess_pieces": config["MODELS"]["chess_pieces_prediction_model"]
        }

        return global_params, model_configs

    except KeyError as e:
        raise KeyError(f"Missing key in configuration: {e}")

# Use the function
global_params, models = read_configuration(CONFIG_FILE_PATH)
chessboard_corners_prediction_model = models["chessboard_corners"]
chess_pieces_prediction_model = models["chess_pieces"]


def calculate_iou(box_1, box_2):
    '''Calculate Intersection over Union (IoU) for two boxes

    Args:
        box_1: First box to calculate IoU for.
            List of four (x,y) tuples creating a box
        box_2: Second box to calculate IoU for.
            List of four (x,y) tuples creating a box

    Returns:
        float: Intersection over Union (IoU) of the two boxes.
            Value between 0 and 1.

    '''
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def box_to_points(box):
    '''Convert a box represented by left, top, right edges to a list of four corners

    Args:
        box: A tuple representing the bounding box
            in the format (left, top, right, bottom).


    Returns:
        List of four corner points of the bounding box
            (top left, top right, bottom right and bottom left)

    '''
    l, t, r, b = box
    tl = [l, t]
    tr = [r, t]
    br = [r, b]
    bl = [l, b]
    box_corners = [tl, tr, br, bl]
    return box_corners


def half_box_to_points(box):
    '''Convert a box represented by left, top, right edges to a list of four corners
        that represent the bottom half of the box

    Args:
        box: A tuple representing the box
            in the format (left, top, right, bottom).


    Returns:
        List of four corner points of bottom half of the box
            (top left, top right, bottom right and bottom left)

    '''
    l, t, r, b = box
    t = t + (b - t) / 2
    tl = [l, t]
    tr = [r, t]
    br = [r, b]
    bl = [l, b]
    box_corners = [tl, tr, br, bl]
    return box_corners


def non_maximum_suppression(predicted_boxes, iou_threshold=0.80):
    '''Eliminate predictions with Intersection over Union (IoU) over 0.8 leaving prediction with more confidence

    Args:
        predicted_boxes: A list of predicted bounding boxes.
            Each bounding box is represented as a tuple (left, top, right, bottom, confidence_score).
        iou_threshold: Threshold for the IoU overlap; any boxes with IoU above this threshold
            will be removed. Defaults to 0.80.
    Returns:
        A list of non-overlapping bounding boxes after applying
            non-maximum suppression.

    '''
    # Sort boxes based on confidence scores
    sorted_boxes = sorted(predicted_boxes, key=lambda x: x[4], reverse=True)
    kept_boxes = []
    while len(sorted_boxes) > 0:
        top_box = sorted_boxes.pop(0)  # Box with highest confidence score
        kept_boxes.append(top_box)

        # Remove all boxes with high IoU overlap
        sorted_boxes = [
            box
            for box in sorted_boxes
            if calculate_iou(box_to_points(top_box[:4]), box_to_points(box[:4]))
            <= iou_threshold
        ]

    return kept_boxes


def board_to_squares(chessboard_image):
    '''Split a chessboard image into individual square regions.

    Given an image of a chessboard, this function divides the image into 8x8 squares,
    returning the coordinates of each square.

    Args:
        chessboard_image: The input image of the chessboard, which should be in the shape (height, width, channels).

    Returns:
        An 8x8x4 array containing the coordinates of each square on the chessboard.
            Each coordinate is in the format (left, top, right, bottom).

    '''
    squares = np.zeros((8, 8, 4))
    height, width = chessboard_image.shape[:2]
    square_width = width / 8
    square_height = height / 8
    for i in range(8):
        for j in range(8):
            squares[j, i] = [
                (square_width * i),
                (square_height * j),
                (square_width * (i + 1)),
                (square_height * (j + 1)),
            ]
    return squares


def detections_to_square(predictions, squares):
    '''Assign detected predictions to the most overlapping squares on a chessboard.

    For each prediction, function calculates Intersection over Union (IoU) of bottm half of the bounding box with all
    squares on the chessboard. The prediction is then assigned to the square with which it has the highest IoU,
    provided that IoU is higher than any previously seen IoU for that square.

    Args:
        predictions:A list of detected object predictions. Each prediction is represented as a tuple,
            where the first four values are bounding box coordinates in the format (left, top, right, bottom),
            and the sixth value is the class index of the predicted object.
        squares:An 8x8x4 array containing the coordinates of each square on the chessboard.
            Each coordinate is in the format (left, top, right, bottom).

    Returns:
        An 8x8 matrix representing the chessboard, where each cell contains a class name
            of the detected object or is an empty string if no object is detected.

    '''
    # Initialize the chessboard with empty strings
    chessboard_with_predictions = np.empty((8, 8), dtype=object)
    chessboard_with_predictions.fill("")  # Fill with empty strings

    # This 8x8 array will store the maximum IoU values we find for each prediction
    max_iou_values = np.zeros((8, 8))

    for prediction in predictions:
        max_iou_for_prediction = 0  # Store the maximum IoU for the current prediction
        max_iou_square_coords = (
            0,
            0,
        )  # Store the coordinates of the square with the max IoU

        for i in range(8):
            for j in range(8):
                square = squares[i, j]
                iou = calculate_iou(
                    half_box_to_points(prediction[:4]), box_to_points(square)
                )

                # If this square has a higher IoU with the prediction than previous squares
                if iou > max_iou_for_prediction:
                    max_iou_for_prediction = iou
                    max_iou_square_coords = (i, j)

        # Once we've found the square with the highest IoU for this prediction, assign the prediction's class to it
        # But only if this IoU is higher than the previous maximum IoU for that square
        if max_iou_for_prediction > max_iou_values[max_iou_square_coords]:
            max_iou_values[max_iou_square_coords] = max_iou_for_prediction

            # Translate the class index to the class name using the dictionary
            class_name = global_params[int(prediction[5])][0]
            chessboard_with_predictions[max_iou_square_coords] = class_name

    return chessboard_with_predictions


def array_to_fen(arr):
    '''Convert a chessboard array into its corresponding Forsyth–Edwards Notation (FEN) representation.

    This function takes an 8x8 matrix representing a chessboard, where each cell contains a class name
    of the chess piece or is an empty string if no piece is on that square. It then translates this
    matrix to a FEN string which is a standard representation for the positions of pieces on a chessboard.

    Args:
        arr: An 8x8 matrix where each cell contains a class name of the chess piece
            or is an empty string if no piece is on that square.

    Returns:
        A string representing the FEN notation of the given chessboard array.
    '''
    fen_list = []

    for row in arr:
        count = 0  # This will keep track of continuous empty squares
        row_str = ""

        for square in row:
            if square == "":
                count += 1
            else:
                if count != 0:
                    row_str += str(count)
                    count = 0
                row_str += square

        # If there are any remaining empty squares at the end of a row
        if count:
            row_str += str(count)

        fen_list.append(row_str)

    return "/".join(fen_list)


def fen_to_link(fen_str):
    '''Open a web browser with a Lichess analysis board for a given FEN string.

    Given a Forsyth–Edwards Notation (FEN) string, this function constructs a URL
    to view the chess position on Lichess's analysis board and opens it in a web browser.

    Args:
        fen_str: A string representing a chess position in FEN format.

    '''
    base_url = "https://lichess.org/analysis/"
    full_url = base_url + fen_str
    # webbrowser.open(full_url)
    return full_url


def corner_boxes_to_points(pred):
    '''Extract the corner points from the detected bounding boxes.

    This function processes the detected bounding boxes to extract the specific
    corner points based on the class index of the box.

    Args:
        pred: A list of detected bounding boxes in an order. Each box is represented as a tuple,
            where the first four values are the bounding box coordinates (left, top, right, bottom),
            and the sixth value is the class index indicating the specific corner.

    Returns:
        A numpy array with shape (4, 2) containing the (x, y) coordinates of the extracted corners.

    '''
    corners = np.zeros((4, 2))
    for box in pred:
        if box[5] == 0 or box[5] == 3:
            corners[int(box[5])] = box_to_points(box[:4])[1]
        else:
            corners[int(box[5])] = box_to_points(box[:4])[0]
    return corners


def four_point_transform(image_np, pts):
    '''Apply a perspective transformation to the given image using four points.

    This function takes in an image and a set of four points, then performs a perspective
    transformation (often called a "birds eye view" transformation) to obtain a top-down view
    of the region defined by the four points.

    Args:
        image_np: The input image represented as a numpy array.
        pts: A list containing four corner points which define the region to be transformed.
            Each point is a tuple with two values (x, y).

    Returns:
        A numpy array representing the warped image after applying the perspective transform.

    '''
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    (tl, tr, br, bl) = pts

    # compute the width of the new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # construct set of destination points to obtain a "birds eye view"
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    # compute the perspective transform matrix and then apply it
    m = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image_np, m, (max_width, max_height))

    # return the warped image as a numpy array
    return warped


def display_image(image):
    '''Display the provided image in the notebook interface.

    This function takes an OpenCV image, converts it to the JPEG format,
    and then displays it directly within the Jupyter Notebook or IPython environment.

    Args:
        image: The input image represented as a numpy array in OpenCV format (BGR).

    '''
    display(Image(data=cv2.imencode(".jpg", image)[1].tobytes()))


def detect_corners(image):
    '''Detect the four corners of a chessboard in the given image.

    This function utilizes a trained YOLO model to identify the four corners
    of a chessboard in the provided image. The model detects and returns bounding boxes for each corner.

    Args:
        image: The input image in which to detect the chessboard corners.

    Returns:
        A numpy array containing the (x, y) coordinates of the four detected corners.

    '''
    
    # YOLO model trained to detect corners on a chessboard
    model_trained = YOLO(chessboard_corners_prediction_model)

    found_corners = []
    for i in range(4):  # Looping over the classes (0, 1, 2, 3)
        corner_prediction = model_trained.predict(
            source=image,
            conf=0.01,
            classes=i,
            max_det=1,
        )
        found_corners.append(corner_prediction[0].boxes.data.numpy()[0])

    # get the corners coordinates from the model
    corners = corner_boxes_to_points(found_corners)

    return corners


def detect_pieces(image):
    '''Detect the chess pieces present on a cropped chessboard image.

    This function utilizes a trained YOLO model to identify the chess pieces
    on the provided chessboard image and returns their bounding boxes.

    Args:
        image: The cropped chessboard image in which to detect the chess pieces.

    Returns:
        A numpy array containing the bounding boxes of the detected chess pieces.
        Each bounding box is represented as [left, top, right, bottom, confidence_score, class_index].

    '''
    # YOLO model trained to detect corners on a cropped chessboard
    model_trained = YOLO(chess_pieces_prediction_model)

    piece_prediction = model_trained.predict(
        source=image, iou=0.85, conf=0.25
    )
    prediction_boxes = piece_prediction[0].boxes.data.numpy()

    return prediction_boxes


def detect_pieces_limited(image):
    '''Detect the chess pieces present on a cropped chessboard image using per-piece model configurations.

    This function leverages a trained YOLO model to identify each type of chess piece
    on the provided chessboard image based on individual configurations from a global
    parameter set. It returns the combined bounding boxes of all detected pieces.

    Args:
        image: The cropped chessboard image in which to detect the chess pieces.

    Returns:
        A numpy array containing the bounding boxes of the detected chess pieces.
        Each bounding box is represented as [left, top, right, bottom, confidence_score, class_index].

    '''
    # YOLO model trained to detect corners on a cropped chessboard
    model_trained = YOLO(chess_pieces_prediction_model)

    predicted_pieces = []
    for class_num, (piece, max_det, conf) in global_params.items():
        prediction = model_trained.predict(
            source=image,
            iou=0.85,
            classes=class_num,
            conf=conf,
            max_det=max_det,
        )
        predicted_pieces.append(prediction[0].boxes.data.numpy())

    flattened = [item for sublist in predicted_pieces for item in sublist]
    prediction_boxes = np.vstack(flattened)

    return prediction_boxes


def display_predictions(image, predictions):
    '''Display an image overlaid with bounding boxes of detected chess pieces.

    This function visualizes the provided image and overlays the detected chess pieces
    using bounding boxes. Each box is color-coded and labeled with the piece type
    and the prediction confidence.

    Args:
        image: The base image on which predictions are to be displayed.
        predictions: A list of detected chess piece predictions. Each prediction is
            represented as [left, top, right, bottom, confidence_score, class_index].

    '''
    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_rgb)

    # Class colors for better visualization
    colors = [
        "r",
        "g",
        "b",
        "y",
        "m",
        "c",
        "orange",
        "purple",
        "grey",
        "lime",
        "pink",
        "brown",
    ]

    for prediction in predictions:
        x1, y1, x2, y2, prob, cls = prediction
        cls_name = global_params[int(cls)][0]
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1,
            edgecolor=colors[int(cls) % len(colors)],
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.text(
            x1,
            y1,
            f"Class: {cls_name}, Prob: {prob:.2f}",
            bbox=dict(facecolor=colors[int(cls) % len(colors)], alpha=0.5),
        )


    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_with_predictions = PILImage.open(buf)
    
    # plt.show()
    
    return img_with_predictions


def image_to_lichess(image_path):
    
    '''
    Convert an image of a chessboard to a Lichess board analysis link.

    This function reads an image of a chessboard, detects the pieces on it,
    converts the detections to a Forsyth-Edwards Notation (FEN) representation,
    and then opens a Lichess board analysis page using the detected FEN.

    Args:
        image_path (str): The path to the input chessboard image.

    Returns:
        cropped_image_base64: cropped image of a chessboard found in the photo in Base64
        image_with_predictions_base64: cropped image of a chessboard found in the photo in Base64
        with bounding boxes of detected chess pieces
        lichess_url: url to analysis of the position on lichess.org
    '''

    # Read the image from the provided path.
    image = cv2.imread(image_path)

    # Display the original image.
    display_image(image)

    # Detect corners of the chessboard using a trained model.
    corners = detect_corners(image)

    # Use the detected corners to extract and warp the chessboard to get a bird's-eye view.
    image_cropped = four_point_transform(image, corners)

    # Display the cropped and transformed chessboard.
    display_image(image_cropped)

    # Uncomment the below lines if you want to detect pieces without limiting instances of each piece.
    # prediction = detect_pieces(image_cropped)
    # boxes_nms = non_maximum_suppression(prediction)

    # Detect pieces on the cropped image using a trained model.
    # This method limits instances of each piece for detection.
    prediction = detect_pieces_limited(image_cropped)

    # Apply Non-Maximum Suppression (NMS) to eliminate overlapping boxes.
    boxes_nms = non_maximum_suppression(prediction)

    # Display the cropped image with the predicted bounding boxes overlaid.
    image_with_predictions = display_predictions(image_cropped, boxes_nms)

    # Convert the bounding box predictions to an 8x8 matrix representation of the chessboard.
    chessboard_matrix = detections_to_square(boxes_nms, board_to_squares(image_cropped))

    # Convert the matrix representation to a FEN string.
    fen = array_to_fen(chessboard_matrix)

    # Open a Lichess board analysis page with the detected FEN.
    lichess_url = fen_to_link(fen)
    
    #change image_cropped to PIL Image format
    cropped_image_pil = PILImage.fromarray(cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB))
    
    # Convert both images to Base64
    buffered = BytesIO()
    cropped_image_pil.save(buffered, format="JPEG")
    cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    buffered = BytesIO()
    image_with_predictions = image_with_predictions.convert("RGB")
    image_with_predictions.save(buffered, format="JPEG")
    image_with_predictions_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    
    return cropped_image_base64, image_with_predictions_base64, lichess_url