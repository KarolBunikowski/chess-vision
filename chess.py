import configparser
from ultralytics import YOLO
from IPython.display import display, Image
import webbrowser
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2
from shapely.geometry import Polygon

# Read Configuration
config = configparser.ConfigParser()
config.read("config.ini")

# Access settings
global_params = {}
for key in config["GLOBAL_PARAMS"]:
    values = config["GLOBAL_PARAMS"][key].split(", ")
    global_params[int(key)] = (values[0], int(values[1]), float(values[2]))

chessboard_corners_prediction_model = config["MODELS"][
    "chessboard_corners_prediction_model"
]
chess_pieces_prediction_model = config["MODELS"]["chess_pieces_prediction_model"]


def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def box_to_points(box):
    l, t, r, b = box
    tl = [l, t]
    tr = [r, t]
    br = [r, b]
    bl = [l, b]
    box_corners = [tl, tr, br, bl]
    return box_corners


def half_box_to_points(box):
    l, t, r, b = box
    t = t + (b - t) / 2
    tl = [l, t]
    tr = [r, t]
    br = [r, b]
    bl = [l, b]
    box_corners = [tl, tr, br, bl]
    return box_corners


def non_maximum_suppression(predicted_boxes, iou_threshold=0.80):
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
    base_url = "https://lichess.org/analysis/"
    full_url = base_url + fen_str
    webbrowser.open(full_url)


def corner_boxes_to_points(pred):
    corners = np.zeros((4, 2))
    for box in pred:
        if box[5] == 0 or box[5] == 3:
            corners[int(box[5])] = box_to_points(box[:4])[1]
        else:
            corners[int(box[5])] = box_to_points(box[:4])[0]
    return corners


def four_point_transform(image_np, pts):

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
    display(Image(data=cv2.imencode(".jpg", image)[1].tobytes()))


def detect_corners(image):
    # YOLO model trained to detect corners on a chessboard
    model_trained = YOLO(chessboard_corners_prediction_model)

    found_corners = []
    for i in range(4):  # Looping over the classes (0, 1, 2, 3)
        corner_prediction = model_trained.predict(
            source=image,
            conf=0.01,
            classes=i,
            save_txt=True,
            save=True,
            save_conf=True,
            max_det=1,
        )
        found_corners.append(corner_prediction[0].boxes.data.numpy()[0])

    # get the corners coordinates from the model
    corners = corner_boxes_to_points(found_corners)

    return corners


def detect_pieces(image):
    # YOLO model trained to detect corners on a cropped chessboard
    model_trained = YOLO(chess_pieces_prediction_model)

    piece_prediction = model_trained.predict(
        source=image, iou=0.85, conf=0.25, save_txt=True, save=True, save_conf=True
    )
    prediction_boxes = piece_prediction[0].boxes.data.numpy()

    return piece_prediction


def detect_pieces_2(image):
    # YOLO model trained to detect corners on a cropped chessboard
    model_trained = YOLO(chess_pieces_prediction_model)

    predicted_pieces = []
    for class_num, (piece, max_det, conf) in global_params.items():
        prediction = model_trained.predict(
            source=image,
            iou=0.85,
            classes=class_num,
            conf=conf,
            save_txt=True,
            save=True,
            save_conf=True,
            max_det=max_det,
        )
        predicted_pieces.append(prediction[0].boxes.data.numpy())

    flattened = [item for sublist in predicted_pieces for item in sublist]
    prediction_boxes = np.vstack(flattened)

    return prediction_boxes


def display_predictions(image, predictions):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

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

    plt.show()


def image_to_lichess(image_path):
    image = cv2.imread(image_path)
    display_image(image)
    corners = detect_corners(image)
    image_cropped = four_point_transform(image, corners)
    display_image(image_cropped)

    # Detection without limiting instances of each piece

    # prediction = detect_pieces(image_cropped)
    # boxes_nms = non_maximum_suppression(prediction)

    # Detection with limiting instances of each piece

    prediction = detect_pieces_2(image_cropped)
    boxes_nms = non_maximum_suppression(prediction)

    display_predictions(image_cropped, boxes_nms)
    final = detections_to_square(boxes_nms, board_to_squares(image_cropped))
    fen = array_to_fen(final)
    fen_to_link(fen)
