# chess-vision
# Chess Vision

Chess Vision is a project that aims to detect and analyze chess boards and pieces from images. Using trained models, the project can identify the positions of chess pieces on a board and convert this information into the standard Forsyth-Edwards Notation (FEN). This FEN can then be used to analyze the game on platforms like Lichess.

## Features

- **Chessboard Corner Detection**: Detects the four corners of a chessboard in an image.
- **Chess Piece Detection**: Identifies the chess pieces on a board.
- **FEN Generation**: Converts the detected chessboard state into a FEN string.
- **Lichess Integration**: Opens a Lichess board analysis page using the detected FEN.

## Usage

1. **Image to Lichess**: Convert an image of a chessboard to a Lichess board analysis link. This function reads an image of a chessboard, detects the pieces on it, converts the detections to FEN representation, and then opens a Lichess board analysis page using the detected FEN.

## Configuration

The project uses a configuration file (`config.ini`) to store global parameters and model paths. The configuration contains:

- **GLOBAL_PARAMS**: Parameters for each chess piece type.
- **MODELS**: Paths to the trained models for chessboard corner prediction and chess piece prediction.

## Dependencies

- `ultralytics`
- `IPython`
- `webbrowser`
- `numpy`
- `matplotlib`
- `cv2`
- `shapely`

## How to Run

1. Clone the repository.
2. Ensure you have the required dependencies installed.
3. Run the `Project.ipynb` notebook to see the project in action.
