# AI Block Rotation Detector

Reverse perspective using a block with 4 points.
Classify clicked blocks using a trained model.

ONLY FOR GRASS BLOCKS

## How to run
1. pip install -r requirements.txt
2. python main.py --image <image_path>

## How to use.
1. Click on 4 corners of a block or square in a clockwise maner, starting at the North-West corner of the block.
2. Watch the corrected image update.
3. Tune the points.
4. Click on a square in the grid to classify it.
5. Press e to export rotation data.
6. Goto the [Texture Rotations](https://github.com/19MisterX98/TextureRotations) or [texture finder cuda](https://github.com/Selyss/texture-finder-cuda) repository and run using formation.txt.

## How to tune points.
- The red point is the last selected point.
- Use wasd keys to move red point.
- Move around points using mouse.
- Moving a point with the mouse will select it.
- **Points can also be selected by pressing 1, 2, 3, or 4.**
- Press R to reset points.
- Press + or - to change step increment.

## Other
- Press q to quit.

This project contains the dataset and processed dataset used for training the model.
The dataset came from fanda857.

# training gen
r
``python generate_variants.py``
defaults to input.png set it to a 0° rotation top texture. I thought --alt did something but maybe I'm crazy oh I did, it's in the datagen branch