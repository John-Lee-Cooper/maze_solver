# CV Maze Solver #

Maze_solver provides a relatively quick computer vision solution to most mazes.

![Maze3](solutions/maze3.gif)

## How it works

Maze_solver reads in images of mazes,
performs thresholding and thinning,
and then converts the remaining "on" pixels to a network.
A simple keyboard UI allows the user to update the starting and ending points.
The maze is solved using a simple dijkstra algorithm.

## Installation

```bash
python -m venv venv
pip install -r requirements.in
source venv/bin/activate
```

## Usage

```bash
src/maze_solver.py
```
 Key    | Result
 :---:  | :---
 s      | Move starting point left
 f      | Move starting point right
 e      | Move starting point up
 d      | Move starting point down
 j      | Move ending point left
 l      | Move ending point right
 i      | Move ending point up
 k      | Move ending point down
 space  | Accept start and end points and solve.
 escape | Exit

![Maze3](solutions/maze1.gif)

## Todo

The next steps in this project are to

* Add a CLI to allow user to run on their own mazes
* Save the name of the file and the starting and ending points in a cache file.
* Allow the user to use a camera instead of an image.
* Allow the user to use the mouse to specify starting and ending positions
* Allow the user to display thinned image
* Handle color mazes - thresholding and thinning

## Contributing

Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Written by

John Lee Cooper  
john.lee.cooper@gatech.edu


## Reference

[ref](https://answers.opencv.org/question/84435/is-there-a-guo-hall-thinning-method-in-opencv-310/)

