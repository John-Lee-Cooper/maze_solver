#!/usr/bin/env python

"""
https://scipython.com/blog/making-a-maze/
"""

import sys
from typing import Generator, List, Tuple, Dict, Optional
from pathlib import Path

import cv2 as cv
import numpy as np

from param import Param
from dijkstra import Node, Network
from frame_writer import FrameWriter


Image = np.ndarray
Color = Tuple[int, ...]
Vertex = Tuple[int, int]

ESCAPE = 27


class Node2(Node):
    """ 3dof extension for Node """

    def __init__(self, x, y):
        super().__init__()

        # state
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def vertex(self) -> Vertex:
        """ Return state as a vertex for drawing """
        return self.x, self.y


class Network2(Network):
    """ Provide a network for an image """

    # Allow solver to go diagonally with the cost of sqrt(2)
    moves = (
        (+1, +0, 1000),
        (+1, +1, 1414),
        (+0, +1, 1000),
        (-1, +1, 1414),
        (-1, +0, 1000),
        (-1, -1, 1414),
        (+0, -1, 1000),
        (+1, -1, 1414),
    )

    def __init__(self, image: Image, gray: Image):
        super().__init__(self._build_nodes(gray))

        # For _display()
        self.image = image
        self.steps = 0
        self.draw_on = 0

    def _neighbors(self, node: Node2) -> Tuple[Node, int]:
        """ Yield neighbors to node, and the cost to get there """

        x = node.x
        y = node.y

        for dx, dy, cost in self.moves:
            neighbor_node = self.nodes.get((x + dx, y + dy))
            if neighbor_node:
                yield neighbor_node, cost

    def _display(self, curr_node: Node) -> None:
        """ Periodically draw the current best solution """

        if self.steps < self.draw_on:
            self.steps += 1
            return

        self.steps = 0
        self.draw_on += 100

        image = self.image.copy()
        draw_path(image, curr_node.path())
        show(image, 1)
        FrameWriter.write(image)

    @staticmethod
    def _build_nodes(image: Image) -> Dict[Tuple, Node]:
        """
        Convert a OpenCV image into a dictionary of nodes - indexed by col, row
        where every non-zero pixel is a node.
        """

        nodes = {}
        rows = range(image.shape[0])
        cols = range(image.shape[1])
        for row in rows:
            for col in cols:
                if image[row, col]:
                    nodes[col, row] = Node2(col, row)
        return nodes

    def find_shortest_path(self, start: Vertex, finish: Vertex) -> Generator:
        """ Convert start and finish to nodes and return shortest path """

        src_node = self.nodes[start]
        dst_node = self.nodes[finish]
        return super().find_shortest_path(src_node, dst_node)


def draw_path(
    image: Image, path: Generator, thickness: int = 2, color: Color = (0, 0, 256)
) -> None:
    """Draw the path - path is a list of nodes """

    if not path:
        return
    node0 = next(path)
    v0 = node0.vertex()
    for node1 in path:
        v1 = node1.vertex()
        cv.line(image, v0, v1, color, thickness, cv.LINE_AA)
        v0 = v1


def show(image: Image, wait: Optional[int] = 0, name: str = "maze") -> None:
    """ Show the image.  If wait is not none, pause for wait ms """

    cv.imshow(name, image)
    cv.moveWindow(name, 10, 10)
    if wait is not None:
        key = cv.waitKey(wait)
        if key == ESCAPE:
            sys.exit(0)


def morph_open(
    image: Image, k1: int = 3, k2: Optional[int] = None, shape: int = cv.MORPH_CROSS
) -> Image:
    """ Return the image opened using k1xk2 shape kernel """
    if k1 <= 0:
        return image
    kernel = cv.getStructuringElement(shape, (k1, k2 or k1))
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)


def preprocess(image: Image) -> Image:
    """ Convert the image to greyscale and return the result of OTSU thresholding """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # blur = cv.medianBlur(image, blur)
    # gray = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    # gray = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    _, gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return gray


def square(
    image: Image, center: Vertex, size: int, color: Color, thickness: int = -1
) -> None:
    """ Draw a size x size square on the image centered at vertex, using color and thickness """
    delta = size // 2
    tl = center[0] - delta, center[1] - delta
    br = center[0] + delta, center[1] + delta
    cv.rectangle(image, tl, br, color, thickness)


def overlay_start_finish(
    image: Image, start: Vertex, finish: Vertex, radius: int = 6
) -> None:
    """ Draw a green square at the start and red square at then end """
    square(image, start, radius, color=(0, 128, 0))
    square(image, finish, radius, color=(0, 0, 255))


def find_nearest(image: Image, loc: Vertex) -> Vertex:
    """ Return the location of the closest non-zero pixel in image to loc """
    col, row = loc
    rows, cols = image.shape
    for k in range(100):
        min_col = max(col - k, 0)
        max_col = min(col + k + 1, cols)
        for r in range(max(row - k, 0), min(row + k + 1, rows)):
            for c in range(min_col, max_col):
                if image[r, c]:
                    return c, r
    return col, row


def mask(image: Image, start: Vertex, finish: Vertex) -> None:
    """ Zero out pixels that are 'outside' of the start and finish region """

    def xxx(a, b, i, n):
        """ TODO """
        if i < n // 8:
            a = max(a, i)
        elif i > n * 7 // 8:
            b = min(b, i)
        return a, b

    rows, cols = image.shape
    l, r = 0, cols
    t, b = 0, rows
    l, r = xxx(l, r, start[0], cols)
    t, b = xxx(t, b, start[1], rows)
    l, r = xxx(l, r, finish[0], cols)
    t, b = xxx(t, b, finish[1], rows)

    if l > 2:
        image[:, : l - 1] = 0
    if r < cols - 2:
        image[:, r + 2 :] = 0
    if t > 2:
        image[: t - 1, :] = 0
    if b < rows - 2:
        image[b + 2 :, :] = 0


def setup(
    image: Image, start: Vertex, finish: Vertex, show_thinned: bool = False
) -> Tuple[Network, Vertex, Vertex]:
    """
    Load and display image.
    Calculate the thinned image and convert it to a network.
    Allow user to modify start and finish points.
    """

    gray = preprocess(image)

    gray = morph_open(gray, 3)

    thinned = cv.ximgproc.thinning(gray)

    # Allow user to modify
    step = 1
    Param.registered = []  # reset
    stop = Param(0, "", [ESCAPE])
    done = Param(0, "", " ")
    start_x = Param(start[0], "s", "f", maximum=image.shape[1] - 1, step=step)
    start_y = Param(start[1], "e", "d", maximum=image.shape[0] - 1, step=step)
    finish_x = Param(finish[0], "j", "l", maximum=image.shape[1] - 1, step=step)
    finish_y = Param(finish[1], "i", "k", maximum=image.shape[0] - 1, step=step)

    while not done.value:

        start = find_nearest(thinned, (start_x.value, start_y.value))
        finish = find_nearest(thinned, (finish_x.value, finish_y.value))

        if show_thinned:
            combined = (thinned // 4 * 3) + (gray // 4)
            # mask(combined, start, finish)
            tmp = cv.cvtColor(combined, cv.COLOR_GRAY2BGR)
        else:
            tmp = image.copy()

        overlay_start_finish(tmp, start, finish)
        show(tmp, None)

        Param.handle(cv.waitKey(0))

        if stop.value:
            sys.exit(0)

    mask(thinned, start, finish)
    network = Network2(image, thinned)
    return network, start, finish


def main(config) -> None:
    """ TODO """

    image_path, start, finish = config
    image = cv.imread(image_path)

    FrameWriter.write(image)

    # Setup network
    network, start, finish = setup(image, start, finish)

    # Find solution
    path = network.find_shortest_path(start, finish)

    # Draw solution
    draw_path(image, path)
    overlay_start_finish(image, start, finish)
    show(image)
    FrameWriter.write(image)


if __name__ == "__main__":
    configs = (
        ("mazes/maze1.jpg", (72, 437), (964, 508)),
        ("mazes/maze2.jpg", (23, 512), (987, 512)),
        ("mazes/maze3.jpg", (644, 8), (655, 860)),
        ("mazes/maze4.jpg", (650, 650), (660, 12)),
        ("mazes/maze5.jpg", (43, 332), (309, 330)),
        ("mazes/maze6.png", (0, 212), (440, 207)),
        ("mazes/maze6.png", (446, 437), (620, 0)),
        ("mazes/maze7.png", (13, 13), (499, 500)),
        ("mazes/maze8.png", (254, 16), (331, 309)),
    )

    for c in configs:
        main(c)
