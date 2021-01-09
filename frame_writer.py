from pathlib import Path

import cv2 as cv


def delete_contents(dir_path):
    for path in dir_path.glob("*"):
        path.unlink()


class FrameWriter:
    frame = 0

    @classmethod
    def write(cls, image):
        dir_path = Path("frames")
        if cls.frame == 0:
            delete_contents(dir_path)
        cls.frame += 1
        path = dir_path / f"frame{cls.frame:04d}.png"
        cv.imwrite(str(path), image)
