#!/usr/bin/env python

from pathlib import Path

from PIL import Image  # pip install Pillow

# import imageio
# images = [imageio.imread(file_path) for file_path in sorted(dir_path.glob("*.png"))]
# imageio.mimsave("movie.gif", images, fps=fps)


def create_gif(
    gif_name="movie.gif",
    dir_name="frames",
    fps=10,
    loop=0,  # loop forever
):
    """
    https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    """
    images = [
        Image.open(str(file_path)) for file_path in sorted(Path(dir_name).glob("*.png"))
    ]

    images[0].save(
        gif_name,
        format="GIF",
        append_images=images[1:],
        save_all=True,
        duration=1000 // fps,
        loop=loop,
    )


if __name__ == "__main__":
    create_gif()
