import argparse
import datetime
import math
import pickle
import time
import os
import signal
import sys

import cv2
from cv2.typing import MatLike


import Gartic
from Gartic import Point


def shutdown() -> None:
    if args.use_gpu:
        cv2.imwrite("evolution.png", np.asnumpy(best_img)) # type: ignore
    else:
        cv2.imwrite("evolution.png", best_img)
    if args.output == "":
        args.output = os.path.splitext(os.path.basename(args.input))[0] + ".gar"
    with open(args.output, "wb") as file:
        pickle.dump(evolved, file)

    print()
    print(
        f"--- total time - {datetime.timedelta(seconds=math.floor(time.time() - start_time))} ---"
    )
    sys.exit(0)

signal.signal(signal.SIGINT, lambda _, b: shutdown())

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="The input image path")
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Path to export the .gar file to",
    default="",
)
parser.add_argument(
    "-c", "--count", type=int, help="Number of objects to draw", default=250
)
parser.add_argument(
    "-b",
    "--batch",
    type=int,
    help="Number of objects to use for each batch",
    default=10000,
)
parser.add_argument(
    "--height",
    type=int,
    help="Vertical resolution of image to work with. Smaller is faster and takes less memory, larger is more detailed",
    default=200,
)
parser.add_argument(
    "--use-gpu",
    action="store_true",
    help="Use CUDA and GPU acceleration instead of CPU",
)

args = parser.parse_args()

if args.use_gpu:
    import cupy as np
else:
    import numpy as np

start_time = time.time()


# From https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle
def draw_shape(img: MatLike, shape: Gartic.ToolShape) -> None:
    overlay = img.copy()
    match shape.tool:
        case Gartic.PEN:
            cv2.line(
                overlay,
                (int(shape.a.x), int(shape.a.y)),
                (int(shape.b.x), int(shape.b.y)),
                Gartic.colors[shape.colorIndex],
                max(
                    int(Gartic.thicknesses[shape.thicknessIndex] * args.height / 400), 1
                ),
                lineType=cv2.LINE_AA,
            )

        case Gartic.ELLIPSE_HOLLOW | Gartic.ELLIPSE:
            centerx = int((shape.a.x + shape.b.x) / 2)
            centery = int((shape.a.y + shape.b.y) / 2)
            center_coordinates = (centerx, centery)
            sizex = int(abs(shape.a.x - shape.b.x) / 2)
            sizey = int(abs(shape.a.y - shape.b.y) / 2)
            axes_lengths = (sizex, sizey)
            color = Gartic.colors[shape.colorIndex]

            if shape.tool == Gartic.ELLIPSE_HOLLOW:
                thickness = max(
                    int(Gartic.thicknesses[shape.thicknessIndex] * args.height / 400), 1
                )
            else:
                thickness = -1

            cv2.ellipse(
                overlay,
                center_coordinates,
                axes_lengths,
                0,
                0,
                360,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )

        case Gartic.RECT_HOLLOW | Gartic.RECT:
            if shape.tool == Gartic.RECT_HOLLOW:
                thickness = max(
                    int(Gartic.thicknesses[shape.thicknessIndex] * args.height / 400), 1
                )
            else:
                thickness = -1

            cv2.rectangle(
                overlay,
                (int(shape.a.x), int(shape.a.y)),
                (int(shape.b.x), int(shape.b.y)),
                Gartic.colors[shape.colorIndex],
                thickness,
                lineType=cv2.LINE_AA,
            )

    cv2.addWeighted(
        overlay,
        Gartic.opacities[shape.opacityIndex],
        img,
        1 - Gartic.opacities[shape.opacityIndex],
        0,
        img,
    )
    del overlay


def imgdiff(a: MatLike, b: MatLike) -> float:
    absdiff = cv2.absdiff(a, b)
    diff = np.sum(absdiff)
    return diff  # type: ignore


def rgb_dist(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return (
        (a[0] - b[0]) * (a[0] - b[0])
        + (a[1] - b[1]) * (a[1] - b[1])
        + (a[2] - b[2]) * (a[2] - b[2])
    )


def get_closest_color(
    color: tuple[int, int, int], colors: list[tuple[int, int, int]]
) -> int:
    mindex = 0
    min_dist = rgb_dist(color, colors[0])

    for i in range(1, len(colors)):
        curr_dist = rgb_dist(color, colors[i])
        if curr_dist < min_dist:
            min_dist = curr_dist
            mindex = i

    return mindex


img = cv2.imread(args.input)
img_height, img_width = img.shape[:2]
img_scale = args.height / img_height
img = cv2.resize(
    img,
    (math.floor(img_width * img_scale), math.floor(img_height * img_scale)),
    interpolation=cv2.INTER_LANCZOS4,
)
img_width, img_height = img.shape[:2]

avg_col = cv2.mean(img)
avg_col = [int(i) for i in avg_col]
bg_color = get_closest_color((avg_col[0], avg_col[1], avg_col[2]), Gartic.colors)

best_img = np.zeros((img_width, img_height, 3), np.uint8)
best_img[::] = Gartic.colors[bg_color]
evolved = Gartic.Image(img_width, img_height)
evolved.add_shape(
    Gartic.ToolShape(
        bg_color,
        len(Gartic.thicknesses) - 1,
        len(Gartic.opacities) - 1,
        Point(10, 10),
        Point(10, 10),
        Gartic.BUCKET,
    )
)


def process_batch(
    original_img: MatLike,
    evo_img: MatLike,
) -> tuple[MatLike, None | Gartic.ToolShape]:
    best_shape: None | Gartic.Shape = None
    best_batch = evo_img.copy()
    best_diff = imgdiff(original_img, evo_img)
    h, w = original_img.shape[:2]

    for _ in range(args.batch):
        test_batch: MatLike = evo_img.copy()
        test_shape = Gartic.ToolShape.random(w, h)
        draw_shape(test_batch, test_shape)

        test_diff = imgdiff(original_img, test_batch)
        if test_diff < best_diff:
            best_batch = test_batch
            best_diff = test_diff
            best_shape = test_shape

    return (best_batch, best_shape)


avg_step_time = 0
for j in range(args.count):
    avg_step_time = (time.time() - start_time) / (j + 1)

    if j == (args.count / 10):
        print(
            f"\rEstimated total time - {datetime.timedelta(seconds=math.floor(avg_step_time * args.count))}"
            + " " * 50
        )

    time_left = avg_step_time * (args.count - j)
    print(
        f"\r({len(evolved.shapes)}) {j + 1}/{args.count} Estimated time left - {datetime.timedelta(seconds=math.floor(time_left))}"
        + " " * 10,
        end="",
    )

    best_batch, best_shape = process_batch(img, best_img)

    if best_shape is not None:
        best_img = best_batch.copy()
        evolved.add_shape(best_shape)

    if (j + 1) % 25 == 0:
        if args.use_gpu:
            cv2.imwrite("evolution.png", np.asnumpy(best_img)) # type: ignore
        else:
            cv2.imwrite("evolution.png", best_img)

print()
shutdown()
