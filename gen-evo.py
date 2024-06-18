import argparse
import datetime
import math
import pickle
import time
import os
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from cv2.typing import MatLike
import numpy as np

import Gartic
from Gartic import Point

executor = None

is_shutdown = False


def shutdown() -> None:
    global is_shutdown
    is_shutdown = True


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
    "-c", "--batch-count", type=int, help="Number of objects to draw", default=250
)
parser.add_argument(
    "-b",
    "--batch-size",
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
    "-t",
    "--threads",
    type=int,
    help="Number of threads to use for batch processing",
    default=4,
)

args = parser.parse_args()
if args.output == "":
    args.output = os.path.splitext(os.path.basename(args.input))[0] + ".gar"

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
img_height, img_width = img.shape[:2]

avg_col = cv2.mean(img)
avg_col = [int(i) for i in avg_col]
bg_color = get_closest_color((avg_col[0], avg_col[1], avg_col[2]), Gartic.colors)

best_img = np.zeros((img_height, img_width, 3), np.uint8)
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
    original_img: MatLike, evo_img: MatLike, roi_start: Point, roi_end: Point
) -> tuple[MatLike, None | Gartic.ToolShape]:
    best_shape: None | Gartic.Shape = None
    best_batch = evo_img.copy()
    best_diff = imgdiff(original_img, evo_img)
    h, w = original_img.shape[:2]

    for _ in range(int(args.batch_size / args.threads)):
        test_batch: MatLike = evo_img.copy()
        roi_size = roi_end - roi_start
        test_shape = Gartic.ToolShape.random(roi_size.x, roi_size.y)
        test_shape.a = test_shape.a + roi_start
        test_shape.b = test_shape.b + roi_start
        draw_shape(test_batch, test_shape)

        test_diff = imgdiff(original_img, test_batch)
        if test_diff < best_diff:
            best_batch = test_batch
            best_diff = test_diff
            best_shape = test_shape

        global is_shutdown
        if is_shutdown:
            return (best_batch, best_shape)

    return (best_batch, best_shape)


def threaded_batch_processing(
    original_img: MatLike,
    evo_img: MatLike,
    roi_start: Point,
    roi_end: Point,
):
    if args.threads == 1:
        return process_batch(original_img, evo_img, roi_start, roi_end)

    global executor
    executor = ThreadPoolExecutor(max_workers=args.threads)

    futures = [
        executor.submit(process_batch, original_img, evo_img, roi_start, roi_end)
        for _ in range(args.threads)
    ]
    best_diff = float("inf")
    best_batch = evo_img.copy()
    best_shape = None

    for future in as_completed(futures):
        batch_result, shape = future.result()
        batch_diff = imgdiff(original_img, batch_result)
        if batch_diff < best_diff:
            best_diff = batch_diff
            best_batch = batch_result
            best_shape = shape

    return best_batch, best_shape


last_write_out = 0
has_printed_total = False
avg_round_time = 0
while len(evolved.shapes) < args.batch_count:
    avg_round_time = (time.time() - start_time) / len(evolved.shapes)

    if len(evolved.shapes) > (args.batch_count / 10) and not has_printed_total:
        print(
            f"\rEstimated total time - {datetime.timedelta(seconds=math.floor(args.batch_count * avg_round_time))}"
            + " " * 50
        )
        has_printed_total = True

    time_left = datetime.timedelta(
        seconds=math.floor(avg_round_time * (args.batch_count - len(evolved.shapes)))
    )
    print(
        f"\r{len(evolved.shapes)}/{args.batch_count} Estimated time left - {time_left}"
        + " " * 10,
        end="",
    )

    diff_img = cv2.absdiff(img, best_img)
    diff_img = (
        255 * (diff_img - np.min(diff_img)) / (np.max(diff_img) - np.min(diff_img))
    ).astype(np.uint8)
    diff_img = diff_img.astype(np.uint8)
    _, binary = cv2.threshold(diff_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary)

    roi_start = Point(0, 0)
    roi_end = Point(img_width, img_height)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        roi_start = Point(x, y)
        roi_end = Point(x + w, y + h)

    best_img, best_shape = threaded_batch_processing(img, best_img, roi_start, roi_end)

    if best_shape is not None:
        evolved.add_shape(best_shape)

    if is_shutdown:
        break

    if len(evolved.shapes) > last_write_out + 25:
        last_write_out = len(evolved.shapes)
        cv2.imwrite(args.output + ".png", best_img)
        with open(args.output, "wb") as file:
            pickle.dump(evolved, file)

print()
cv2.imwrite(args.output + ".png", best_img)
with open(args.output, "wb") as file:
    pickle.dump(evolved, file)
print()
print(
    f"--- total time - {datetime.timedelta(seconds=math.floor(time.time() - start_time))} ---"
)
print("Difference score (lower is better):", round(imgdiff(img, best_img) / 100000, 2))
