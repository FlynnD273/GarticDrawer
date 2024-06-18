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
parser.add_argument(
    "--top",
    type=int,
    help="How many shapes to keep from each batch",
    default=1,
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
    original_img: MatLike,
    evo_img: MatLike,
    topn: int,
) -> list[tuple[Gartic.ToolShape, float]]:
    top_shapes: list[tuple[Gartic.ToolShape, float]] = []
    h, w = original_img.shape[:2]

    for _ in range(int(args.batch_size / args.threads)):
        test_batch: MatLike = evo_img.copy()
        test_shape = Gartic.ToolShape.random(w, h)
        draw_shape(test_batch, test_shape)

        test_diff = imgdiff(original_img, test_batch)
        if len(top_shapes) == 0:
            top_shapes.append((test_shape, test_diff))
        elif len(top_shapes) < topn or test_diff < top_shapes[0][1]:
            insert_index = 0
            while (
                insert_index < len(top_shapes)
                and test_diff < top_shapes[insert_index][1]
            ):
                insert_index += 1
            top_shapes.insert(insert_index, (test_shape, test_diff))

        if len(top_shapes) > topn:
            del top_shapes[0]

        global is_shutdown
        if is_shutdown:
            return top_shapes

    return top_shapes


def threaded_batch_processing(
    original_img: MatLike,
    evo_img: MatLike,
    topn: int,
    curr_diff: float,
) -> list[Gartic.ToolShape]:
    thread_shapes: list[list[tuple[Gartic.ToolShape, float]]] = []
    final_shapes: list[Gartic.ToolShape] = []

    if args.threads == 1:
        thread_shapes = [process_batch(original_img, evo_img, topn)]
    else:
        global executor
        executor = ThreadPoolExecutor(max_workers=args.threads)

        futures = [
            executor.submit(process_batch, original_img, evo_img, topn)
            for _ in range(args.threads)
        ]

        for future in as_completed(futures):
            l = future.result()
            if len(l) > 0:
                thread_shapes.append(l)

    img = evo_img.copy()

    while len(final_shapes) < topn:
        best_shape = thread_shapes[0][-1][0]
        best_diff = thread_shapes[0][-1][1]
        thread_index = 0

        for i in range(1, len(thread_shapes)):
            thread_shape = thread_shapes[i][-1][0]
            thread_diff = thread_shapes[i][-1][1]

            if thread_diff < best_diff:
                best_diff = thread_diff
                best_shape = thread_shape
                thread_index = i

        del thread_shapes[thread_index][-1]
        if len(thread_shapes[thread_index]) == 0:
            del thread_shapes[thread_index]

        draw_shape(img, best_shape)

        if imgdiff(img, original_img) < curr_diff:
            final_shapes.append(best_shape)
        else:
            break

    return final_shapes


last_write_out = 0
has_printed_total = False
avg_round_time = 0
round_count = 0
# This is a pretty bad estimate
# 92.4592 + 0.3707 * Total + -10.1489 * Top + -0.0000 * Total^2 + -0.0079 * Total Top + 0.1996 * Top^2
# est_rounds = (
#     92.4592 + 0.3707 * args.batch_count + -10.1489 * args.top + 0.1996 * args.top**2
# )
est_rounds = args.batch_count / args.top
while len(evolved.shapes) < args.batch_count:
    round_count += 1
    avg_round_time = (time.time() - start_time) / round_count

    if len(evolved.shapes) > (args.batch_count / 10) and not has_printed_total:
        print(
            f"\rEstimated total time - {datetime.timedelta(seconds=math.floor(est_rounds * avg_round_time))}"
            + " " * 50
        )
        has_printed_total = True

    if round_count > est_rounds:
        time_left = "NaN"
    else:
        time_left = datetime.timedelta(
            seconds=math.floor(avg_round_time * (est_rounds - round_count))
        )
    print(
        f"\r{len(evolved.shapes)}/{args.batch_count} Estimated time left - {time_left}"
        + " " * 10,
        end="",
    )

    curr_diff = imgdiff(img, best_img)

    # count = len(evolved.shapes) / args.batch_count
    # count = min(count * 2, 1)
    # count *= args.top
    # count = max(int(count), 1)
    count = args.top
    best_shapes = threaded_batch_processing(img, best_img, count, curr_diff)

    if len(best_shapes) > 0:
        for s in best_shapes:
            draw_shape(best_img, s)
            evolved.add_shape(s)

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
