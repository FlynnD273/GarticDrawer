import random
from playwright.sync_api import Page
from dataclasses import dataclass

colors = [
    (0, 0, 0),
    (102, 102, 102),
    (205, 80, 0),
    (255, 255, 255),
    (170, 170, 170),
    (255, 201, 38),
    (32, 116, 1),
    (0, 0, 153),
    (18, 65, 150),
    (60, 176, 17),
    (19, 0, 255),
    (41, 120, 255),
    (28, 112, 176),
    (78, 0, 153),
    (87, 90, 203),
    (38, 193, 255),
    (143, 0, 255),
    (168, 175, 254),
]
thicknesses = [2, 6, 10, 14, 18]
opacities = [i / 10 for i in range(1, 11)]


def clamp(i: int, minimum: int, maximum: int):
    return min(max(i, minimum), maximum)


class Shape:
    colorIndex: int
    thicknessIndex: int
    opacityIndex: int

    def __init__(self, colorIndex: int, thicknessIndex: int, opacityIndex: int) -> None:
        self.colorIndex = colorIndex
        self.thicknessIndex = thicknessIndex
        self.opacityIndex = opacityIndex

    def __copy__(self):
        return Shape(self.colorIndex, self.thicknessIndex, self.opacityIndex)

    def __mul__(self, b):
        return self

    @staticmethod
    def random():
        color = random.randint(0, len(colors) - 1)
        thickness = random.randint(0, len(thicknesses) - 1)
        opacity = random.randint(0, len(opacities) - 1)
        return Shape(color, thickness, opacity)


@dataclass
class Point:
    x: float
    y: float

    def __add__(self, b):
        return Point(self.x + b.x, self.y + b.y)

    def __sub__(self, b):
        return self + (-b)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __mul__(self, b):
        match b:
            case int() | float() as bnum:
                return Point(self.x * bnum, self.y * bnum)
            case Point() as bpoint:
                return Point(self.x * bpoint.x, self.y * bpoint.y)

    def __truediv__(self, b):
        return Point(self.x / b.x, self.y / b.y)


class ToolShape(Shape):
    a: Point
    b: Point
    tool: str

    def __init__(
        self,
        colorIndex: int,
        thicknessIndex: int,
        opacityIndex: int,
        a: Point,
        b: Point,
        tool: str,
    ) -> None:
        super().__init__(colorIndex, thicknessIndex, opacityIndex)
        self.a = a
        self.b = b
        self.tool = tool

    def __copy__(self):
        return ToolShape(
            self.colorIndex,
            self.thicknessIndex,
            self.opacityIndex,
            self.a,
            self.b,
            self.tool,
        )

    def __mul__(self, b):
        return ToolShape(
            self.colorIndex,
            self.thicknessIndex,
            self.opacityIndex,
            self.a * b,
            self.b * b,
            self.tool,
        )

    def __add__(self, b):
        return ToolShape(
            self.colorIndex,
            self.thicknessIndex,
            self.opacityIndex,
            self.a + b,
            self.b + b,
            self.tool,
        )

    @staticmethod
    def random(w: int, h: int):
        a = Point(random.random() * (w - 1), random.random() * (h - 1))
        b = Point(random.random() * (w - 1), random.random() * (h - 1))
        tool = tools[random.randint(0, len(tools) - 1)]
        shape = Shape.random()
        return ToolShape(
            shape.colorIndex, shape.thicknessIndex, shape.opacityIndex, a, b, tool
        )

    def draw(self, page: Page):
        set_shape(page, self)
        set_tool(page, self.tool)
        click_drag(page, self.a, self.b)


class Image:
    shapes: list[ToolShape]
    height: float
    width: float

    def __init__(self, width: float, height: float):
        self.shapes = []
        self.height = height
        self.width = width

    def add_shape(self, shape: ToolShape):
        self.shapes.append(shape)

    def scale_to(self, img_w: float, img_h: float) -> float:
        scale = min(img_w / self.width, img_h / self.height)
        self.height *= scale
        self.width *= scale
        for i in range(len(self.shapes)):
            self.shapes[i] = self.shapes[i] * scale
        return scale


def click_point(page: Page, point: Point) -> None:
    page.locator("canvas").nth(3).click(position={"x": point.x, "y": point.y})


def click_drag(page: Page, pointA: Point, pointB) -> None:
    canvas = page.locator("canvas").nth(3)
    bounds = canvas.bounding_box()

    if bounds is None:
        return

    page.mouse.move(bounds["x"] + pointA.x, bounds["y"] + pointA.y)
    page.mouse.down()
    page.mouse.move(bounds["x"] + pointB.x, bounds["y"] + pointB.y)
    page.mouse.up()


def set_color(page: Page, i: int) -> None:
    elem = page.locator(f".colorslist").first.locator("div").all()[i]
    if "sel" not in (elem.get_attribute("class") or ""):
        elem.click()


def set_thickness(page: Page, i: int) -> None:
    elem = page.locator(".thickness").all()[i].first
    if "sel" not in (elem.get_attribute("class") or ""):
        elem.click()


def set_opacity(page: Page, i: int) -> None:
    page.get_by_role("slider").fill(str(opacities[i]))


PEN = "pen"
ERASER = "ers"
RECT_HOLLOW = "reb"
ELLIPSE_HOLLOW = "ellb"
RECT = "rec"
ELLIPSE = "ell"
LINE = "lin"
BUCKET = "fil"
UNDO = "undo"
REDO = "redo"

tools = [PEN, RECT_HOLLOW, ELLIPSE_HOLLOW, RECT, ELLIPSE]


def set_tool(page: Page, tool: str) -> None:
    elem = page.locator("." + tool)
    if "sel" not in (elem.get_attribute("class") or ""):
        elem.click()


def set_shape(page: Page, shape: Shape) -> None:
    set_color(page, shape.colorIndex)
    set_thickness(page, shape.thicknessIndex)
    set_opacity(page, shape.opacityIndex)
