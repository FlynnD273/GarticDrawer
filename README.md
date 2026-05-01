# ⚠️ This repository has migrated to [Codeberg](https://codeberg.org/FlynnD273/gartic-drawer)

# GarticDrawer
Draws an image in Gartic Phone

## Requirements

You may find it useful to create a [virtual environment](https://docs.python.org/3/library/venv.html) to make it easier to organise the installed modules. 

### Dependencies

To install dependencies, run `pip install -r ./requirements.txt` in the root project directory. 

### Playwright

To initialise Playwright, run `playwright install firefox`.

Set up the Firefox instance with `python ./setup.py`. This will take you to the uBlock Origin installation page. Accept and install the extension so Gartic Phone loads significantly faster. You can skip this if you object to the use of adblockers, but it will break `python ./draw.py --demo`. You can still run it in manual mode though. 

## Running the Script

### Image Generation

To generate images to draw, run `python ./gen-evo.py <input file path>`. This will output a shape file to `<image name>.gar` by default. Run `python ./gen-evo.py --help` for more information on command-line flags.

To cancel generation midway through, press CTRL-C. This will export what it's generated so far and exit. 

### Drawing in Gartic

In most cases, drawing will only occur when the browser is focused.

Run `python ./draw.py --demo <input file path>` to start a single-player instance of Gartic Phone and draw the image. Run `python ./draw.py` to run it in interactive mode (if you wanted to use this in an actual round, for example)

In interactive mode, enter the path to the .gar file to start drawing. Only do this once a drawing round has started. Currently there is no way to stop the drawing after it's started.

### Creating a GIF

You must have [gifski](https://gif.ski/) installed and added to your PATH.

Run `python ./make-gif.py <path to .gar>` to create a GIF. \
Run `python ./make-gif.py -h` for information on the available flags.

## Example Drawings

Examples were generated with a `batch-count` of 750 and a `batch-size` of 450,000. All other settings were default. 

![Dusa from Hades](https://github.com/FlynnD273/GarticDrawer/blob/main/Examples/dusa.png)
![Dusa from Hades](https://github.com/FlynnD273/GarticDrawer/blob/main/Examples/dusa.gif)

![Dusa from Hades](https://github.com/FlynnD273/GarticDrawer/blob/main/Examples/kirby.png)
![Dusa from Hades](https://github.com/FlynnD273/GarticDrawer/blob/main/Examples/darth2.png)
