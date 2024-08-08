import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pygame import image as pyg_image
from pygame.transform import flip as img_flip

_BASE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

SPRITES_PATH = str(_BASE_DIR / "sprites")

def _load_sprite(filename, convert, alpha=True):
    img = pyg_image.load(f"{SPRITES_PATH}/{filename}")
    return (
        img.convert_alpha() if convert and alpha else img.convert() if convert else img
    )

def load_images(
    convert,
) -> Dict[str, Any]:
    """Loads and returns the image assets of the game."""
    images = {}

    try:
        # Sprites with the number for the score display:
        images["numbers"] = tuple(
            [_load_sprite(f"{n}.png", convert=convert, alpha=True) for n in range(10)]
        )

        # Sprite for the base (ground):
        images["base"] = _load_sprite("base.png", convert=convert, alpha=True)

        # Background sprite:
        
        images["background"] = _load_sprite(
                "background-day.png", convert=convert, alpha=False
            )

        # Bird sprites:
        images["player"] = (
            _load_sprite("yellowbird-upflap.png", convert=convert, alpha=True),
            _load_sprite("yellowbird-midflap.png", convert=convert, alpha=True),
            _load_sprite("yellowbird-downflap.png", convert=convert, alpha=True),
        )

        # Pipe sprites:
        pipe_sprite = _load_sprite(
            "pipe-green.png", convert=convert, alpha=True
        )
        images["pipe"] = (img_flip(pipe_sprite, False, True), pipe_sprite)
    except FileNotFoundError as ex:
        raise FileNotFoundError(
            "Can't find the sprites folder! No such file or"
            f" directory: {SPRITES_PATH}"
        ) from ex

    return images
#