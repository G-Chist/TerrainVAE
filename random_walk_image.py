import numpy as np
from PIL import Image, ImageDraw
import os
import random


def random_walk_image(brush_size=8, steps=3000):
    assert 8 <= brush_size <= 48, "Brush size must be between 8 and 48."

    size = 101
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    x, y = size // 2, size // 2

    count = 0
    while count < steps:
        # draw filled black circle
        r = brush_size // 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0))

        # random direction step
        dx, dy = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        x = max(0, min(size - 1, x + dx))
        y = max(0, min(size - 1, y + dy))

        count += 1

    os.makedirs("inputs", exist_ok=True)
    out_path = os.path.join("inputs", f"random_example_{brush_size}_{steps}.png")
    img.save(out_path, format="PNG")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    random_walk_image(brush_size=16, steps=5000)
