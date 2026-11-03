"""Shared image renderer. Labels are not required to render input pixels."""
import random
import numpy as np
from PIL import Image, ImageDraw

PALETTES = {'red': ('#d33138','#e13c37','#c92d36'), 'blue': ('#2466d9','#286ee0','#245bc5')}

def render(pixels, colors, seed=0):
    rng = random.Random(seed)
    image = Image.new('RGB', (336,112), '#eeeeee')
    draw = ImageDraw.Draw(image)
    for i,(digit,color) in enumerate(zip(pixels,colors)):
        x = i * 112
        border = rng.choice(PALETTES[color])
        draw.rectangle((x+4,4,x+107,107), fill='white', outline=border, width=rng.choice((5,6,7)))
        size = rng.choice((76,80,84))
        digit = Image.fromarray(255 - np.asarray(digit, dtype=np.uint8)).convert('RGB')
        digit = digit.resize((size,size), Image.Resampling.NEAREST)
        dx,dy = rng.randint(-2,2),rng.randint(-2,2)
        image.paste(digit,(x+(112-size)//2+dx,(112-size)//2+dy))
    return image
