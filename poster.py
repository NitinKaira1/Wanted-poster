import os
import random
import textwrap
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

POSTER_W, POSTER_H = 600, 850
OUTPUT_DIR = "output"

COLOR_BG         = (210, 180, 120)
COLOR_DARK       = (60,  35,  10)
COLOR_RED        = (160, 20,  20)
COLOR_BORDER_OUT = (80,  45,  10)
COLOR_BORDER_IN  = (140, 90,  30)


def load_font(size):
    candidates = [
        "C:/Windows/Fonts/georgia.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def add_aged_texture(img):
    w, h = img.size
    draw = ImageDraw.Draw(img)

    for _ in range(1500):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        darkness = random.randint(0, 40)
        r = max(0, COLOR_BG[0] - darkness)
        g = max(0, COLOR_BG[1] - darkness)
        b = max(0, COLOR_BG[2] - darkness)
        draw.point((x, y), fill=(r, g, b))

    vignette = Image.new("L", (w, h), 0)
    vd = ImageDraw.Draw(vignette)
    pad = int(min(w, h) * 0.15)
    vd.ellipse([pad, pad, w - pad, h - pad], fill=255)
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=min(w, h) // 4))

    import numpy as np
    arr   = np.array(img).astype(float)
    vmask = np.array(vignette).astype(float) / 255.0
    strength = 0.45
    factor   = strength + (1 - strength) * vmask
    arr      = arr * factor[:, :, None]
    arr      = arr.clip(0, 255).astype("uint8")

    return Image.fromarray(arr)


def wrap_text_to_width(draw, text, font, max_width):
    """
    Wrap text so that each rendered line fits within max_width pixels.
    Much more accurate than estimating via char count.
    """
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = (current + " " + word).strip()
        w = draw.textlength(test, font=font)
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines if lines else [text]


def fit_font_to_width(text, max_width, max_size, min_size=14):
    """
    Shrink font size until the text fits within max_width pixels.
    Used for single-line elements like the alias.
    """
    size = max_size
    while size >= min_size:
        font = load_font(size)
        # Create a temporary image just to measure
        tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        if tmp.textlength(text, font=font) <= max_width:
            return font, size
        size -= 2
    return load_font(min_size), min_size


def draw_centered_text(draw, y, text, font, color, poster_w,
                       max_width=None, spacing=8):
    """Draw text centered horizontally, wrapping to max_width if given."""
    if max_width:
        lines = wrap_text_to_width(draw, text, font, max_width)
    else:
        lines = [text]

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x  = (poster_w - tw) // 2
        draw.text((x, y), line, font=font, fill=color)
        y += th + spacing

    return y


def build_poster(face_image_path: str, roast: dict, output_path: str = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "wanted_poster.jpg")

    # Background gradient
    poster = Image.new("RGB", (POSTER_W, POSTER_H), COLOR_BG)
    draw   = ImageDraw.Draw(poster)

    for y in range(POSTER_H):
        ratio = abs(y / POSTER_H - 0.5) * 2
        shade = int(15 * ratio)
        draw.line([(0, y), (POSTER_W, y)],
                  fill=(max(0, COLOR_BG[0] - shade),
                        max(0, COLOR_BG[1] - shade),
                        max(0, COLOR_BG[2] - shade)))

    # Borders
    draw.rectangle([8, 8, POSTER_W - 9, POSTER_H - 9],
                   outline=COLOR_BORDER_OUT, width=6)
    draw.rectangle([18, 18, POSTER_W - 19, POSTER_H - 19],
                   outline=COLOR_BORDER_IN, width=2)

    # Fonts
    font_wanted = load_font(90)
    font_dead   = load_font(36)
    font_crime  = load_font(20)
    font_bounty = load_font(52)
    font_small  = load_font(17)

    # WANTED header
    y = 28
    y = draw_centered_text(draw, y, "WANTED", font_wanted, COLOR_RED, POSTER_W)
    y = draw_centered_text(draw, y - 8, "DEAD OR ALIVE", font_dead, COLOR_DARK, POSTER_W)
    y += 10
    draw.line([(40, y), (POSTER_W - 40, y)], fill=COLOR_DARK, width=2)
    y += 14

    # Face photo
    face_img = Image.open(face_image_path).convert("RGB")
    face_img.thumbnail((260, 290))

    border = 4
    framed = Image.new("RGB",
                       (face_img.width + border * 2, face_img.height + border * 2),
                       COLOR_DARK)
    framed.paste(face_img, (border, border))
    framed = ImageEnhance.Color(framed).enhance(0.8)

    face_x = (POSTER_W - framed.width) // 2
    poster.paste(framed, (face_x, y))
    y += framed.height + 16

    # Alias — auto-shrink font so it always fits on one or two lines
    draw.line([(40, y), (POSTER_W - 40, y)], fill=COLOR_DARK, width=2)
    y += 12

    alias      = roast.get("alias", "UNKNOWN SUSPECT").upper()
    alias_text = f'A.K.A  "{alias}"'
    max_text_w = POSTER_W - 60  # 30px margin each side

    # Try to fit on one line first (shrink from 38 down)
    alias_font, alias_size = fit_font_to_width(alias_text, max_text_w, max_size=38)

    # If even at min size it's too long, wrap at size 28
    if alias_size <= 14:
        alias_font = load_font(28)

    y = draw_centered_text(draw, y, alias_text, alias_font, COLOR_DARK, POSTER_W,
                           max_width=max_text_w, spacing=6)
    y += 6

    # Crime
    y = draw_centered_text(draw, y, "CRIME:", font_crime, COLOR_RED, POSTER_W)
    y = draw_centered_text(draw, y,
                           roast.get("crime", "Being suspiciously ordinary"),
                           font_crime, COLOR_DARK, POSTER_W,
                           max_width=max_text_w, spacing=6)
    y += 8

    # Description
    draw.line([(40, y), (POSTER_W - 40, y)], fill=COLOR_DARK, width=1)
    y += 8
    y = draw_centered_text(draw, y,
                           roast.get("description", "Dangerous. Approach with snacks."),
                           font_small, COLOR_DARK, POSTER_W,
                           max_width=max_text_w, spacing=5)
    y += 8

    # Bounty
    draw.line([(40, y), (POSTER_W - 40, y)], fill=COLOR_DARK, width=2)
    y += 12
    y = draw_centered_text(draw, y, "REWARD", font_dead, COLOR_DARK, POSTER_W)
    y = draw_centered_text(draw, y, f"${roast.get('bounty', 1000):,}",
                           font_bounty, COLOR_RED, POSTER_W)
    y = draw_centered_text(draw, y, "FOR CAPTURE  •  DEAD OR ALIVE",
                           font_small, COLOR_DARK, POSTER_W)

    # Footer
    footer_y = POSTER_H - 48
    draw.line([(40, footer_y), (POSTER_W - 40, footer_y)], fill=COLOR_DARK, width=2)
    draw_centered_text(draw, footer_y + 8,
                       "REPORT TO YOUR LOCAL SHERIFF  •  NO QUESTIONS ASKED",
                       font_small, COLOR_DARK, POSTER_W)

    # Aged texture
    poster = add_aged_texture(poster)

    poster.save(output_path, quality=95)
    print(f"[poster] Saved → {output_path}")
    return output_path


if __name__ == "__main__":
    from roast import get_roast
    build_poster("output/captured_face.jpg", get_roast())
