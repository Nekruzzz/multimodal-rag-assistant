"""Generate two small sample images (with caption sidecars) for the demo corpus."""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "data" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def soil_moisture_chart():
    img = Image.new("RGB", (480, 300), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([50, 20, 450, 260], outline="black")
    values = [42, 39, 36, 33, 29, 27, 41]  # percent, one per day; irrigation on day 6
    pts = [(50 + i * 400 / 6, 260 - v * 4.8) for i, v in enumerate(values)]
    d.line(pts, fill=(30, 110, 200), width=3)
    d.line([(50, 260 - 30 * 4.8), (450, 260 - 30 * 4.8)], fill=(200, 60, 60), width=1)  # 30% threshold
    d.text((55, 24), "Soil moisture (%) over 7 days - threshold 30%", fill="black")
    img.save(OUT / "soil_moisture_week.png")
    (OUT / "soil_moisture_week.png.txt").write_text(
        "Line chart of soil moisture over seven days. Moisture drops below the 30 percent "
        "irrigation threshold on day five and recovers after irrigation on day six.\n"
    )


def leaf_lesions():
    img = Image.new("RGB", (480, 300), (225, 240, 215))
    d = ImageDraw.Draw(img)
    d.ellipse([90, 40, 390, 260], fill=(70, 150, 70), outline=(30, 90, 30), width=3)
    for cx, cy, r in [(180, 140, 14), (250, 110, 10), (300, 180, 16), (220, 200, 9)]:
        d.ellipse([cx - r - 5, cy - r - 5, cx + r + 5, cy + r + 5], fill=(220, 200, 80))  # yellow halo
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(130, 85, 40))  # brown lesion
    img.save(OUT / "leaf_blight_example.png")
    (OUT / "leaf_blight_example.png.txt").write_text(
        "Illustration of a green leaf with several brown blight lesions, each surrounded by "
        "a yellow halo, spread across the leaf surface.\n"
    )


if __name__ == "__main__":
    soil_moisture_chart()
    leaf_lesions()
    print("wrote sample images to", OUT)
