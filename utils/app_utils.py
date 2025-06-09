from PIL import Image
import io

def pil_to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()