from PIL import Image, ImageOps

MAX_UPLOAD_MB = 15
MAX_PIXELS = 25_000_000
PHOTO_FORMATS = ("JPEG", "PNG", "WEBP")


class InvalidPhoto(ValueError):
    pass


def read_photo(uploaded):
    if uploaded.getbuffer().nbytes > MAX_UPLOAD_MB * 1024**2:
        raise InvalidPhoto(f"Choose a photo no larger than {MAX_UPLOAD_MB} MB.")

    try:
        with Image.open(uploaded, formats=PHOTO_FORMATS) as source:
            if source.width * source.height > MAX_PIXELS:
                raise InvalidPhoto("Choose a photo with at most 25 megapixels.")
            if getattr(source, "is_animated", False):
                raise InvalidPhoto("Choose a single-frame photo.")
            photo = ImageOps.exif_transpose(source).convert("RGB")
    except Image.DecompressionBombError:
        raise InvalidPhoto("Choose a photo with at most 25 megapixels.") from None
    except InvalidPhoto:
        raise
    except (OSError, SyntaxError, ValueError):
        raise InvalidPhoto("Choose a valid JPG, PNG, or WebP photo.") from None

    # Preserve the pixels and orientation without passing metadata to previews or exports.
    photo.info.clear()
    return photo
