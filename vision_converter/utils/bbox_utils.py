from ..formats.bounding_box import CenterAbsoluteBoundingBox, TopLeftAbsoluteBoundingBox, CornerAbsoluteBoundingBox, CenterNormalizedBoundingBox


def CenterNormalized_to_CornerAbsolute(bbox: CenterNormalizedBoundingBox, image_width: int, image_height: int) -> CornerAbsoluteBoundingBox:
    """Converts a bounding box from center-normalized coordinates (CenterNormalizedBoundingBox) to absolute corner coordinates (CornerAbsoluteBoundingBox).
    
    Args:
        bbox (CenterNormalizedBoundingBox): Object representing the bounding box with normalized center coordinates and dimensions (values between 0 and 1).
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.
    
    Returns:
        CornerAbsoluteBoundingBox: Bounding box with absolute coordinates of the top-left (x_min, y_min) and bottom-right (x_max, y_max) corners, in pixels.
    """
    x_center_abs = bbox.x_center * image_width
    y_center_abs = bbox.y_center * image_height
    width_abs = bbox.width * image_width
    height_abs = bbox.height * image_height

    x_min = round(x_center_abs - width_abs / 2)
    y_min = round(y_center_abs - height_abs / 2)
    x_max = round(x_center_abs + width_abs / 2)
    y_max = round(y_center_abs + height_abs / 2)

    return CornerAbsoluteBoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max
    )

def TopLeftAbsolute_to_CornerAbsolute(bbox: TopLeftAbsoluteBoundingBox) -> CornerAbsoluteBoundingBox:
    """Converts a bounding box defined by the top-left corner and dimensions (TopLeftAbsoluteBoundingBox) to absolute corner coordinates
    (CornerAbsoluteBoundingBox).

    Args:
        bbox (TopLeftAbsoluteBoundingBox): Object with (x_min, y_min) coordinates and (width, height) in pixels.

    Returns:
        CornerAbsoluteBoundingBox: Object containing absolute coordinates of (x_min, y_min, x_max, y_max) in pixels.
    """

    x_max_raw = bbox.x_min + bbox.width
    y_max_raw = bbox.y_min + bbox.height

    x_min = round(bbox.x_min)
    y_min = round(bbox.y_min)
    x_max = round(x_max_raw)
    y_max = round(y_max_raw)

    return CornerAbsoluteBoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max
    )

def CenterAbsolute_to_CornerAbsolute(bbox: CenterAbsoluteBoundingBox) -> CornerAbsoluteBoundingBox:
    """Converts a bounding box from center-absolute coordinates (CenterAbsoluteBoundingBox) to absolute corner coordinates (CornerAbsoluteBoundingBox).

    Args:
        bbox (CenterAbsoluteBoundingBox): Bounding box with absolute center coordinates (x_center, y_center) and size (width, height), in pixels.

    Returns:
        CornerAbsoluteBoundingBox: Absolute coordinates of the corners (x_min, y_min, x_max, y_max) in pixels.
    """
    x_min = round(bbox.x_center - bbox.width / 2)
    y_min = round(bbox.y_center - bbox.height / 2)
    x_max = round(bbox.x_center + bbox.width / 2)
    y_max = round(bbox.y_center + bbox.height / 2)
    
    return CornerAbsoluteBoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max
    )


def CornerAbsolute_to_CenterNormalized(bbox: CornerAbsoluteBoundingBox, image_width: int, image_height: int) -> CenterNormalizedBoundingBox:
    """Converts absolute corner coordinates (CornerAbsoluteBoundingBox) to normalized center-based format (CenterNormalizedBoundingBox).

    Args:
        bbox (CornerAbsoluteBoundingBox): Absolute bounding box with corners (x_min, y_min, x_max, y_max) in pixels.
        image_width (int): Original image width in pixels.
        image_height (int): Original image height in pixels.

    Returns:
        CenterNormalizedBoundingBox: Bounding box with normalized center coordinates and size (values from 0 to 1).
    """
    x_center = ((bbox.x_min + bbox.x_max) / 2) / image_width
    y_center = ((bbox.y_min + bbox.y_max) / 2) / image_height
    width = (bbox.x_max - bbox.x_min) / image_width
    height = (bbox.y_max - bbox.y_min) / image_height

    return CenterNormalizedBoundingBox(
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height
    )

def CornerAbsolute_to_TopLeftAbsolute(bbox: CornerAbsoluteBoundingBox) -> TopLeftAbsoluteBoundingBox:
    """Converts a bounding box in absolute corner coordinates (CornerAbsoluteBoundingBox) to top-left absolute and size format (TopLeftAbsoluteBoundingBox).

    Args:
        bbox (CornerAbsoluteBoundingBox): Bounding box with (x_min, y_min, x_max, y_max) in pixels.

    Returns:
        TopLeftAbsoluteBoundingBox: Bounding box with (x_min, y_min) coordinates and (width, height) in pixels.
    """
    x_min = bbox.x_min
    y_min = bbox.y_min
    width = bbox.x_max - bbox.x_min
    height = bbox.y_max - bbox.y_min

    return TopLeftAbsoluteBoundingBox(
        x_min=x_min,
        y_min=y_min,
        width=width,
        height=height
    )


def CornerAbsolute_to_CenterAbsolute(bbox: CornerAbsoluteBoundingBox) -> CenterAbsoluteBoundingBox:
    """Converts a bounding box with absolute corner coordinates (CornerAbsoluteBoundingBox) into center-absolute format (CenterAbsoluteBoundingBox).

    Args:
        bbox (CornerAbsoluteBoundingBox): Absolute coordinates of the corners (x_min, y_min, x_max, y_max) in pixels.

    Returns:
        CenterAbsoluteBoundingBox: Center coordinates (x_center, y_center) and size (width, height), in pixels.
    """
    x_center = (bbox.x_min + bbox.x_max) / 2
    y_center = (bbox.y_min + bbox.y_max) / 2
    width = bbox.x_max - bbox.x_min
    height = bbox.y_max - bbox.y_min
    
    return CenterAbsoluteBoundingBox(
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height
    )
