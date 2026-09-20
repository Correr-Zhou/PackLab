"""Stateless rendering helpers for packing visualizations."""

import numpy as np


def project_bbox_ndc_center(points, view_matrix, projection_matrix):
    # Project the bounding box center in NDC space for camera centering.
    try:
        view = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4, order="F")
        projection = np.asarray(projection_matrix, dtype=np.float64).reshape(4, 4, order="F")
        points = np.asarray(points, dtype=np.float64)
        homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1).T
        clip = projection @ view @ homo
        valid = np.abs(clip[3]) > 1e-8
        if not np.any(valid):
            return None
        ndc = clip[:2, valid] / clip[3:4, valid]
        ndc = ndc[:, np.all(np.isfinite(ndc), axis=0)]
        if ndc.size == 0:
            return None
        return np.array([(ndc[0].min() + ndc[0].max()) / 2.0, (ndc[1].min() + ndc[1].max()) / 2.0])
    except Exception:
        return None


def to_rgb_uint8(image):
    # Normalize any image-like input to RGB uint8.
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if image.size and image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        alpha = image[:, :, 3:4].astype(float) / 255.0
        image = (image[:, :, :3].astype(float) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return image[:, :, :3]


def resize_rgb_lanczos(image, height, width):
    from PIL import Image

    pil = Image.fromarray(to_rgb_uint8(image))
    resample = getattr(Image, "Resampling", Image).LANCZOS
    return np.asarray(pil.resize((width, height), resample=resample), dtype=np.uint8)


def resize_to_height(image, target_height):
    image = to_rgb_uint8(image)
    src_h, src_w = image.shape[:2]
    target_width = max(1, int(round(src_w * target_height / src_h)))
    return resize_rgb_lanczos(image, target_height, target_width)


def resize_mask_nearest(mask, height, width):
    mask = np.asarray(mask, dtype=bool)
    src_h, src_w = mask.shape[:2]
    y_idx = np.minimum((np.arange(height) * src_h / height).astype(int), src_h - 1)
    x_idx = np.minimum((np.arange(width) * src_w / width).astype(int), src_w - 1)
    return mask[y_idx][:, x_idx]


def enhance_rgb(image, brightness=1.0, contrast=1.0, gamma=1.0):
    f = to_rgb_uint8(image).astype(np.float32) / 255.0
    if gamma != 1.0:
        f = np.power(np.clip(f, 0.0, 1.0), gamma)
    f = f * 255.0
    f = (f - 127.5) * contrast + 127.5
    f = f * brightness
    return np.clip(f, 0, 255).astype(np.uint8)


def depth_background_mask(depth_buffer, height, width):
    try:
        depth = np.asarray(depth_buffer, dtype=np.float32).reshape(height, width)
    except Exception:
        return np.zeros((height, width), dtype=bool)
    return depth >= 0.9999


def segmentation_body_mask(segmentation, height, width, body_id):
    try:
        segmentation = np.asarray(segmentation, dtype=np.int64).reshape(height, width)
    except Exception:
        return np.zeros((height, width), dtype=bool)
    object_ids = segmentation & ((1 << 24) - 1)
    return object_ids == int(body_id)


def apply_ground_background(image, background_mask, view_matrix=None, projection_matrix=None):
    # Replace background pixels with a checkerboard ground plane.
    image = to_rgb_uint8(image).copy()
    mask = np.asarray(background_mask, dtype=bool)
    if not np.any(mask):
        return image
    height, width = image.shape[:2]
    yy, xx = np.nonzero(mask)
    world_x = world_y = None
    if view_matrix is not None and projection_matrix is not None:
        try:
            view = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4, order="F")
            projection = np.asarray(projection_matrix, dtype=np.float64).reshape(4, 4, order="F")
            inv = np.linalg.inv(projection @ view)
            ndc_x = (xx.astype(np.float64) + 0.5) / float(width) * 2.0 - 1.0
            ndc_y = 1.0 - (yy.astype(np.float64) + 0.5) / float(height) * 2.0
            near_clip = np.stack([ndc_x, ndc_y, np.full_like(ndc_x, -1.0), np.ones_like(ndc_x)], axis=0)
            far_clip = np.stack([ndc_x, ndc_y, np.ones_like(ndc_x), np.ones_like(ndc_x)], axis=0)
            nw = inv @ near_clip
            fw = inv @ far_clip
            nw = nw[:3] / nw[3:4]
            fw = fw[:3] / fw[3:4]
            ray = fw - nw
            denom = ray[2]
            hit = denom < -1e-8
            t = np.zeros_like(denom)
            t[hit] = -nw[2, hit] / denom[hit]
            hit &= t > 0.0
            hw = nw + ray * t[None, :]
            world_x = np.empty_like(ndc_x)
            world_y = np.empty_like(ndc_y)
            world_x[hit] = hw[0, hit]
            world_y[hit] = hw[1, hit]
            if np.any(~hit):
                fx, fy = _screen_ground_coordinates(xx[~hit], yy[~hit], width, height)
                world_x[~hit] = fx
                world_y[~hit] = fy
        except Exception:
            world_x = world_y = None
    if world_x is None or world_y is None:
        world_x, world_y = _screen_ground_coordinates(xx, yy, width, height)
    image[mask] = _checker_ground_colors(world_x, world_y, yy, height)
    return image


def _screen_ground_coordinates(x_pixels, y_pixels, width, height):
    x = np.asarray(x_pixels, dtype=np.float64)
    y = np.asarray(y_pixels, dtype=np.float64)
    u = (x + 0.5 - width / 2.0) / float(max(width, height))
    v = (y + 0.5) / float(height)
    distance_scale = 1.0 / np.clip(v + 0.08, 0.08, 1.2)
    return u * distance_scale * 1.8, (1.0 - v) * distance_scale * 1.2


def _checker_ground_colors(world_x, world_y, y_pixels, height):
    cell_size = 0.84
    x_cell = np.floor(np.asarray(world_x) / cell_size).astype(np.int64)
    y_cell = np.floor(np.asarray(world_y) / cell_size).astype(np.int64)
    checker = ((x_cell + y_cell) & 1).astype(bool)
    light = np.array([248, 250, 255], dtype=np.float32)
    dark = np.array([166, 194, 229], dtype=np.float32)
    colors = np.where(checker[:, None], light, dark)
    row_shading = 0.92 + 0.08 * (np.asarray(y_pixels, dtype=np.float32) / float(height))
    colors *= row_shading[:, None]
    return np.clip(colors, 0, 255).astype(np.uint8)


# ---------------- Image and Video Writing ----------------

def save_rgb_image(path, image):
    import imageio.v2 as imageio

    imageio.imwrite(path, to_rgb_uint8(image))


def save_rgba_image(path, image):
    import imageio.v2 as imageio

    image_uint8 = np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)
    imageio.imwrite(path, image_uint8)


def save_video(path, frames, fps):
    import imageio.v2 as imageio

    try:
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        return path
    except Exception as exc:
        fallback = str(path).rsplit(".", 1)[0] + ".gif"
        print(f"MP4 writing failed ({exc}); falling back to GIF: {fallback}")
        imageio.mimsave(fallback, frames, fps=fps)
        return fallback


def concat_step_images(buffer_image, placement_image, height_map_image, step_index):
    # Concatenate per-step panels: before placement, after placement, and height map.
    from PIL import Image, ImageDraw, ImageFont

    target_height = 1080
    panels = [resize_to_height(buffer_image, target_height), resize_to_height(placement_image, target_height), resize_to_height(height_map_image, target_height)]
    labels = ["Before Placement", "After Placement", "Height Map"]
    sep = 8
    title_h = 180
    label_h = 145
    bottom = 32
    total_w = sum(p.shape[1] for p in panels) + sep * (len(panels) - 1)
    total_h = title_h + target_height + label_h + bottom
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
    x = 0
    bounds = []
    for p in panels:
        w = p.shape[1]
        canvas[title_h:title_h + target_height, x:x + w] = p
        bounds.append((x, x + w))
        x += w + sep
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    title_font = _load_font(ImageFont, 118)
    label_font = _load_font(ImageFont, 82)
    _draw_centered(draw, (0, 0, total_w, title_h), f"Step {step_index}", title_font)
    for label, (x0, x1) in zip(labels, bounds):
        _draw_centered(draw, (x0, title_h + target_height, x1, title_h + target_height + label_h), label, label_font)
    return np.asarray(image, dtype=np.uint8)


def _load_font(image_font_module, size):
    for fp in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return image_font_module.truetype(fp, size)
        except Exception:
            pass
    try:
        return image_font_module.load_default(size=size)
    except TypeError:
        return image_font_module.load_default()


def _draw_centered(draw, box, text, font):
    x0, y0, x1, y1 = box
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)
    draw.text((x0 + (x1 - x0 - tw) / 2.0, y0 + (y1 - y0 - th) / 2.0), text, fill=(0, 0, 0), font=font)


def append_held_frame(frames, frame, hold_frames):
    # Repeat a frame to control video pacing.
    for _ in range(max(1, int(hold_frames))):
        frames.append(frame)


def concat_images_vertically(images):
    # Stack images vertically for all-step summaries.
    images = [to_rgb_uint8(img) for img in images]
    max_width = max(img.shape[1] for img in images)
    separator = np.full((8, max_width, 3), 255, dtype=np.uint8)
    padded = []
    for img in images:
        if img.shape[1] < max_width:
            pad = np.full((img.shape[0], max_width - img.shape[1], 3), 255, dtype=np.uint8)
            img = np.concatenate([img, pad], axis=1)
        padded.append(img)
        padded.append(separator)
    return np.concatenate(padded[:-1], axis=0)
