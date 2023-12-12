import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms
import tqdm
from PIL import Image

from data import FillShape, localize
from net import DFPmodel
from postprocess import clean_room, fill_holes
from util import bchw2colormap, boundary2rgb, room2rgb


def inference(args):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(module)s - %(levelname)s"
        + "- %(funcName)s: %(lineno)d - %(message)s",
    )

    if not os.path.isdir(args.dst_dir):
        logging.info(f"Creating dst_dir: {args.dst_dir}")
        os.makedirs(args.dst_dir)

    device = args.device
    logging.info(f"Using device: {device}")
    logging.info(f"Loading weights from: {args.weights_path}")
    model = DFPmodel(
        room_channels=args.room_channels, boundary_channels=args.boundary_channels
    )
    model.load_state_dict(torch.load(args.weights_path))
    model.to(device)
    model.eval()
    logging.info(
        f"Instantiated model with room_channels = {args.room_channels},"
        + f" boundary_channels = {args.boundary_channels}."
    )

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    src_images = []
    for ext in ["png", "jpg", "jpeg", "webp"]:
        src_images.extend(list(src_dir.glob("*." + ext)))
    logging.info(f"Found {len(src_images)} images in src_dir: {src_dir}")

    logging.info(f"Begin inference with dst_dir: {args.dst_dir}")
    logging.info(f"Use postprocessing: {args.postprocess}")
    fill_shape_tsfm = FillShape(type="center")
    to_tensor = torchvision.transforms.ToTensor()
    with torch.inference_mode():
        for i, file in tqdm.tqdm(enumerate(src_images), total=len(src_images)):
            image = np.asarray(Image.open(file))
            h, w, c = image.shape
            tmp_bd, tmp_room = np.zeros((h, w)), np.zeros((h, w))

            image, _, _ = localize(image, tmp_bd, tmp_room)
            image, _, _ = fill_shape_tsfm(image, tmp_bd, tmp_room)
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
            image = to_tensor(image.astype(np.float32) / 255.0).unsqueeze(0)
            image = image.to(device)

            logits_r, logits_cw = model(image)
            predboundary = bchw2colormap(logits_cw)
            predroom = bchw2colormap(logits_r)
            predroom_post = clean_room(predroom, predboundary)

            rgb_room_raw = room2rgb(predroom)
            rgb_room_post = room2rgb(predroom_post)
            rgb_boundary = boundary2rgb(predboundary)

            rgb_full = rgb_boundary.copy()
            rgb_full[rgb_boundary.sum(axis=2) == 0] = rgb_room_raw[
                rgb_boundary.sum(axis=2) == 0
            ]
            if not args.postprocess:
                pred_image = Image.fromarray(rgb_full.astype(np.uint8))
                pred_image.save(dst_dir / f"{file.stem}.png")
                continue

            rgb_full_post = rgb_boundary.copy()
            rgb_full_post[rgb_boundary.sum(axis=2) == 0] = rgb_room_post[
                rgb_boundary.sum(axis=2) == 0
            ]
            rgb_full_post[rgb_full_post.sum(axis=2) == 0] = rgb_room_raw[
                rgb_full_post.sum(axis=2) == 0
            ]
            rgb_full_post = fill_holes(rgb_full_post)

            pred_image = Image.fromarray(rgb_full_post.astype(np.uint8))
            pred_image.save(dst_dir / f"{file.stem}.png")
        logging.info("Finished inference")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights_path", type=str, required=True)
    p.add_argument("--room_channels", type=int, required=True)
    p.add_argument("--boundary_channels", type=int, required=True)
    p.add_argument("--src_dir", type=str, required=True)
    p.add_argument("--dst_dir", type=str, required=True)
    p.add_argument("--postprocess", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    inference(args)
