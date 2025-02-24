"""
Visualization utility for AT-GAN
Copyright (c) Xingyuangfy 2025. All rights reserved.

This module provides visualization tools for displaying and saving training images and logs.
"""

import numpy as np
import os
import cv2
import time
import unidecode
from . import util
from . import html
from pdb import set_trace as st


class Visualizer:
    """
    Visualization tool class for displaying and saving training images and logs.
    """

    def __init__(self, opt):
        """
        Initialize the Visualizer.

        Args:
            opt: Configuration object containing visualization parameters.
        """
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.numClasses = opt.numClasses
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
        self.isTrain = opt.isTrain
        self.save_freq = opt.save_display_freq if self.isTrain else None

        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)
            self.display_single_pane_ncols = opt.display_single_pane_ncols

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print(f"Creating web directory {self.web_dir}...")
            util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        if self.isTrain:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write(f"================ Training Loss ({now}) ================\n")

    def display_current_results(self, visuals, it, classes, ncols=None):
        """
        Display current results in Visdom or HTML page.

        Args:
            visuals: Dictionary containing images.
            it: Current iteration number.
            classes: Class information.
            ncols: Number of images per row.
        """
        if self.display_id > 0:
            self._display_results_with_visdom(visuals, it, classes, ncols)
        if self.use_html:
            self._display_results_with_html(visuals, it, classes)

    def _display_results_with_visdom(self, visuals, it, classes, ncols):
        """
        Display current results using Visdom.
        """
        if ncols is None:
            ncols = self.display_single_pane_ncols
        if ncols > 0:
            self._display_single_pane(visuals, it, classes, ncols)
        else:
            self._display_individual_images(visuals, it)

    def _display_single_pane(self, visuals, it, classes, ncols):
        """
        Display images in a single panel.
        """
        h, w = next(iter(visuals.values())).shape[:2]
        table_css = f"""
        <style>
        table {{border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}}
        table td {{width: {w}px; height: {h}px; padding: 4px; outline: 4px solid black}}
        </style>
        """
        title = self.name
        label_html = ""
        label_html_row = ""
        nrows = int(np.ceil(len(visuals) / ncols))
        images = []
        idx = 0

        for label, image_numpy in visuals.items():
            label_html_row += f"<td>{label}</td>"
            images.append(self._prepare_image(image_numpy))
            idx += 1
            if idx % ncols == 0:
                label_html += f"<tr>{label_html_row}</tr>"
                label_html_row = ""

        self._pad_images(images, ncols)
        label_html += f"<tr>{label_html_row}</tr>" if label_html_row else ""

        self.vis.images(images, nrow=ncols, win=self.display_id + 1, padding=2, opts=dict(title=title + " images"))
        label_html = f"<table>{label_html}</table>"
        self.vis.text(table_css + label_html, win=self.display_id + 2, opts=dict(title=title + " labels"))

    def _display_individual_images(self, visuals, it):
        """
        Display each image separately.
        """
        for idx, (label, image_numpy) in enumerate(visuals.items()):
            self.vis.image(self._prepare_image(image_numpy), opts=dict(title=label), win=self.display_id + idx + 1)

    def _prepare_image(self, image_numpy):
        """
        Prepare image for display.
        """
        if image_numpy.ndim < 3:
            image_numpy = np.expand_dims(image_numpy, 2)
            image_numpy = np.tile(image_numpy, (1, 1, 3))
        return image_numpy.transpose([2, 0, 1])

    def _pad_images(self, images, ncols):
        """
        Pad image list to match number of columns per row.
        """
        white_image = np.ones_like(images[0]) * 255
        while len(images) % ncols != 0:
            images.append(white_image)

    def plot_current_errors(self, epoch, counter_ratio, errors):
        """
        Plot current errors.

        Args:
            epoch: Current epoch.
            counter_ratio: Progress ratio.
            errors: Dictionary of errors.
        """
        if not hasattr(self, "plot_data"):
            self.plot_data = {"X": [], "Y": [], "legend": list(errors.keys())}
        self.plot_data["X"].append(epoch + counter_ratio)
        self.plot_data["Y"].append([errors[k] for k in self.plot_data["legend"]])
        self.vis.line(
            X=np.stack([np.array(self.plot_data["X"])] * len(self.plot_data["legend"]), 1),
            Y=np.array(self.plot_data["Y"]),
            opts={
                "title": f"{self.name} loss over time",
                "legend": self.plot_data["legend"],
                "xlabel": "epoch",
                "ylabel": "loss",
            },
            win=self.display_id,
        )

    def print_current_errors(self, epoch, i, errors, t):
        """
        Print current errors.

        Args:
            epoch: Current epoch.
            i: Current iteration.
            errors: Dictionary of errors.
            t: Time elapsed.
        """
        message = f"(epoch: {epoch}, iters: {i}, time: {t:.3f}) "
        for k, v in errors.items():
            message += f"{k}: {v:.3f} "
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write(f"{message}\n")

    def save_matrix_image(self, visuals, epoch):
        """
        Save matrix image.

        Args:
            visuals: Dictionary containing images.
            epoch: Current epoch.
        """
        matrix_img = self._create_matrix_image(visuals)
        epoch_txt = f"epoch_{epoch}" if epoch != "latest" else "latest"
        image_path = os.path.join(self.img_dir, f"sample_batch_{epoch_txt}.png")
        util.save_image(matrix_img, image_path)

    def _create_matrix_image(self, visuals):
        """
        Create matrix image.
        """
        visuals = visuals[0]
        orig_img = visuals["orig_img"]
        matrix_img = orig_img
        for cls in range(self.numClasses):
            next_im = visuals[f"tex_trans_to_class_{cls}"]
            matrix_img = np.concatenate((matrix_img, next_im), 1)
        return matrix_img

    def save_row_image(self, visuals, image_path, traverse=False):
        """
        Save row image.

        Args:
            visuals: Dictionary containing images.
            image_path: Path to save the image.
            traverse: Whether to traverse.
        """
        visual = visuals[0]
        orig_img = visual["orig_img"]
        traversal_img = np.concatenate((orig_img, np.full((orig_img.shape[0], 10, 3), 255, dtype=np.uint8)), 1)
        out_classes = len(visual) - 1 if traverse else self.numClasses
        for cls in range(out_classes):
            next_im = visual[f"tex_trans_to_class_{cls}"]
            traversal_img = np.concatenate((traversal_img, next_im), 1)
        util.save_image(traversal_img, image_path)

    def make_video(self, visuals, video_path):
        """
        Create video.

        Args:
            visuals: Dictionary containing images.
            video_path: Path to save the video.
        """
        fps = 20
        visual = visuals[0]
        orig_img = visual["orig_img"]
        h, w = orig_img.shape[:2]
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        out_classes = len(visual) - 1
        for cls in range(out_classes):
            next_im = visual[f"tex_trans_to_class_{cls}"]
            writer.write(next_im[:, :, ::-1])
        writer.release()

    def save_images_deploy(self, visuals, image_path):
        """
        Save images (deployment mode).

        Args:
            visuals: Dictionary containing images.
            image_path: Path to save the images.
        """
        for i in range(len(visuals)):
            visual = visuals[i]
            for label, image_numpy in visual.items():
                save_path = f"{image_path}_{label}.png"
                util.save_image(image_numpy, save_path)

    def save_images(self, webpage, visuals, image_path, gt_visuals=None, gt_path=None):
        """
        Save images to HTML page.

        Args:
            webpage: HTML page object.
            visuals: Dictionary containing images.
            image_path: Image path.
            gt_visuals: Ground truth images.
            gt_path: Ground truth path.
        """
        image_dir = webpage.get_image_dir()
        if gt_visuals is None or gt_path is None:
            self._save_images_without_gt(webpage, visuals, image_path, image_dir)
        else:
            self._save_images_with_gt(webpage, visuals, image_path, gt_visuals, gt_path, image_dir)

    def _save_images_without_gt(self, webpage, visuals, image_path, image_dir):
        """
        Save images without ground truth.
        """
        for i in range(len(visuals)):
            visual = visuals[i]
            short_path = os.path.basename(image_path[i])
            name = unidecode.unidecode(os.path.splitext(short_path)[0])
            webpage.add_header(name)
            ims, txts, links = [], [], []

            for label, image_numpy in visual.items():
                image_name = f"{name}_{label}.png"
                save_path = os.path.join(image_dir, image_name)
                util.save_image(image_numpy, save_path)

                ims.append(image_name)
                txts.append(label)
                links.append(image_name)

            webpage.add_images(ims, txts, links, width=self.win_size, cols=self.numClasses + 1)

    def _save_images_with_gt(self, webpage, visuals, image_path, gt_visuals, gt_path, image_dir):
        """
        Save images with ground truth.
        """
        batchSize = len(image_path)
        gt_short_path = os.path.basename(gt_path[0])
        gt_name = os.path.splitext(gt_short_path)[0]
        gt_ims, gt_txts, gt_links = [], [], []

        for label, image_numpy in gt_visuals.items():
            image_name = f"{gt_name}_{label}.png"
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            gt_ims.append(image_name)
            gt_txts.append(label)
            gt_links.append(image_name)

        for i in range(batchSize):
            short_path = os.path.basename(image_path[i])
            name = os.path.splitext(short_path)[0]

            webpage.add_header(gt_name)
            webpage.add_images(gt_ims, gt_txts, gt_links, width=self.win_size, cols=batchSize)

            ims, txts, links = [], [], []
            for label, image_numpy in visuals[i].items():
                image_name = f"{name}_{label}.png"
                save_path = os.path.join(image_dir, image_name)
                util.save_image(image_numpy, save_path)

                ims.append(image_name)
                txts.append(label)
                links.append(image_name)

            webpage.add_header(name)
            webpage.add_images(ims, txts, links, width=self.win_size, cols=self.numClasses + 1)