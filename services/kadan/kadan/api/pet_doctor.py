"""Conservative scale correction from corresponding interior image features."""

import math

import cv2
import numpy as np
from PIL import Image

from .pet_artwork import CELL, SOURCE_CELL, _bounds, _encode, _png
from .pydantic_models.pets import PetScaleCorrection


class PetDoctor:
    def __init__(self, reference: bytes):
        self.reference = _png(reference, SOURCE_CELL)
        self.detector = cv2.SIFT_create(nfeatures=600, contrastThreshold=0.025)
        self.points, self.descriptors = self.features(self.reference, reference=True)

    def features(self, image: Image.Image, *, reference: bool = False):
        rgba = np.array(image)
        mask = (rgba[:, :, 3] > 128).astype(np.uint8) * 255
        # Background boundaries change with articulation. Match interior detail.
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        if reference:
            left, top, right, bottom = _bounds(image)
            width, height = right - left, bottom - top
            core = np.zeros_like(mask)
            core[
                top : top + round(height * 0.8),
                left + round(width * 0.2) : right - round(width * 0.2),
            ] = 255
            mask = cv2.bitwise_and(mask, core)
        rgb = rgba[:, :, :3].copy()
        rgb[rgba[:, :, 3] <= 16] = 0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return self.detector.detectAndCompute(gray, mask)

    def correct(self, png: bytes) -> tuple[bytes, PetScaleCorrection]:
        frame = _png(png, SOURCE_CELL)
        bounds = _bounds(frame)

        def unchanged(reason: str, outcome: str = "skipped"):
            return png, PetScaleCorrection(outcome=outcome, reason=reason)

        if frame.tobytes() == self.reference.tobytes():
            return unchanged("Matches the character reference.", "consistent")
        points, descriptors = self.features(frame)
        if self.descriptors is None or descriptors is None:
            return unchanged("Not enough distinctive body details to measure scale.")
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        forward = matcher.knnMatch(descriptors, self.descriptors, k=2)
        reverse = matcher.knnMatch(self.descriptors, descriptors, k=2)
        mutual = {
            (pair[0].trainIdx, pair[0].queryIdx)
            for pair in reverse
            if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
        }
        matches = []
        seen_source, seen_target = set(), set()
        for pair in forward:
            if len(pair) != 2:
                continue
            best, other = pair
            if (
                best.distance >= 0.7 * other.distance
                or (best.queryIdx, best.trainIdx) not in mutual
            ):
                continue
            source = tuple(round(v) for v in points[best.queryIdx].pt)
            target = tuple(round(v) for v in self.points[best.trainIdx].pt)
            # SIFT can assign several orientations to the same physical feature.
            if source in seen_source or target in seen_target:
                continue
            seen_source.add(source)
            seen_target.add(target)
            matches.append(best)
        if len(matches) < 6:
            return unchanged("Too few matching body details; pose left unchanged.")
        source = np.float32([points[m.queryIdx].pt for m in matches])
        target = np.float32([self.points[m.trainIdx].pt for m in matches])
        transform, inliers = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=1000,
            confidence=0.99,
            refineIters=10,
        )
        if transform is None or inliers is None:
            return unchanged("Matching details do not agree on a body scale.")
        accepted = inliers.ravel().astype(bool)
        body = source[accepted]
        left, top, right, bottom = bounds
        if (
            len(body) < 6
            or accepted.mean() < 0.6
            or np.ptp(body[:, 0]) < max(16, (right - left) * 0.15)
            or np.ptp(body[:, 1]) < max(16, (bottom - top) * 0.1)
        ):
            return unchanged("Matching details cover too little of the body reliably.")
        scale = math.hypot(transform[0, 0], transform[1, 0])
        rotation = abs(math.degrees(math.atan2(transform[1, 0], transform[0, 0])))
        if not 0.67 <= scale <= 1.5 or rotation > 12:
            return unchanged("Pose differs too much for a reliable scale correction.")
        detail_scales = np.array(
            [
                self.points[m.trainIdx].size / points[m.queryIdx].size
                for m, keep in zip(matches, accepted, strict=True)
                if keep
            ]
        )
        if np.median(np.abs(detail_scales / scale - 1)) > 0.12:
            return unchanged("Body measurements disagree; pose left unchanged.")
        if abs(scale - 1) <= 0.03:
            return unchanged("Body scale is already within tolerance.", "consistent")
        # Keep body position and the existing support/airborne height. Neither
        # silhouette height nor extended arms determine the scale or center.
        anchor_x, anchor_y = float(body[:, 0].mean()), bottom
        tx, ty = anchor_x * (1 - scale), anchor_y * (1 - scale)
        if (
            left * scale + tx < 4
            or right * scale + tx > SOURCE_CELL[0] - 4
            or top * scale + ty < 4
            or bottom * scale + ty > SOURCE_CELL[1] - 4
        ):
            return unchanged(
                "Correction would clip the character; pose left unchanged."
            )
        corrected = (
            frame.convert("RGBa")
            .transform(
                SOURCE_CELL,
                Image.Transform.AFFINE,
                (1 / scale, 0, -tx / scale, 0, 1 / scale, -ty / scale),
                resample=Image.Resampling.BICUBIC,
            )
            .convert("RGBA")
        )
        try:
            _bounds(corrected)
            _bounds(corrected.resize(CELL, Image.Resampling.LANCZOS))
        except ValueError:
            return unchanged("Correction would leave insufficient clear margins.")
        return _encode(corrected), PetScaleCorrection(
            outcome="corrected",
            scale=scale,
            reason="Uniform scale from matching interior details; foot height retained.",
        )
