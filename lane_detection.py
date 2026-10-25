from pathlib import Path
import argparse

import cv2
import numpy as np


BASE = Path(__file__).resolve().parent
DOCS = BASE / "docs" / "lane_detection"
VIDEO_OUTPUT = BASE / "output_videos"


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Ne mogu da sacuvam: {path}")


def load_calibration():
    path = BASE / "camera_calibration.npz"

    if not path.exists():
        raise RuntimeError("Prvo pokreni calibration.py.")

    with np.load(path) as data:
        return (
            data["camera_matrix"].copy(),
            data["distortion"].copy(),
            tuple(int(v) for v in data["image_size"]),
        )


class LaneDetector:
    def __init__(self, calibration):
        self.matrix, self.distortion, self.calibration_size = calibration
        self.previous_fit = None
        self.missed_frames = 0
        self.maps = {}

    def undistort(self, image):
        h, w = image.shape[:2]
        key = (w, h)

        if key not in self.maps:
            cw, ch = self.calibration_size
            matrix = self.matrix.copy()
            matrix[0, :] *= w / cw
            matrix[1, :] *= h / ch

            self.maps[key] = cv2.initUndistortRectifyMap(
                matrix,
                self.distortion,
                None,
                matrix,
                (w, h),
                cv2.CV_32FC1,
            )

        map_x, map_y = self.maps[key]

        return cv2.remap(
            image, map_x, map_y, cv2.INTER_LINEAR
        )

    def binary_image(self, image):
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

        # Lokalni kontrast izdvaja uske svetle oznake od svetlog asfalta.
        h, w = image.shape[:2]
        kernel_width = max(15, int(round(w * 0.025)) | 1)
        lightness = hls[:, :, 1]
        local_contrast = cv2.morphologyEx(
            lightness,
            cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3)),
        )
        white = (
            (lightness >= 160)
            & (hsv[:, :, 2] >= 180)
            & (hsv[:, :, 1] <= 100)
            & (local_contrast >= 15)
        )

        yellow = cv2.inRange(
            hsv,
            np.array([15, 65, 80], dtype=np.uint8),
            np.array([40, 255, 255], dtype=np.uint8),
        ) > 0

        binary = (white | yellow).astype(np.uint8) * 255
        h, w = binary.shape

        mask = np.zeros_like(binary)
        polygon = np.array(
            [[
                (int(0.02 * w), h - 1),
                (int(0.43 * w), int(0.64 * h)),
                (int(0.60 * w), int(0.64 * h)),
                (w - 1, h - 1),
            ]],
            dtype=np.int32,
        )

        cv2.fillPoly(mask, polygon, 255)
        binary = cv2.bitwise_and(binary, mask)

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )

        return cv2.bitwise_and(binary, mask)

    def perspective(self, binary):
        h, w = binary.shape

        src = np.float32([
            [0.45 * w, 0.64 * h],
            [0.57 * w, 0.64 * h],
            [0.86 * w, 0.94 * h],
            [0.12 * w, 0.94 * h],
        ])

        dst = np.float32([
            [0.25 * w, 0],
            [0.75 * w, 0],
            [0.75 * w, h - 1],
            [0.25 * w, h - 1],
        ])

        matrix = cv2.getPerspectiveTransform(src, dst)
        inverse = cv2.getPerspectiveTransform(dst, src)

        warped = cv2.warpPerspective(
            binary,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
        )

        return warped, inverse, src

    def find_fit(self, warped):
        h, w = warped.shape
        histogram = np.sum(warped[h // 2:] > 0, axis=0)
        midpoint = w // 2

        visualization = cv2.cvtColor(
            warped, cv2.COLOR_GRAY2BGR
        )

        if (
            histogram[:midpoint].max() == 0
            or histogram[midpoint:].max() == 0
        ):
            return None, visualization

        centers = [
            int(np.argmax(histogram[:midpoint])),
            int(np.argmax(histogram[midpoint:]) + midpoint),
        ]

        y, x = warped.nonzero()
        selected = [[], []]
        margin = max(25, int(0.07 * w))
        levels = np.linspace(h, 0, 10).astype(int)

        for bottom, top in zip(levels[:-1], levels[1:]):
            for side in range(2):
                left = centers[side] - margin
                right = centers[side] + margin

                indices = np.flatnonzero(
                    (y >= top)
                    & (y < bottom)
                    & (x >= left)
                    & (x < right)
                )

                selected[side].append(indices)

                cv2.rectangle(
                    visualization,
                    (max(0, left), top),
                    (min(w - 1, right), bottom - 1),
                    (0, 140, 0),
                    2,
                )

                if len(indices) >= 30:
                    centers[side] = int(
                        np.mean(x[indices])
                    )

        fits = []

        for side, color in enumerate(
            [(0, 0, 255), (255, 0, 0)]
        ):
            indices = np.concatenate(selected[side])

            if len(indices) < 100:
                return None, visualization

            if np.ptp(y[indices]) < 0.30 * h:
                return None, visualization

            fit = np.polyfit(
                y[indices], x[indices], 2
            )

            fits.append(fit)
            visualization[y[indices], x[indices]] = color

        fits = np.array(fits)

        if not self.valid_fit(fits, h, w):
            return None, visualization

        return fits, visualization

    @staticmethod
    def valid_fit(fits, h, w):
        if not np.isfinite(fits).all():
            return False

        y = np.linspace(0, h - 1, 30)
        left = np.polyval(fits[0], y)
        right = np.polyval(fits[1], y)
        widths = right - left

        return bool(
            np.all(widths > 0.20 * w)
            and np.all(widths < 0.85 * w)
            and np.ptp(widths) < 0.35 * w
            and np.all(left > -0.20 * w)
            and np.all(right < 1.20 * w)
        )

    @staticmethod
    def measurements(fits, h, w):
        # Priblizna skala: sirina trake 3.7 m,
        # vidljiva duzina puta 30 m.
        xm = 3.7 / (0.50 * w)
        ym = 30.0 / max(h - 1, 1)

        y = np.linspace(0, h - 1, h)
        radii = []

        for fit in fits:
            x = np.polyval(fit, y)
            a, b, _ = np.polyfit(
                y * ym, x * xm, 2
            )

            denominator = abs(2 * a)

            if denominator < 1e-9:
                radius = float("inf")
            else:
                radius = (
                    (1 + (2 * a * y[-1] * ym + b) ** 2)
                    ** 1.5
                    / denominator
                )

            radii.append(radius)

        lane_center = (
            np.polyval(fits[0], h - 1)
            + np.polyval(fits[1], h - 1)
        ) / 2

        # Kamera se smatra postavljenom na sredini vozila.
        # Pozitivno odstupanje znaci da je vozilo desno.
        offset = (w / 2 - lane_center) * xm

        return float(np.mean(radii)), float(offset)

    @staticmethod
    def draw_text_panel(image, messages):
        h, w = image.shape[:2]
        scale = max(0.5, w / 1600)
        thickness = 1
        padding = 15

        sizes = [
            cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                thickness,
            )
            for text in messages
        ]

        text_height = max(size[0][1] for size in sizes)
        baseline = max(size[1] for size in sizes)
        line_height = text_height + baseline + 14

        panel_width = min(
            w,
            max(size[0][0] for size in sizes) + 2 * padding,
        )
        panel_height = min(
            h, padding + line_height * len(messages)
        )

        cv2.rectangle(
            image,
            (0, 0),
            (panel_width - 1, panel_height - 1),
            (25, 25, 25),
            -1,
        )

        for index, text in enumerate(messages):
            cv2.putText(
                image,
                text,
                (
                    padding,
                    padding + text_height + index * line_height,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    def process(self, frame, tracking=False):
        undistorted = self.undistort(frame)
        binary = self.binary_image(undistorted)
        warped, inverse, src = self.perspective(binary)
        fits, fit_view = self.find_fit(warped)

        h, w = binary.shape
        status = "Detektovana"

        if fits is not None:
            if tracking and self.previous_fit is not None:
                sample_y = np.linspace(0, h - 1, 15)

                movement = max(
                    np.max(
                        np.abs(
                            np.polyval(fits[i], sample_y)
                            - np.polyval(
                                self.previous_fit[i], sample_y
                            )
                        )
                    )
                    for i in range(2)
                )

                if movement > 0.15 * w:
                    fits = None
                else:
                    fits = (
                        0.25 * fits
                        + 0.75 * self.previous_fit
                    )

        if tracking:
            if fits is not None:
                self.previous_fit = fits.copy()
                self.missed_frames = 0
            else:
                self.missed_frames += 1

                if (
                    self.previous_fit is not None
                    and self.missed_frames <= 5
                ):
                    fits = self.previous_fit.copy()
                    status = "Prethodna procena"
                else:
                    self.previous_fit = None

        result = undistorted.copy()

        if fits is not None:
            y = np.arange(h, dtype=np.float64)
            left = np.polyval(fits[0], y)
            right = np.polyval(fits[1], y)

            left_points = np.column_stack(
                (left, y)
            ).astype(np.int32)

            right_points = np.column_stack(
                (right, y)
            ).astype(np.int32)

            polygon = np.vstack(
                (left_points, right_points[::-1])
            )

            overlay = np.zeros_like(undistorted)
            cv2.fillPoly(
                overlay, [polygon], (0, 180, 0)
            )

            cv2.polylines(
                overlay,
                [left_points, right_points],
                False,
                (0, 255, 255),
                8,
            )

            road_overlay = cv2.warpPerspective(
                overlay, inverse, (w, h)
            )

            result = cv2.addWeighted(
                undistorted,
                1.0,
                road_overlay,
                0.35,
                0,
            )

            cv2.polylines(
                fit_view,
                [left_points, right_points],
                False,
                (0, 255, 255),
                3,
            )

            radius, offset = self.measurements(
                fits, h, w
            )

            if not np.isfinite(radius) or radius > 10000:
                radius_text = "priblizno prava"
            else:
                radius_text = f"{radius:.0f} m"

            if abs(offset) < 0.01:
                offset_text = "u sredini"
            else:
                direction = "desno" if offset > 0 else "levo"
                offset_text = (
                    f"{abs(offset):.2f} m {direction}"
                )

            messages = [
                f"Procena poluprecnika: {radius_text}",
                f"Procena odstupanja: {offset_text}",
                status,
            ]
        else:
            messages = [
                "Traka nije pouzdano detektovana"
            ]

        self.draw_text_panel(result, messages)

        source_view = undistorted.copy()

        cv2.polylines(
            source_view,
            [src.astype(np.int32)],
            True,
            (0, 0, 255),
            3,
        )

        stages = {
            "undistorted.jpg": undistorted,
            "binary.png": binary,
            "perspective_source.jpg": source_view,
            "warped.png": warped,
            "lane_fit.jpg": fit_view,
            "result.jpg": result,
        }

        return result, stages, fits is not None


def process_images(calibration):
    input_dir = BASE / "test_images"

    if not input_dir.exists():
        raise RuntimeError("Nedostaje test_images.")

    paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not paths:
        raise RuntimeError("Nema ulaznih slika.")

    detector = LaneDetector(calibration)
    detected = 0
    processed = 0

    for path in paths:
        image = cv2.imread(str(path))

        if image is None:
            print(f"Ne mogu da ucitam: {path.name}")
            continue

        _, stages, valid = detector.process(image)

        destination = DOCS / path.stem
        destination.mkdir(
            parents=True, exist_ok=True
        )

        for name, stage in stages.items():
            save_image(destination / name, stage)

        processed += 1
        detected += int(valid)

        status = (
            "detektovana" if valid
            else "nije pouzdana"
        )
        print(f"{path.name}: {status}")

    print(f"\nObradjeno slika: {processed}/{len(paths)}")
    print(
        f"Detekcija prihvacena na "
        f"{detected}/{processed} slika."
    )
    print(f"Rezultati: {DOCS}")


def process_video(path, calibration):
    if not path.exists():
        raise RuntimeError(
            f"Ne postoji video: {path}"
        )

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Ne mogu da otvorim video: {path}"
        )

    writer = None
    detector = LaneDetector(calibration)
    count = 0
    accepted = 0

    try:
        success, frame = capture.read()

        if not success:
            raise RuntimeError(
                "Video nema citljiv prvi kadar."
            )

        h, w = frame.shape[:2]
        fps = capture.get(cv2.CAP_PROP_FPS)

        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        VIDEO_OUTPUT.mkdir(exist_ok=True)
        output = (
            VIDEO_OUTPUT
            / f"{path.stem}_result.mp4"
        )

        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Ne mogu da napravim izlazni MP4."
            )

        while success:
            result, _, valid = detector.process(
                frame, tracking=True
            )

            writer.write(result)
            count += 1
            accepted += int(valid)

            if count % 100 == 0:
                print(
                    f"Obradjeno {count} kadrova..."
                )

            success, frame = capture.read()

        print(f"\nSacuvan video: {output}")
        print(f"Ukupno kadrova: {count}")
        print(
            "Kadrovi sa prikazanom procenom: "
            f"{accepted}"
        )

    finally:
        capture.release()

        if writer is not None:
            writer.release()


def main():
    parser = argparse.ArgumentParser(
        description="Detekcija saobracajne trake."
    )

    parser.add_argument(
        "--video",
        help=(
            "Putanja do videa. Bez ove opcije "
            "obradjuju se slike iz test_images."
        ),
    )

    args = parser.parse_args()
    calibration = load_calibration()

    if args.video:
        path = Path(args.video)

        if not path.is_absolute():
            path = BASE / path

        process_video(path, calibration)
    else:
        process_images(calibration)


if __name__ == "__main__":
    main()