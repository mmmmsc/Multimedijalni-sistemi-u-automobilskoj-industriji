from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = BASE_DIR / "camera_cal"
DOCS_DIR = BASE_DIR / "docs"

PATTERN_SIZE = (9, 6)
COMPARISON_IMAGE = "calibration1.jpg"


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Nije uspelo cuvanje slike: {path}")


def calibrate_camera():
    DOCS_DIR.mkdir(exist_ok=True)

    # Ovu fotografiju koristimo za prikaz pre i posle ispravke.
    example_path = CALIBRATION_DIR / COMPARISON_IMAGE
    original = cv2.imread(str(example_path))

    if original is None:
        raise RuntimeError(f"Ne mogu da ucitam: {example_path}")

    image_size = (original.shape[1], original.shape[0])

    columns, rows = PATTERN_SIZE

    # Koordinate unutrasnjih uglova na ravnoj sahovskoj tabli.
    object_points = np.zeros(
        (columns * rows, 3), dtype=np.float32
    )
    object_points[:, :2] = np.mgrid[
        0:columns, 0:rows
    ].T.reshape(-1, 2)

    all_object_points = []
    all_image_points = []
    corners_example = None

    image_paths = sorted(CALIBRATION_DIR.glob("*.jpg"))

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Ne mogu da ucitam: {image_path.name}")
            continue

        current_size = (image.shape[1], image.shape[0])

        if current_size != image_size:
            print(f"Druga rezolucija, preskacem: {image_path.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            PATTERN_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            print(f"Uglovi nisu pronadjeni: {image_path.name}")
            continue

        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria,
        )

        all_object_points.append(object_points.copy())
        all_image_points.append(refined_corners)

        print(f"Uglovi pronadjeni: {image_path.name}")

        # Poseban primer uspesno pronadjenih uglova.
        if corners_example is None:
            corners_example = image.copy()
            cv2.drawChessboardCorners(
                corners_example,
                PATTERN_SIZE,
                refined_corners,
                True,
            )

    if len(all_object_points) < 3:
        raise RuntimeError(
            "Premalo uspesnih slika za kalibraciju. "
            "Proveri folder camera_cal i PATTERN_SIZE."
        )

    # Izracunavanje parametara kamere.
    rms_error, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        all_object_points,
        all_image_points,
        image_size,
        None,
        None,
    )

    np.savez(
        BASE_DIR / "camera_calibration.npz",
        camera_matrix=camera_matrix,
        distortion=distortion,
        image_size=np.array(image_size),
        rms_error=rms_error,
    )

    # Primena parametara na calibration1.jpg.
    corrected = cv2.undistort(
        original,
        camera_matrix,
        distortion,
        None,
        camera_matrix,
    )

    save_image(
        DOCS_DIR / "calibration_corners.jpg",
        corners_example,
    )
    save_image(
        DOCS_DIR / "calibration_original.jpg",
        original,
    )
    save_image(
        DOCS_DIR / "calibration_undistorted.jpg",
        corrected,
    )

    # Uporedni prikaz sa naslovima iznad fotografija.
    height, width = original.shape[:2]
    header_height = 70

    comparison = np.full(
        (height + header_height, width * 2, 3),
        245,
        dtype=np.uint8,
    )

    comparison[header_height:, :width] = original
    comparison[header_height:, width:] = corrected

    for text, x in (
        ("Original", 25),
        ("Undistorted", width + 25),
    ):
        cv2.putText(
            comparison,
            text,
            (x, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

    cv2.line(
        comparison,
        (width, 0),
        (width, comparison.shape[0] - 1),
        (120, 120, 120),
        2,
    )

    save_image(
        DOCS_DIR / "calibration_comparison.jpg",
        comparison,
    )

    print("\nKalibracija je zavrsena.")
    print(f"Iskorisceno slika: {len(all_object_points)}/{len(image_paths)}")
    print(f"RMS reprojekciona greska: {rms_error:.4f} piksela")
    print(f"Slika za poredjenje: {COMPARISON_IMAGE}")

    print("\nMatrica kamere:")
    print(camera_matrix)

    print("\nKoeficijenti izoblicenja:")
    print(distortion)

    print("\nParametri su sacuvani u camera_calibration.npz.")
    print("Slike su sacuvane u folderu docs.")


if __name__ == "__main__":
    calibrate_camera()