from typing import Final
from pathlib import Path
import cv2
import tone_mapping

IMAGE_PATH: Final[Path] = Path("test.tif")


def main() -> None:
    img = cv2.imread(
        str(IMAGE_PATH),
        cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR,
    )
    img = tone_mapping.tone_map_pq_histogram(img)

    cv2.imshow("Example", img)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
