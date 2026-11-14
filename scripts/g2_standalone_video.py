"""Export the author evidence video from saved policy-query images."""

from pathlib import Path
import sys
import numpy as np
from pace_client_smoke import Video


def main(directory):
    paths = sorted(directory.glob("request-*.npz"))
    if not paths:
        return
    video = Video(directory / "query-cameras.mp4")
    try:
        for path in paths:
            with np.load(path, allow_pickle=False) as frame:
                image = np.concatenate(
                    [
                        frame[key].transpose(1, 2, 0)
                        for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
                    ],
                    axis=1,
                )
                video.frame({"observation": {"head_camera": {"rgb": image}}})
    finally:
        video.close()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
