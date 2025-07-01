import os

import cv2


class VideoSegmentExtractor:
    """A class to extract video segments."""

    def frame_to_timestamp(self, frame_idx: int, fps: int) -> str:
        """Converts a frame index to a timestamp string.

        Args:
            frame_idx: The frame index.
            fps: The frames per second of the video.

        Returns:
            The timestamp string in the format HH:MM:SS.
        """
        seconds = frame_idx / fps
        mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        return f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"

    def extract_video_segment(
        self,
        video_path: str,
        frame_idx: int,
        fps: int,
        duration: int = 5,
        output_dir: str = "./segments",
    ) -> str:
        """Extracts a video segment around a given frame index.

        Args:
            video_path: The path to the video.
            frame_idx: The frame index around which to extract the segment.
            fps: The frames per second of the video.
            duration: The duration of the segment to extract in seconds.
            output_dir: The directory to save the extracted segment.

        Returns:
            The path to the extracted video segment.
        """
        start_frame = max(0, frame_idx - int(fps * duration // 2))
        end_frame = frame_idx + int(fps * duration // 2)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = min(end_frame, total_frames - 1)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"segment_{os.path.basename(video_path).split('.')[0]}_{frame_idx}.mp4",
        )

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
        return output_path
