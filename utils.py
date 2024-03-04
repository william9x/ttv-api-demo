import tempfile
from typing import List, Union

import PIL.Image
import PIL.ImageOps
import numpy as np
import torch
from diffusers.utils.import_utils import is_opencv_available, BACKENDS_MAPPING


def export_to_video(
        video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 8
) -> str:
    if is_opencv_available():
        import cv2
    else:
        raise ImportError(BACKENDS_MAPPING["opencv"][1].format("export_to_video"))
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

    print(f"Video generated: {output_video_path}")
    return output_video_path


def generate_video(
        pipe=None,
        prompt="a group of penguins swimming with the waves",
        negative_prompt="bad quality",
        num_inference_steps=4,
        height=512,
        width=512,
        num_frames=16,
        output_path=None,
        guidance_scale=1.5,
):
    # Generate video frames
    video_frames = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        num_frames=num_frames,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        generator=torch.Generator(),
    ).frames

    torch.cuda.empty_cache()

    return export_to_video(video_frames[0], output_path)
