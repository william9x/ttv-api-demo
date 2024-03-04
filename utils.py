import os
import time
from datetime import datetime

import torch
from diffusers.utils import export_to_video


def export_frames_to_video(frames, output_path):
    print(f"Exporting to video at {datetime.now()}")
    tmp_path = "/tmp" + str(int(time.time())) + ".mp4"
    export_to_video(frames, output_video_path=tmp_path)

    print(f"Converting to H264 at {datetime.now()}")
    os.system(f"ffmpeg -y -hide_banner -loglevel error -i {tmp_path} -vcodec libx264 -preset superfast {output_path}")

    print(f"Video generated: {output_path} at {datetime.now()}")
    return output_path


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

    return export_frames_to_video(video_frames[0], output_path)
