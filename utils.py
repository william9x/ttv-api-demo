import os
import time

import torch
from diffusers.utils import export_to_video


def export_frames_to_video(frames, output_path):
    tmp_path = "/tmp" + str(int(time.time())) + ".mp4"
    export_to_video(frames, output_video_path=tmp_path)
    os.system(f"ffmpeg -y -i {tmp_path} -vcodec libx264 {output_path}")
    print(f"Video generated: {output_path}")
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
