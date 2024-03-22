import os
import random
import time
from datetime import datetime

import torch
from compel import Compel
from diffusers.utils import export_to_video

MIN_VAL = -0x8000_0000_0000_0000
MAX_VAL = 0xffff_ffff_ffff_ffff


def export_frames_to_video(frames, output_path: str, to_h264: bool = True):
    print(f"Exporting to video at {datetime.now()}")

    if not to_h264:
        return export_to_video(video_frames=frames, output_video_path=output_path)

    tmp_path = "/tmp" + str(int(time.time())) + ".mp4"
    export_to_video(video_frames=frames, output_video_path=tmp_path)

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
        seed=None,
        use_compel=False,
        to_h264=True
):
    seed = seed if seed else random.randint(MIN_VAL, MAX_VAL)

    # Generate video frames
    if use_compel:
        print("using compel")
        compel = Compel(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            truncate_long_prompts=False,
            device="cpu"
        )
        conditioning = compel.build_conditioning_tensor(prompt)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length(
            [conditioning, neg_conditioning])

        video_frames = pipe(
            prompt_embeds=conditioning,
            negative_prompt_embeds=neg_conditioning,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames
        torch.cuda.empty_cache()
        return export_frames_to_video(video_frames[0], output_path, to_h264)

    with torch.inference_mode():
        video_frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames

        torch.cuda.empty_cache()
        return export_frames_to_video(video_frames[0], output_path, to_h264)
