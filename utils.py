import random
import time
from datetime import datetime

import ffmpeg
import torch
from compel import Compel
from diffusers.utils import export_to_video

MIN_VAL = -0x8000_0000_0000_0000
MAX_VAL = 0xffff_ffff_ffff_ffff


def generate_thumbnail(input_path: str, output_path: str = None) -> str:
    try:
        thumbnail_path = output_path if output_path else input_path.replace(".mp4", "_thumbnail.jpg")
        (
            ffmpeg
            .input(input_path)
            .filter('scale', 512, -1)
            .output(thumbnail_path, vframes=1)
            .run(overwrite_output=True, quiet=True)
        )
        return thumbnail_path
    except ffmpeg.Error as e:
        raise Exception(f"Error generating thumbnail: {e.stderr.decode()}")


def to_h264(input_path: str, output_path: str) -> str:
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, vcodec="libx264", preset="superfast", f="mp4")
            .run(overwrite_output=True, quiet=True)
        )
        return output_path
    except ffmpeg.Error as e:
        raise Exception(f"Error converting to H264: {e.stderr.decode()}")


def export_frames_to_video(frames, vid_output_path: str) -> (str, str):
    print(f"Exporting to video at {datetime.now()}")

    tmp_path = "/tmp" + str(int(time.time())) + ".mp4"
    export_to_video(video_frames=frames, output_video_path=tmp_path)

    print(f"Converting to H264 at {datetime.now()}")
    vid_output_path = to_h264(input_path=tmp_path, output_path=vid_output_path)
    thumbnail_out_path = generate_thumbnail(input_path=vid_output_path)

    print(f"Video generated: {vid_output_path} at {datetime.now()}")
    return vid_output_path, thumbnail_out_path


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
) -> (str, str):
    seed = seed if seed else random.randint(MIN_VAL, MAX_VAL)

    # Generate video frames
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

        return export_frames_to_video(video_frames[0], output_path)


def generate_video_with_compel(
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
) -> (str, str):
    seed = seed if seed else random.randint(MIN_VAL, MAX_VAL)

    # Generate video frames
    with torch.inference_mode():
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
        return export_frames_to_video(video_frames[0], output_path)
