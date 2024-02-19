import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video


class CerspenseInferer:

    # Function to initialize the DiffusionPipeline
    def initialize_diffusion_pipeline(self, model_name, dtype=torch.float16, chunk_size=1, dim=1):
        print(f"Initializing the pipeline with model: {model_name}")
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        # pipe.unet.enable_forward_chunking(chunk_size=chunk_size, dim=dim)
        return pipe

    # Function to export frames to a video
    def export_frames_to_video(self, frames, output_path):
        video_path = export_to_video(frames, output_video_path=output_path)
        print(f"Video generated: {video_path}")
        return video_path

    # Main function to generate videos
    def generate_video(self, prompt=None, num_inference_steps=None, num_upscale_steps=None, height=None, width=None,
                       upscale=None, upscaled_height=None, upscaled_width=None, num_frames=None, strength=None,
                       output_path=None, negative_prompt=None, guidance_scale=None):
        prompt = prompt.strip() if prompt is not None else "Space scenery"
        num_inference_steps = int(num_inference_steps) if num_inference_steps is not None else 30
        num_upscale_steps = int(num_upscale_steps) if num_upscale_steps is not None else 30
        height = int(height) if height is not None else 576
        width = int(width) if width is not None else 1024
        upscale = upscale if upscale is not None else False
        upscaled_height = int(upscaled_height) if upscaled_height is not None else 576
        upscaled_width = int(upscaled_width) if upscaled_width is not None else 1024
        num_frames = int(num_frames) if num_frames is not None else 30
        strength = float(strength) if strength is not None else 0.6
        negative_prompt = negative_prompt.strip() if negative_prompt is not None else None  # Convert to a single string
        guidance_scale = float(guidance_scale) if guidance_scale is not None else 10.0
        output_path = output_path or "./output/"

        # Create the pipeline once outside the loop
        pipe = initialize_diffusion_pipeline("cerspense/zeroscope_v2_576w")

        # Generate video frames
        video_frames = pipe(prompt, num_inference_steps=num_inference_steps, height=height, width=width,
                            num_frames=num_frames, negative_prompt=negative_prompt,
                            guidance_scale=guidance_scale).frames

        if upscale:
            # Clear memory before using the pipeline with larger model
            del pipe
            torch.cuda.empty_cache()

            pipe = initialize_diffusion_pipeline("cerspense/zeroscope_v2_XL")

            upscaled_size = (upscaled_width, upscaled_height)
            video = [Image.fromarray((frame * 255).astype(np.uint8)).resize(upscaled_size) for frame in video_frames[0]]

            video_frames = pipe(prompt, num_inference_steps=num_upscale_steps, video=video, strength=strength,
                                negative_prompt=negative_prompt, guidance_scale=guidance_scale).frames

            # Clear memory after using the pipeline with larger model
            del pipe
            torch.cuda.empty_cache()

            # Re-initialize the pipeline with the smaller model
            # pipe = initialize_diffusion_pipeline("cerspense/zeroscope_v2_576w")

        return export_frames_to_video(video_frames[0], output_path)
