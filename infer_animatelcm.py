import torch
from DeepCache import DeepCacheSDHelper

from diffusers import AnimateDiffPipeline, MotionAdapter, LCMScheduler
from diffusers.utils import export_to_video


class AnimateLCMInfer:
    def __init__(self):
        # config models
        self.motion_adapter = "wangfuyun/AnimateLCM"

        # config lora
        self.lora_model = "wangfuyun/AnimateLCM"
        self.lora_name = "AnimateLCM_sd15_t2v_lora.safetensors"
        self.lora_adapter_name = "lcm-lora"
        self.lora_adapter_weight = 0.8

    # Function to initialize the AnimateDiffPipeline
    def initialize_animate_diff_pipeline(self, dtype=torch.float16, chunk_size=1, dim=1,
                                         model_path="emilianJR/epiCRealism"):
        adapter = MotionAdapter.from_pretrained(
            self.motion_adapter,
            torch_dtype=dtype,
        )

        pipe = AnimateDiffPipeline.from_pretrained(
            model_path,
            motion_adapter=adapter,
            torch_dtype=dtype,
            # max_memory={0: "8GiB"}
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
        pipe.safety_checker = None

        pipe.load_lora_weights(self.lora_model, weight_name=self.lora_name, adapter_name=self.lora_adapter_name)

        pipe.set_adapters([self.lora_adapter_name], [self.lora_adapter_weight])

        # Must be in order
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()

        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()

        return pipe

    # Function to export frames to a video
    def export_frames_to_video(self, frames, output_path):
        video_path = export_to_video(frames, output_video_path=output_path)
        print(f"Video generated: {video_path}")
        return video_path

    # Main function to generate videos
    def generate_video(self, prompt=None, num_inference_steps=None, height=None, width=None,
                       num_frames=None, output_path=None, negative_prompt=None, guidance_scale=None, strength=None,
                       model_path=None):
        prompt = prompt.strip() if prompt is not None else "Space scenery"
        num_inference_steps = int(num_inference_steps) if num_inference_steps is not None else 30
        height = int(height) if height is not None else 576
        width = int(width) if width is not None else 1024
        num_frames = int(num_frames) if num_frames is not None else 30
        negative_prompt = negative_prompt.strip() if negative_prompt is not None else None  # Convert to a single string
        guidance_scale = float(guidance_scale) if guidance_scale is not None else 10.0
        output_path = output_path or "./output/"

        # Create the pipeline once outside the loop
        pipe = self.initialize_animate_diff_pipeline(model_path=model_path)

        # Generate video frames
        video_frames = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device='cuda'),
        ).frames

        del pipe

        return self.export_frames_to_video(video_frames[0], output_path)
