import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, LCMScheduler, PixArtAlphaPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video, is_xformers_available


class AnimateLCMInfer:
    def __init__(self):
        # config dirs
        # self.basedir = os.getcwd()
        # self.stable_diffusion_dir = os.path.join(
        #     self.basedir, "models", "StableDiffusion")
        # self.motion_module_dir = os.path.join(
        #     self.basedir, "models", "Motion_Module")
        # self.personalized_model_dir = os.path.join(
        #     self.basedir, "models", "DreamBooth_LoRA")
        # self.savedir = os.path.join(
        #     self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        # self.savedir_sample = os.path.join(self.savedir, "sample")
        # self.lcm_lora_path = "models/LCM_LoRA/sd15_t2v_beta_lora.safetensors"
        # os.makedirs(self.savedir, exist_ok=True)

        # self.stable_diffusion_list = []
        # self.motion_module_list = []
        # self.personalized_model_list = []

        # self.refresh_stable_diffusion()
        # self.refresh_motion_module()
        # self.refresh_personalized_model()

        # config models
        self.motion_adapter = "wangfuyun/AnimateLCM"
        self.base_image_model = "PixArt-alpha/PixArt-XL-2-512x512"

        # config lora
        self.lora_model = "wangfuyun/AnimateLCM"
        self.lora_name = "sd15_lora_beta.safetensors"
        self.lora_adapter_name = "lcm-lora"
        self.lora_adapter_weight = 0.8

        # self.inference_config = OmegaConf.load("configs/inference.yaml")

    # Function to initialize the AnimateDiffPipeline
    def initialize_animate_diff_pipeline(self, dtype=torch.float16, chunk_size=1, dim=1):
        adapter = MotionAdapter.from_pretrained(self.motion_adapter, torch_dtype=dtype)

        pipe = PixArtAlphaPipeline.from_pretrained(self.base_image_model, motion_adapter=adapter, torch_dtype=dtype)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)

        # pipe = AnimateDiffPipeline.from_pretrained(self.base_image_model, motion_adapter=adapter, torch_dtype=dtype)
        # pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
        # pipe.load_lora_weights(self.lora_model, weight_name=self.lora_name, adapter_name=self.lora_adapter_name)
        # pipe.set_adapters([self.lora_adapter_name], [self.lora_adapter_weight])

        if is_xformers_available():
            pipe.enable_xformers_memory_efficient_attention()

        # pipe.unet.enable_forward_chunking(chunk_size=chunk_size, dim=dim)
        # pipe.enable_vae_slicing()

        # Must be last
        # pipe.enable_model_cpu_offload()
        pipe.to("cuda")

        return pipe

    # Function to export frames to a video
    def export_frames_to_video(self, frames, output_path):
        video_path = export_to_video(frames, output_video_path=output_path)
        print(f"Video generated: {video_path}")
        return video_path

    # Main function to generate videos
    def generate_video(self, prompt=None, num_inference_steps=None, height=None, width=None,
                       num_frames=None, output_path=None, negative_prompt=None, guidance_scale=None, strength=None):
        prompt = prompt.strip() if prompt is not None else "Space scenery"
        num_inference_steps = int(num_inference_steps) if num_inference_steps is not None else 30
        height = int(height) if height is not None else 576
        width = int(width) if width is not None else 1024
        num_frames = int(num_frames) if num_frames is not None else 30
        negative_prompt = negative_prompt.strip() if negative_prompt is not None else None  # Convert to a single string
        guidance_scale = float(guidance_scale) if guidance_scale is not None else 10.0
        output_path = output_path or "./output/"

        # Create the pipeline once outside the loop
        pipe = self.initialize_animate_diff_pipeline()

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

        return self.export_frames_to_video(video_frames[0], output_path)
