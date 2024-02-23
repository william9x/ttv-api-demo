import os

import torch
from animatelcm.models.unet import UNet3DConditionModel
from animatelcm.pipelines.pipeline_animation import AnimationPipeline
from animatelcm.scheduler.lcm_scheduler import LCMScheduler
from animatelcm.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint
from animatelcm.utils.lcm_utils import convert_lcm_lora
from animatelcm.utils.util import save_videos_grid
from diffusers import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from safetensors import safe_open
from transformers import CLIPTextModel, CLIPTokenizer


class AnimateController:
    def __init__(self):

        # config dirs
        self.basedir = os.getcwd()
        self.stable_diffusion_dir = os.path.join(self.basedir, "models", "StableDiffusion")
        self.motion_module_dir = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir = os.path.join(self.basedir, "models", "DreamBooth_LoRA")
        self.lcm_lora_path = "models/LCM_LoRA/sd15_t2v_beta_lora.safetensors"

        # config models
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.pipeline = None
        self.lora_model_state_dict = {}

        self.inference_config = OmegaConf.load("configs/inference.yaml")

    def update_stable_diffusion(self, sd_model):
        sd_model = os.path.join(self.stable_diffusion_dir, sd_model)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            sd_model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_model, subfolder="text_encoder").cuda()
        self.vae = AutoencoderKL.from_pretrained(
            sd_model, subfolder="vae").cuda()
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            sd_model, subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs)).cuda()

    def update_motion_module(self, motion_model):
        motion_model = os.path.join(self.motion_module_dir, motion_model)
        motion_module_state_dict = torch.load(motion_model, map_location="cpu")
        missing, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)
        del motion_module_state_dict
        assert len(unexpected) == 0

    def update_base_model(self, base_model):
        base_model = os.path.join(self.personalized_model_dir, base_model)
        base_model_state_dict = {}
        with safe_open(base_model, framework="pt", device="cpu") as f:
            for key in f.keys():
                base_model_state_dict[key] = f.get_tensor(key)

        converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_model_state_dict, self.vae.config)
        self.vae.load_state_dict(converted_vae_checkpoint)

        converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_model_state_dict, self.unet.config)
        self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

        del converted_unet_checkpoint
        del converted_vae_checkpoint
        del base_model_state_dict

    @torch.no_grad()
    def animate(
            self,
            prompt=None,
            neg_prompt=None,
            num_inference_steps=4,
            width=512,
            height=512,
            num_frames=16,
            guidance_scale=2.0,
            seed=-1,
            spatial_lora_slider=0.8,
            output_path="./output/out.mp4",
    ):
        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=LCMScheduler(**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        ).to("cuda")

        original_state_dict = {k: v.cpu().clone() for k, v in pipeline.unet.state_dict().items() if
                               "motion_modules." not in k}
        pipeline.unet = convert_lcm_lora(pipeline.unet, self.lcm_lora_path, spatial_lora_slider)

        pipeline.to("cuda")

        if seed != -1 and seed != "":
            torch.manual_seed(int(seed))
        else:
            torch.seed()
        seed = torch.initial_seed()

        with torch.autocast("cuda"):
            sample = pipeline(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                video_length=num_frames,
            ).videos

        pipeline.unet.load_state_dict(original_state_dict, strict=False)
        del original_state_dict

        save_videos_grid(sample, output_path)


if __name__ == "__main__":
    print("Starting server...")
    controller = AnimateController()
    controller.update_stable_diffusion("stable-diffusion-v1-5")
    controller.update_motion_module("sd15_t2v_beta_motion.ckpt")
    controller.update_base_model("realistic2.safetensors")
    print("Server started.")
