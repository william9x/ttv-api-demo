from typing import Dict

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class AnimateDiffLightningFactory:
    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.float16

        # config models
        self.motion_adapter = "ByteDance/AnimateDiff-Lightning"
        self._4step_file = f"animatediff_lightning_4step_diffusers.safetensors"
        self._8step_file = f"animatediff_lightning_8step_diffusers.safetensors"

        # config lora
        self.motions: Dict = {
            "Zoom_in": "guoyww/animatediff-motion-lora-zoom-in",
            "Zoom_out": "guoyww/animatediff-motion-lora-zoom-out",
            "Tilt_up": "guoyww/animatediff-motion-lora-tilt-up",
            "Tilt_down": "guoyww/animatediff-motion-lora-tilt-down",
            "Pan_left": "guoyww/animatediff-motion-lora-pan-left",
            "Pan_right": "guoyww/animatediff-motion-lora-pan-right",
            "Roll_left": "guoyww/animatediff-motion-lora-rolling-anticlockwise",
            "Roll_right": "guoyww/animatediff-motion-lora-rolling-clockwise",
        }

        self.pipelines: Dict[str, AnimateDiffPipeline] = {}

    def initialize_animate_diff_pipeline(self, model_path: str = None, motion: str = None):
        pipe_key = f"{model_path}|{motion if motion else 'None'}"
        if pipe_key in self.pipelines:
            print(f"[AnimateDiffLightningFactory] Model {pipe_key} loaded")
            return self.pipelines[pipe_key]

        print(f"[AnimateDiffLightningFactory] Loading motion adapter for {model_path}")
        adapter = MotionAdapter().to(self.device, self.dtype)
        adapter.load_state_dict(load_file(hf_hub_download(self.motion_adapter, self._8step_file), device=self.device))

        print(f"[AnimateDiffLightningFactory] Loading base model for {model_path}")
        pipe = AnimateDiffPipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            motion_adapter=adapter,
        ).to(self.device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            beta_schedule="linear"
        )

        print(f"[AnimateDiffLightningFactory] Loading lora for {model_path}")
        motion = self.motions.get(motion, None) if motion else None
        if motion:
            pipe.load_lora_weights(self.motions.get(motion), adapter_name="motion")
            pipe.set_adapters(["motion"], [0.8])

        # Must be in order
        # print(f"[AnimateDiffLightningFactory] Optimizing model {model_path}")
        # pipe.enable_vae_slicing()
        # pipe.enable_model_cpu_offload()

        # pipe.to("cuda")
        # tomesd.apply_patch(pipe, ratio=0.5)

        # helper = DeepCacheSDHelper(pipe=pipe)
        # helper.set_params(cache_interval=3, cache_branch_id=0)
        # helper.enable()

        self.pipelines[pipe_key] = pipe

        print(f"[AnimateDiffLightningFactory] Model {pipe_key} loaded")
        return pipe
