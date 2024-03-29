import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, LCMScheduler


class AnimateDiffFactory:
    def __init__(self):
        # config models
        self.motion_adapter = "wangfuyun/AnimateLCM"

        # config lora
        self.lora_model = "wangfuyun/AnimateLCM"
        self.lora_name = "AnimateLCM_sd15_t2v_lora.safetensors"
        self.lora_adapter_name = "lcm-lora"
        self.lora_adapter_weight = 0.8

    def initialize_animate_diff_pipeline(self, model_path=None) -> AnimateDiffPipeline:
        print(f"[AnimateDiffFactory] Loading motion adapter for {model_path}")
        adapter = MotionAdapter.from_pretrained(
            self.motion_adapter,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        print(f"[AnimateDiffFactory] Loading model from {model_path}")
        pipe: AnimateDiffPipeline = AnimateDiffPipeline.from_pretrained(
            model_path,
            motion_adapter=adapter,
            torch_dtype=torch.float16,
            use_safetensors=True,
            device_map="auto"
        )

        print(f"[AnimateDiffFactory] Loading scheduler for {model_path}")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
        pipe.safety_checker = None

        print(f"[AnimateDiffFactory] Loading lora for {model_path}")
        pipe.load_lora_weights(
            self.lora_model,
            weight_name=self.lora_name,
            adapter_name=self.lora_adapter_name
        )

        pipe.set_adapters([self.lora_adapter_name], [self.lora_adapter_weight])

        # Must be in order
        print(f"[AnimateDiffFactory] Optimizing model {model_path}")
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()

        print(f"[AnimateDiffFactory] Model {model_path} loaded")
        return pipe
