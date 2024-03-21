import os
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from animate_lcm_factory import AnimateDiffFactory
from magic_prompt_model import MagicPromptModel
from utils import generate_video

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:256,backend:cudaMallocAsync"

magicPrompt = MagicPromptModel()
factory = AnimateDiffFactory()
app = FastAPI()


class AnimateLCMInferReq(BaseModel):
    model_id: str | None
    prompt: str | None
    negative_prompt: str | None
    num_inference_steps: int = 25
    num_frames: int = 16
    width: int = 512
    height: int = 512
    guidance_scale: float = 2.0
    seed: int = 0


@app.post("/infer/animate_lcm", tags=["Infer"], response_class=FileResponse)
def infer(req: AnimateLCMInferReq):
    output_path = f"{os.getcwd()}/output/animate_lcm.mp4"
    try:
        pipe = factory.initialize_animate_diff_pipeline(req.model_id)
        video_path = generate_video(
            pipe=pipe,
            prompt=req.prompt,
            num_inference_steps=req.num_inference_steps,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            negative_prompt=req.negative_prompt,
            guidance_scale=req.guidance_scale,
            output_path=output_path,
            seed=req.seed if req.seed != 0 else None,
            to_h264=False
        )
    except Exception as e:
        print(e)
        return JSONResponse(content={"message": "Internal Server Error"}, status_code=500)

    return FileResponse(
        path=video_path,
        status_code=201,
        media_type="application/octet-stream",
        filename=f"animate_lcm.mp4",
    )


class MagicPromptInferReq(BaseModel):
    prompt: str = ""
    max_length: int = 77
    num_return_sequences: int = 4
    seed: int | None = None


@app.post("/infer/magic_prompt", tags=["Infer"], response_class=JSONResponse)
def infer(req: MagicPromptInferReq):
    try:
        start_time = time.time()
        response = magicPrompt.generate(
            prompt=req.prompt,
            max_length=None,
            num_return_sequences=req.num_return_sequences,
            seed=req.seed if req.seed != 0 else None,
        )
        end_time = time.time()
        duration = round(end_time - start_time, 3)
        return JSONResponse(content={"took": f"{duration} seconds", "new_prompt": response}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={"message": "Internal Server Error"}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    import transformers

    transformers.utils.move_cache()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
    )
