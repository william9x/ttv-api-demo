from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from animate_lcm_model import ModelList
from utils import generate_video

app = FastAPI()
model_list = ModelList()


class AnimateLCMInferReq(BaseModel):
    prompt: str | None
    negative_prompt: str | None
    num_inference_steps: int = 25
    num_frames: int = 16
    width: int = 512
    height: int = 512
    guidance_scale: float = 2.0
    output_file_path: str | None
    model_id: str | None
    seed: int = None


@app.post("/api/v1/infer/animate_lcm", tags=["Infer"], response_class=JSONResponse)
async def infer(req: AnimateLCMInferReq):
    if req.output_file_path is None:
        return JSONResponse(content={"message": "missing output path"}, status_code=400)

    print(req)
    print(f"OUTPUT PATH: {req.output_file_path}")
    try:
        pipe = model_list.get_pipe(req.model_id)
        video_path, thumbnail_path = generate_video(
            pipe=pipe,
            prompt=req.prompt,
            num_inference_steps=8,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            negative_prompt=req.negative_prompt,
            guidance_scale=1,
            output_path=req.output_file_path,
            seed=None,
        )
    except Exception as e:
        print(e)
        return JSONResponse(content={"message": "Internal Server Error"}, status_code=500)

    return JSONResponse(content={
        "video_path": video_path,
        "thumbnail_path": thumbnail_path,
    }, status_code=201)
