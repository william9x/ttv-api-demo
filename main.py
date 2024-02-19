import os
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from infer_animatelcm import AnimateLCMInfer
from infer_zeroscope import ZeroScopeInfer

app = FastAPI()

lock = False


class ZeroScopeInferReq(BaseModel):
    prompt: str | None
    num_inference_steps: int = 25
    num_upscale_steps: int = 25
    height: int = 576
    width: int = 1024
    upscale: bool = False
    upscaled_height: int = 576
    upscaled_width: int = 1024
    num_frames: int = 30
    strength: float = 0.6
    negative_prompt: str | None
    guidance_scale: float = 10.0


@app.post("/infer/zeroscope", tags=["Infer"], response_class=FileResponse)
def infer(req: ZeroScopeInferReq):
    global lock
    if lock:
        return JSONResponse(content={"message": "Server is busy"}, status_code=503)
    lock = True

    now = datetime.now().strftime("%m%d_%H%M%S")
    output_path = f"{os.getcwd()}/output/{now}.mp4"
    video_path = ZeroScopeInfer().generate_video(
        prompt=req.prompt,
        num_inference_steps=req.num_inference_steps,
        num_upscale_steps=req.num_upscale_steps,
        height=req.height,
        width=req.width,
        upscale=req.upscale,
        upscaled_height=req.upscaled_height,
        upscaled_width=req.upscaled_width,
        num_frames=req.num_frames,
        strength=req.strength,
        output_path=output_path,
        negative_prompt=req.negative_prompt,
        guidance_scale=req.guidance_scale,
    )

    lock = False
    return FileResponse(
        path=video_path,
        status_code=201,
        media_type="application/octet-stream",
        filename=f"{now}.mp4",
    )


class AnimateLCMInferReq(BaseModel):
    prompt: str | None
    num_inference_steps: int = 25
    height: int = 576
    width: int = 1024
    num_frames: int = 30
    negative_prompt: str | None
    guidance_scale: float = 10.0
    strength: float = 2.0


@app.post("/infer/animate_lcm", tags=["Infer"], response_class=FileResponse)
def infer(req: AnimateLCMInferReq):
    global lock
    if lock:
        return JSONResponse(content={"message": "Server is busy"}, status_code=503)
    lock = True

    now = datetime.now().strftime("%m%d_%H%M%S")
    output_path = f"{os.getcwd()}/output/{now}.mp4"
    try:
        video_path = AnimateLCMInfer().generate_video(
            prompt=req.prompt,
            num_inference_steps=req.num_inference_steps,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            negative_prompt=req.negative_prompt,
            guidance_scale=req.guidance_scale,
            strength=req.strength,
            output_path=output_path,
        )
    except Exception as e:
        lock = False
        print(e)
        return JSONResponse(content={"message": "Internal Server Error"}, status_code=500)

    lock = False
    return FileResponse(
        path=video_path,
        status_code=201,
        media_type="application/octet-stream",
        filename=f"{now}.mp4",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
