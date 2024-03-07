import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from animate_lcm_factory import AnimateDiffFactory
from utils import generate_video

app = FastAPI()
factory = AnimateDiffFactory()
lock = False


class AnimateLCMInferReq(BaseModel):
    model_id: str | None
    prompt: str | None
    negative_prompt: str | None
    num_inference_steps: int = 25
    num_frames: int = 16
    width: int = 512
    height: int = 512
    guidance_scale: float = 2.0


@app.post("/infer/animate_lcm", tags=["Infer"], response_class=FileResponse)
def infer(req: AnimateLCMInferReq):
    global lock
    lock = True
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
        )
    except Exception as e:
        print(e)
        lock = False
        return JSONResponse(content={"message": "Internal Server Error"}, status_code=500)

    lock = False
    return FileResponse(
        path=video_path,
        status_code=201,
        media_type="application/octet-stream",
        filename=f"animate_lcm.mp4",
    )


if __name__ == "__main__":
    import uvicorn
    import transformers

    transformers.utils.move_cache()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
    )
