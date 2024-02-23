import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from animate_controller import AnimateController

app = FastAPI()

counter = 1


class AnimateLCMInferReq(BaseModel):
    prompt: str | None
    negative_prompt: str | None
    num_inference_steps: int = 25
    num_frames: int = 16
    width: int = 512
    height: int = 512
    guidance_scale: float = 2.0


@app.post("/infer/animate_lcm", tags=["Infer"], response_class=FileResponse)
def infer(req: AnimateLCMInferReq):
    global counter

    if req.num_inference_steps > 25:
        return JSONResponse(content={"message": "maximum num_inference_steps is 25"}, status_code=400)

    if req.num_frames > 20:
        return JSONResponse(content={"message": "maximum num_frames is 20"}, status_code=400)

    if req.guidance_scale > 2:
        return JSONResponse(content={"message": "maximum guidance_scale is 2.0"}, status_code=400)

    output_path = f"{os.getcwd()}/output/animate_lcm_{counter}.mp4"
    try:
        video_path = AnimateController().animate(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.num_inference_steps,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            guidance_scale=req.guidance_scale,
            output_path=output_path,
        )
    except Exception as e:
        print(e)
        return JSONResponse(content={"message": "Internal Server Error"}, status_code=500)

    counter += 1
    return FileResponse(
        path=video_path,
        status_code=201,
        media_type="application/octet-stream",
        filename=f"animate_lcm_{counter}.mp4",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
