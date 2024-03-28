import copy
from datetime import datetime

from animate_lcm_factory import AnimateDiffFactory

MODEL_PATHS = [
    "amnd_general#runwayml/stable-diffusion-v1-5",
    "amnd_anime#liamhvn/nuke-colormax-anime",
    "amnd_pixar#liamhvn/disney-pixar-cartoon-b",
    "amnd_realistic#liamhvn/realistic-vision-v51",
    "amnd_pixel#liamhvn/stylizedpixel",
    "amnd_vangogh#liamhvn/van-gogh-diffusion",
]


class Model:
    def __init__(self, model_id, pipe):
        self.id = model_id
        self.pipe = pipe


class ModelList:
    def __init__(self):

        raw_img_path = "s3://0xcaptcha-label-studio-source/image_label_area_select/click on something that does not fit with the others/003213147c6c9343d5fa65a4a7a1a590.png"
        if raw_img_path.startswith("s3://"):
            raw_img_path = raw_img_path.replace("s3://", "http://84.247.149.30:9000/")
        else:
            raw_img_path = f"/data/{raw_img_path}"

        factory = AnimateDiffFactory()
        self._models = {}

        for path in MODEL_PATHS:
            self.init_models(factory, path)

    def init_models(self, factory, model_path):
        model_id_and_path = model_path.split("#")
        model_id = model_id_and_path[0]
        mode_path = model_id_and_path[1]
        print(f"Loading model ${model_id} from {mode_path}")

        pipe = factory.initialize_animate_diff_pipeline(mode_path)
        self._models[model_id] = Model(model_id=model_id, pipe=pipe)

    def get_pipe(self, mode_id):
        model = self._models.get(mode_id)
        print(f"Model selected: {model.id} at {datetime.now()}")
        pipe = copy.copy(model.pipe)
        print(f"Model copied: {model.id} at {datetime.now()}")
        return pipe
