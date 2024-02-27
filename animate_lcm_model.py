from animate_lcm_factory import AnimateDiffFactory

MODEL_PATHS = [
    "anmd_base#runwayml/stable-diffusion-v1-5",
    "anmd_anime#stablediffusionapi/nuke-colormax-anime",
    # "anmd_cartoon#stablediffusionapi/disney-pixar-cartoon",
    # "anmd_realistic#stablediffusionapi/realistic-vision-v51",
    # "anmd_pixel#stablediffusionapi/stylizedpixel",
    # "anmd_van_gogh#stablediffusionapi/van-gogh-diffusion",
]


class Model:
    def __init__(self, model_id, pipe):
        self.id = model_id
        self.pipe = pipe


class ModelList:
    def __init__(self):
        factory = AnimateDiffFactory()
        self._models = {}

        for path in MODEL_PATHS:
            self.init_models(factory, path)

        print(f"------Models: {self._models}")

    def init_models(self, factory, model_path):
        model_id_and_path = model_path.split("#")
        model_id = model_id_and_path[0]
        mode_path = model_id_and_path[1]
        print(f"Loading model ${model_id} from {mode_path}")

        pipe = factory.initialize_animate_diff_pipeline(mode_path)
        self._models[model_id] = Model(model_id=model_id, pipe=pipe)

    def get_pipe(self, mode_id):
        model = self._models.get(mode_id)
        print(f"Model selected: {model.id}")
        return model.pipe
