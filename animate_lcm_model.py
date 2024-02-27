from animate_lcm_factory import AnimateDiffFactory


class Model:
    def __init__(self, model_id, factory, path):
        self.id = model_id
        self.pipe = factory.initialize_animate_diff_pipeline(path)


class ModelList:
    def __init__(self):
        model_paths = {
            "anmd_base": "runwayml/stable-diffusion-v1-5",
            "anmd_cartoon": "stablediffusionapi/disney-pixar-cartoon",
            "anmd_anime": "stablediffusionapi/nuke-colormax-anime",
            "anmd_realistic": "stablediffusionapi/realistic-vision-v51",
            "anmd_pixel": "stablediffusionapi/stylizedpixel",
            "anmd_van_gogh": "stablediffusionapi/van-gogh-diffusion",
        }
        self._models = {}

        factory = AnimateDiffFactory()
        for model_id in model_paths:
            self._models[model_id] = Model(model_id=model_id, factory=factory, path=model_paths[model_id])

    def get_pipe(self, mode_id):
        return self._models.get(mode_id)
