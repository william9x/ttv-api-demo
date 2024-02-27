from multiprocessing import Pool, cpu_count

from animate_lcm_factory import AnimateDiffFactory

MODEL_PATHS = [
    "anmd_base#runwayml/stable-diffusion-v1-5",
    "anmd_cartoon#stablediffusionapi/disney-pixar-cartoon",
    "anmd_anime#stablediffusionapi/nuke-colormax-anime",
    "anmd_realistic#stablediffusionapi/realistic-vision-v51",
    "anmd_pixel#stablediffusionapi/stylizedpixel",
    "anmd_van_gogh#stablediffusionapi/van-gogh-diffusion",
]


class Model:
    def __init__(self, model_id, pipe):
        self.id = model_id
        self.pipe = pipe


class ModelList:
    def __init__(self):
        self.factory = AnimateDiffFactory()
        self._models = {}

        pool = Pool(cpu_count())
        print(f"Init a pool with ${cpu_count()} workers")

        pool.map(self.init_models, MODEL_PATHS)
        pool.close()
        pool.join()

    def init_models(self, model_path):
        model_id_and_path = model_path.split("#")
        model_id = model_id_and_path[0]
        mode_path = model_id_and_path[1]
        print(f"Loading model ${model_id} from {mode_path}")

        pipe = self.factory.initialize_animate_diff_pipeline(mode_path)
        self._models[model_id] = Model(model_id=model_id, pipe=pipe)

    def get_pipe(self, mode_id):
        return self._models.get(mode_id)
