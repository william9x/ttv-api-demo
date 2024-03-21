import random

from transformers import pipeline, set_seed, Pipeline


class MagicPromptModel:
    def __init__(self, model_path: str = None, tokenizer: str = None, ideas_file_path: str = None):
        # config models
        self.task = "text-generation"
        self.model = model_path if model_path else "Gustavosta/MagicPrompt-Stable-Diffusion"
        self.tokenizer = tokenizer if tokenizer else "gpt2"

        with open(ideas_file_path if ideas_file_path else "resources/ideas.txt", "r") as f:
            line = f.readlines()
        self.ideas = line
        self.pipe = self.initialize_pipeline()

    def initialize_pipeline(self) -> Pipeline:
        print(f"[MagicPromptFactory] Loading motion adapter for {self.model}")
        return pipeline(task=self.task, model=self.model, tokenizer=self.tokenizer)

    def generate(self,
                 prompt: str = None,
                 max_length: int = None,
                 num_return_sequences: int = 4,
                 seed: int = None):
        r"""
        max_length (`int`, *optional*, defaults to 20):
            Maximum length that will be used by default in the `generate` method of the model.
        num_return_sequences (`int`, *optional*, defaults to 1):
            Number of independently computed returned sequences for each element in the batch that will be used by
            default in the `generate` method of the model.
        """
        prompt = prompt if prompt \
            else self.ideas[random.randrange(0, len(self.ideas))].replace("\n", "").lower().capitalize()

        seed = seed if seed else random.randint(100, 1000000)
        set_seed(seed)

        max_length = (max_length - len(prompt)) if max_length else (len(prompt) + random.randint(60, 90))

        response = self.pipe(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences
        )
        response_list = []
        for x in response:
            resp = x["generated_text"].strip()
            if resp != prompt and len(resp) > (len(prompt) + 4) and resp.endswith((":", "-", "—")) is False:
                response_list.append(resp + '\n')

        response_end = "\n".join(response_list)
        response_end = (response_end
                        .replace("\n", " ")
                        .replace("<", "")
                        .replace(">", ""))
        return response_end
