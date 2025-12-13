import platform
from llama_cpp import Llama

MODEL_PATH = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

MAX_CONTEXT = 8192

class LocalLLM:
    def __init__(self, model_path=MODEL_PATH, n_ctx=MAX_CONTEXT):
        use_metal = False
        if platform.system() == "Darwin":
            try:
                from torch.backends import mps
                use_metal = mps.is_available()
            except ImportError:
                use_metal = False
        
        print(f"Loading LLaMA model from {model_path} (Metal GPU: {use_metal})...")
        self.model = Llama(model_path=model_path, verbose=False, metal=use_metal, n_ctx=n_ctx)

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000):
        """
        system_prompt: system message content
        user_prompt: user query/content
        """

        full_prompt = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        response = self.model(full_prompt, max_tokens=max_tokens)
        return response['choices'][0]['text']

