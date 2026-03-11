from transformers import pipeline

class LLM:

    def __init__(self):

        self.generator = pipeline(
            "text-generation",
            model="google/flan-t5-base"
        )

    
    def generate(self, prompt):


        result = self.generator(prompt, max_length=120)

        answer = result[0]["generated_text"]

        return answer.strip()