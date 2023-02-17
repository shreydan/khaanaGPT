from transformers import pipeline

model_path = './khaanaGPT'

def create_prompt(ingredients):
    ingredients = ','.join([x.strip() for x in ingredients.split(',')])
    ingredients = ingredients.strip().replace(',','\n').lower()
    s = f"<|startoftext|>Ingredients:\n{ingredients}\n\nInstructions:\n"
    return s

def generate(prompt):
    recipe = model(prompt,
                    max_new_tokens=512,
                    penalty_alpha=0.5,
                    top_k=5,
                    pad_token_id=50259,
                    )[0]['generated_text']
    recipe = recipe.replace('<|startoftext|>','')
    return recipe


model = pipeline(task='text-generation',model=model_path)
