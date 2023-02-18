from transformers import pipeline
import warnings
warnings.simplefilter('ignore')

model_path = './khaanaGPT'

contrastive_search_config = dict(
    penalty_alpha = 0.5,
    top_k = 5,
    max_new_tokens = 300,
    pad_token_id = 50259
)

model = pipeline('text-generation',model=model_path)

def create_prompt(ingredients):
    ingredients = ','.join([x.strip() for x in ingredients.split(',')])
    ingredients = ingredients.strip().replace(',','\n').lower()
    s = f"<|startoftext|>Ingredients:\n{ingredients}\n\nInstructions:\n"
    return s

def generate(prompt):
    recipe = model(prompt,**contrastive_search_config)[0]['generated_text']
    recipe = recipe.replace('<|startoftext|>','')
    return recipe




sample = 'tomatoes, yellow dal, turmeric, oil'
prompt = create_prompt(sample)
recipe = generate(prompt)
print(recipe)
