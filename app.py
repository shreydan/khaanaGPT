from transformers import pipeline
import gradio as gr

import warnings
warnings.simplefilter('ignore')

model_path = './khaanaGPT'

contrastive_search_config = dict(
    penalty_alpha = 0.5,
    top_k = 5,
    max_new_tokens = 512,
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

def wrapper(ingredients):
    prompt = create_prompt(ingredients)
    recipe = generate(prompt)
    return recipe


with gr.Blocks() as demo:
    gr.HTML('<center><h1>KhaanaGPT</h1></center>')
    ingredients = gr.Textbox(label="ingredients",placeholder='separate the ingredients with a comma.')
    output = gr.Textbox(label="recipe",lines=15,)
    greet_btn = gr.Button("Create a recipe!")
    gr.Examples(['yellow dal, turmeric, green peas, tomatoes',
                'chicken, soy sauce, tomato sauce, vinegar'],
                inputs=ingredients
            )
    greet_btn.click(fn=wrapper, inputs=ingredients, outputs=output)

demo.launch()