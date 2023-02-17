import streamlit as st
from transformers import pipeline

model_path = './khaanaGPT'

def create_prompt(ingredients):
    ingredients = ','.join([x.strip() for x in ingredients.split(',')])
    ingredients = ingredients.strip().replace(',','\n').lower()
    s = f"<|startoftext|>Ingredients:\n{ingredients}\n\nInstructions:\n"
    return s


model = pipeline(task='text-generation',model=model_path)

st.title('खानाGPT')
st.caption("it's not perfect, might ± ingredients, it's just a fun project to showcase fine-tuning a causal language model like GPT-2")

with st.container():
    st.subheader('Ingredients')
    text = st.text_input('separate ingredients with a comma.',placeholder='e.g. chicken, tomatoes, red bell peppers, ...')
    text = text.strip()
    submitted = st.button('Create Recipe',type='primary')

with st.container():
    if len(text)!=0 or submitted:
        prompt = create_prompt(text)
        with st.spinner('cooking something for you...'):
            recipe = model(prompt,
                    max_new_tokens=512,
                    penalty_alpha=0.5,
                    top_k=5,
                    pad_token_id=50259,
                    )[0]['generated_text']
            recipe = recipe.replace('<|startoftext|>','')
        st.write(recipe)