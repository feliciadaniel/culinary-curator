from flask import Flask, request, jsonify
from flask import send_file
from flask_cors import CORS
from transformers import FlaxAutoModelForSeq2SeqLM, AutoTokenizer
import google.generativeai as genai

app = Flask(__name__)
CORS(app)
# enter your api key 
genai.configure(api_key="")

MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

prefix = "items: "
generation_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.95
}
special_tokens = tokenizer.all_special_tokens
tokens_map = {
    "<sep>": "--",
    "<section>": "\n"
}

def skip_special_tokens(text, special_tokens):
    for token in special_tokens:
        text = text.replace(token, "")
    return text

def target_postprocessing(texts, special_tokens):
    if not isinstance(texts, list):
        texts = [texts]
    new_texts = []
    for text in texts:
        text = skip_special_tokens(text, special_tokens)
        for k, v in tokens_map.items():
            text = text.replace(k, v)
        new_texts.append(text)
    return new_texts

def generate_recipe(ingredients):
    _inputs = [prefix + ingredients]
    inputs = tokenizer(
        _inputs,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="jax"
    )
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_kwargs
    )
    generated = output_ids.sequences
    generated_recipe = target_postprocessing(
        tokenizer.batch_decode(generated, skip_special_tokens=False),
        special_tokens
    )
    return generated_recipe

def generate_meal_plan(data):
    context = (
    f"Location: {data.get('location')}, "
    f"Age: {data.get('age')}, "
    f"Weekly Calorie Goal: {data.get('weekly_calorie_goal')}, "
    f"Allergies: {', '.join(data.get('allergies', '').split(',')) if data.get('allergies') else 'None'}, "
    f"Diet Type: {data.get('diet_type')}, "
    f"Meal Types: {', '.join(data.get('meal_type', '').split(',')) if data.get('meal_type') else 'None'}, "
    f"Meals Per Day: {data.get('meal_quantity')}, "
    f"Preferred Ingredients: {', '.join(data.get('preferred_ingredients', '').split(',')) if data.get('preferred_ingredients') else 'None'}, "
    f"Avoided Ingredients: {', '.join(data.get('avoided_ingredients', '').split(',')) if data.get('avoided_ingredients') else 'None'}, "
    f"Cooking Skill Level: {data.get('cooking_skill_level')}, "
    f"Max Cooking Time: {data.get('cooking_time')} minutes, "
    f"Difficulty Level: {data.get('difficulty_level')}, "
    f"Cuisine Preferences: {', '.join(data.get('cuisine_preferences', [])) if isinstance(data.get('cuisine_preferences', []), list) else 'None'}, "
    f"Days in Plan: {data.get('days_in_plan')}, "
    f"Budget Constraints: {data.get('budget_constraints')}"
)

    prompt = f"""
    I want a weekly meal plan with recipes tailored to the following specifications:

    {context}

    Please generate recipes according to the specifications, ensuring a balance of nutrition, variety, and simplicity suited to my preferences. Provide a short description, ingredient list, and step-by-step cooking instructions for each recipe.
    Make sure to adhere to the specifications.The number of recipes generated should be according to the 'days in plan' and 'meals per day',
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    ingredients = data.get('ingredients', '').strip()  
    if not ingredients:
        return jsonify({'error': 'No ingredients provided'}), 400 
    recipe = generate_recipe(ingredients)
    return jsonify({'recipe': recipe})

@app.route('/meal-plan', methods=['POST'])
def meal_plan():
    data = request.json
    # required_fields = ['location', 'age', 'weekly_calorie_goal', 'dietary_restrictions', 'meal_specifications', 'recipe_preferences', 'meal_plan_customization']
    # for field in required_fields:
    #     if field not in data:
    #         return jsonify({'error': f'Missing required field: {field}'}), 400
    
    meal_plan_response = generate_meal_plan(data)

    # Instead of saving to a file, return it in JSON response
    return jsonify({'meal_plan': meal_plan_response})


if __name__ == '__main__':
    app.run(debug=True)
