import gradio as gr
from transformers import pipeline
import time

# Use a model specifically for English to Indic languages
translator = pipeline('translation', model='Helsinki-NLP/opus-mt-en-inc')

def translate_english_to_assamese(text):
    """Function to translate English text to Assamese"""
    if not text.strip():
        return "Please enter some English text.", ""
        
    start_time = time.time()
    
    try:
        # Force Assamese output by using language code prefix
        forced_text = f">>asm<< {text}"
        
        # Use the pre-trained model
        result = translator(forced_text)[0]
        translated_text = result['translation_text']
        
        # If still in English, try without forcing
        if translated_text.isascii():
            result = translator(text)[0]
            translated_text = result['translation_text']
        
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        
        return translated_text, f"Translation generated in {time_taken} seconds."
    
    except Exception as e:
        return f"Error: {str(e)}", "An error occurred during translation."

# Create the Gradio interface
with gr.Blocks(title="BhashaAI Translator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌐 BhashaAI: English to Assamese Translator")
    gr.Markdown("A neural machine translation interface for English to Assamese translation.")
    
    with gr.Row():
        with gr.Column():
            english_input = gr.Textbox(label="Input English Text", lines=3, 
                                      placeholder="Type English text here...")
            translate_btn = gr.Button("Translate", variant="primary")
        
        with gr.Column():
            assamese_output = gr.Textbox(label="Translated Assamese Text", lines=3, 
                                        interactive=False)
            time_output = gr.Textbox(label="Status", interactive=False)
    
    gr.Examples(
        examples=[
            ["Hello"],
            ["Thank you"],
            ["Good morning"],
            ["How are you?"]
        ],
        inputs=english_input
    )
    
    translate_btn.click(
        fn=translate_english_to_assamese,
        inputs=english_input,
        outputs=[assamese_output, time_output]
    )

demo.launch()
