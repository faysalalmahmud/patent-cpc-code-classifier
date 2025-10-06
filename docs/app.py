from flask import Flask, render_template, request
from gradio_client import Client
import json
import os # Import os for environment variables if needed


# Gradio Space ID (used by the Gradio Client)
HF_SPACE_ID = "faysalalmahmud/Patent-CPC-Code-Classifier"

app = Flask(__name__)

# Initialize the Gradio Client globally
# This creates the client once when the app starts.
try:
    client = Client(HF_SPACE_ID)
except Exception as e:
    print(f"ERROR: Could not initialize Gradio Client for {HF_SPACE_ID}. Details: {e}")
    client = None

def call_classifier_api(text_input):
    """
    Calls the Hugging Face Gradio API using gradio_client.
    Returns a tuple: (labels_list, error_message)
    """
    if not client:
        return None, "Gradio Client failed to initialize. Check the space name and status."
        
    try:
        # Use the working client.predict() method to call the API
        result = client.predict(
            abstract=text_input,
            api_name="/predict"
        )
        
        # Log the full result for debugging purposes
        print("--- API Response Received from Gradio Client ---")
        print(json.dumps(result, indent=2))
        
        # --- GRADIO OUTPUT PARSING LOGIC ---
        # Gradio 'Label' components typically return a dictionary with a 'confidences' key.
        
        if result and isinstance(result, dict) and 'confidences' in result:
            
            # Format the list of dicts into the expected structure for the template
            labels = []
            for item in result['confidences']:
                # Ensure score is between 0 and 1
                score = float(item.get('confidence', 0.0))
                
                labels.append({
                    'label': item.get('label', 'N/A'),
                    'score': score
                })
            
            # Sort by score in descending order
            labels.sort(key=lambda x: x['score'], reverse=True)
            return labels, None
        else:
            return None, "API Response Error: The model returned an unexpected structure or no data."

    except Exception as e:
        print(f"--- Gradio Client Call Failed --- \n{e}")
        return None, f"Gradio Client Error: Failed to get prediction from model. Details: {e}"


@app.route('/', methods=['GET', 'POST'])
def index():
    labels = None
    text_input = ""
    api_error = None

    if request.method == 'POST':
        # Get text from the form
        text_input = request.form.get('text_input', '').strip()

        if text_input:
            # Call the external API
            labels, api_error = call_classifier_api(text_input)
        else:
            api_error = "Please enter some text to classify."

    # Pass all variables to the template for rendering
    return render_template('index.html', 
                           labels=labels, 
                           text_input=text_input, 
                           api_error=api_error)


if __name__ == '__main__':
    app.run(debug=False)