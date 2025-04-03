from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading
import power_converter.boost_gui as boost
import power_converter.buck_gui as buck
import power_converter.buckboost_gui as buckboost
import power_converter.cuk_gui as cuk
import power_converter.flyback_gui as flyback
import power_converter.forward_gui as forward
import power_converter.fullbridge_gui as fullbridge
import power_converter.halfbridge_gui as halfbridge
import power_converter.pushpull_gui as pushpull
import power_converter.sepic_gui as sepic
import power_converter.zeta_gui as zeta

app = Flask(__name__, template_folder="templates", static_folder="static")

# Configure session storage
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Load model in a separate thread
model = None
tokenizer = None
is_model_loaded = False


def load_model():
    global model, tokenizer, is_model_loaded
    try:
        checkpoint = "vrpatel/power-converter"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
        is_model_loaded = True
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

# Start loading the model in the background
threading.Thread(target=load_model, daemon=True).start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    if not is_model_loaded:
        return jsonify({"error": "Model is still loading. Please wait."}), 503

    user_text = request.json.get("message", "").strip()
    if not user_text:
        return jsonify({"error": "Empty message."}), 400

    # Initialize chat history in session if not present
    if "chat_history" not in session:
        session["chat_history"] = []

    # Append user input to chat history
    session["chat_history"].append({"role": "system", "content": "You are a helpful assistant knowledgeable about power converters."})
    session["chat_history"].append({"role": "user", "content": user_text})

    # Generate response using entire chat history
    response = query_finetuned_llm(session["chat_history"])

    # Append model response to chat history
    session["chat_history"].append({"role": "assistant", "content": response})
    
    return jsonify({"response": response})

# Predefined mapping of converter names to modules and functions
CONVERTER_MAP = {
    "boost": (boost, "run_boost_gui"),
    "buck": (buck, "run_buck_gui"),
    "buckboost": (buckboost, "run_buckboost_gui"),
    "cuk": (cuk, "run_cuk_gui"),
    "flyback": (flyback, "run_flyback_gui"),
    "forward": (forward, "run_forward_gui"),
    "fullbridge": (fullbridge, "run_fullbridge_gui"),
    "halfbridge": (halfbridge, "run_halfbridge_gui"),
    "pushpull": (pushpull, "run_pushpull_gui"),
    "sepic": (sepic, "run_sepic_gui"),
    "zeta": (zeta, "run_zeta_gui"),
}
@app.route('/run_gui', methods=['POST'])
def run_gui():
    try:
        data = request.json
        converter_type = data.get("converter", "").strip().lower().replace(" ", "")  # Normalize input

        if not converter_type:
            return jsonify({"error": "No converter type specified."}), 400

        # Check if the converter is supported
        if converter_type not in CONVERTER_MAP:
            return jsonify({"error": f"Unknown or unsupported converter type: {converter_type}"}), 400

        # Retrieve the correct module and function
        module, function_name = CONVERTER_MAP[converter_type]
        results = getattr(module, function_name)()  # Call function

        # Store result in chat history
        if "chat_history" not in session:
            session["chat_history"] = []
        session["chat_history"].append({"role": "assistant", "content": f"GUI Results: {results}"})

        return jsonify({"response": f"{converter_type.capitalize()} Converter GUI Results: {results}"})

    except Exception as e:
        return jsonify({"error": f"Failed to run {converter_type} GUI: {str(e)}"})


def query_finetuned_llm(chat_history):
    # Pass entire chat history
    inputs = tokenizer.apply_chat_template(
        chat_history,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs, 
        max_new_tokens=2048, 
        do_sample=True
    )

    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):], 
        skip_special_tokens=True
    )

    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
