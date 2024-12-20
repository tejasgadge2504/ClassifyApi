from flask import Flask, request, jsonify
from transformers import pipeline

# Create the Flask app
app = Flask(__name__)

# Load the zero-shot classification pipeline with explicit model name
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# Define the possible intents
intents = ["Check Balance", "Transfer Money", "Bill Payment", "Account Information"]

@app.route('/classify', methods=['POST'])
def classify_intent():
    try:
        # Get JSON input from the request
        data = request.get_json()
        user_input = data.get('user_input')

        if not user_input:
            return jsonify({"error": "user_input is required"}), 400

        # Classify the user's intent
        result = classifier(user_input, intents)

        # Return the classification result
        return jsonify({
            "input": user_input,
            "result": result
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Run Flask server without debug mode to avoid auto-reloads
    app.run(host='0.0.0.0', port=5000, debug=False)
