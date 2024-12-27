from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Route to calculate
@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    expression = data.get('expression')
    print(f"Received expression: {expression}")  # Debugging print
    try:
        result = eval(expression)
        print(f"Result: {result}")  # Debugging print
        return jsonify({"result": result}), 200
    except Exception as e:
        print(f"Error: {e}")  # Debugging print
        return jsonify({"error": "Invalid expression"}), 400


if __name__ == "__main__":
    app.run()
