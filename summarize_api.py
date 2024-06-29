from threading import Thread
from flask import Flask, request, jsonify
from queue import Queue
import traceback
from sum_news import ModelGusevSum

# Initialize Flask app
app = Flask(__name__)

# Initialize a Queue to store incoming requests
request_queue = Queue()
model = ModelGusevSum()

# Worker function to process requests from the queue
def worker():
    with app.app_context():  # Ensure the Flask app context is active
        while True:
            text, response_queue = request_queue.get()
            try:
                # Process the request
                response = jsonify(model.summary_title(text[:1000], 1000))
            except Exception as e:
                response = jsonify({"error": str(e)})
                traceback.print_exc()
            # Put the response in the response queue
            response_queue.put(response)
            request_queue.task_done()


# Start the worker thread
Thread(target=worker, daemon=True).start()

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        text = request.args.get('text', '')  # Get the 'text' parameter from the request
    elif request.method == 'POST':
        data = request.json  
        text = data.get('text', '') 

    response_queue = Queue()
    request_queue.put((text, response_queue))
    response = response_queue.get()
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)