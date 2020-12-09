from flask import Flask, jsonify, request
import requests
from train import train
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		company = request.get_json()
		# print(company)
		pred = train(company)
		# result = pred[0]
		# print(pred)
		return jsonify({'prediction':str(pred)})
	return 'OK'

if __name__ == "__main__":
	app.run()