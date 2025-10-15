from flask import Flask, render_template, request, jsonify
from model import alnime_reply
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    msg = data.get('message', '')
    reply = alnime_reply(msg)
    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)
