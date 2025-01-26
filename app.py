from flask import Flask, render_template, request, redirect, url_for, jsonify
from Final_1 import HybridAgent, Config, QueryResult
import os

app = Flask(__name__)
config = Config()
agent = None
query_history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    global agent
    if request.method == 'POST':
        db_uri = request.form['db_uri']
        docs_path = request.form['docs_path']
        
        agent = HybridAgent(database_uri=db_uri, docs_directory=docs_path, config=config)
        return redirect(url_for('query'))

    return render_template('index.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    global query_history
    if request.method == 'POST':
        query_text = request.form['query']
        result = agent.process_query(query_text)
        query_history.append((query_text, result))
    
    return render_template('query.html', query_history=query_history)

@app.route('/process_query', methods=['POST'])
def process_query():
    global query_history
    query_text = request.json['query']
    result = agent.process_query(query_text)
    query_history.append((query_text, result))
    return jsonify({
        'query': query_text,
        'result': result.content,
        'source': result.source,
        'confidence': result.confidence
    })

if __name__ == '__main__':
    app.run(debug=True)