from flask import Flask, render_template, request, jsonify
from utils import load_model_and_tokenizer, compute_word_importance

# 初始化Flask应用
app = Flask(__name__)

# 加载模型和分词器
model, tokenizer = load_model_and_tokenizer()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    prompt = request.form['prompt']
    word_scores = compute_word_importance(prompt, model, tokenizer)

    result = []
    for generated_word, scores in word_scores:
        result.append({
            "generated_word": generated_word,
            "scores": [{"input_word": word, "score": score} for word, score in scores]
        })

    print(result)  # 添加打印语句
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

