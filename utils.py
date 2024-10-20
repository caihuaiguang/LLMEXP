# Step 1: 安装 transformers 库和 PyTorch
# Step 2: 导入所需的库
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print(torch.cuda.is_available())
# Step 3: 定义加载模型和分词器的函数
def load_model_and_tokenizer():
    model_name = "gpt2-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # 设置填充标记为结束标记
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    model.eval()  # 评估模式
    model.float()  # 使用全精度计算（float32）
    return model, tokenizer

# Step 4: 定义 hook 函数以捕获激活值
activation = None  # 全局变量，用于存储激活值

def hook_fn(module, input, output):
    global activation
    activation = output
    activation.requires_grad_()  # 确保 requires_grad 为 True
    activation.retain_grad()  # 确保梯度被保留

# 注册 hook 到最后一个 transformer 块中的 LayerNorm
def register_hooks(model):
    transformer_blocks = model.transformer.h  # 获取所有 transformer 块
    norm_layer = transformer_blocks[-2].ln_2  # 最后一个 transformer 块中的第二个 LayerNorm 层
    norm_layer.register_forward_hook(hook_fn)  # 注册 hook

# Step 5: 计算词的重要性得分
def compute_word_importance(prompt, model, tokenizer, target_word):
    # 注册 hook
    register_hooks(model)

    # 将 input_ids 转换为嵌入
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
    input_ids = inputs['input_ids']
    embeddings = model.get_input_embeddings()(input_ids)  # 获取嵌入
    embeddings.requires_grad_()  # 设置嵌入张量的 require_grad

    # Define CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss()

    # Get the target token id
    target_token_id = tokenizer.encode(target_word, add_special_tokens=False)[0]

    # Create target tensor
    targets = torch.tensor([target_token_id]).to("cuda")

    # Get gradients for the LayerNorm activation
    outputs = model(inputs_embeds=embeddings)  # 使用嵌入进行前向传播，触发 hook
    logits = outputs.logits
    next_token_logits = logits[0, -1, :]

    loss = criterion(next_token_logits.unsqueeze(0), targets)
    model.zero_grad()
    loss.backward()

    # 获取 LayerNorm 激活的梯度
    norm_layer_grads = activation.grad  # 来自 hook 函数捕获的梯度
    word_importance_scores = []

    # 计算每个词的梯度得分
    for i, word_id in enumerate(input_ids[0].tolist()):
        # word_embedding = model.get_input_embeddings()(torch.tensor([word_id]).to("cuda")).squeeze(0)
        # score = torch.einsum("d,d->", norm_layer_grads[0, i], activation[0, i])
        score = torch.dot(activation[0, i], norm_layer_grads[0, i])
        word_importance_scores.append((tokenizer.decode(word_id), score.item()))

    # 根据输入的目标词找到相关的输入
    # target_word_scores = [(word, abs(score)) for word, score in word_importance_scores]
    target_word_scores = [(word, score) for word, score in word_importance_scores]
    target_word_scores.sort(key=lambda x: x[1], reverse=True)

    return target_word_scores[:3]  # 返回最相关的三个词

# Step 6: 生成文本并计算词的重要性
def generate_and_analyze(model, tokenizer, prompt, max_length=50):
    generated_text = prompt
    input_ids = tokenizer(prompt, return_tensors='pt').to("cuda:0")['input_ids']

    for _ in range(max_length):
        outputs = model.generate(input_ids, max_length=len(input_ids[0]) + 1, pad_token_id=tokenizer.pad_token_id)

        # 获取生成的词
        new_token_id = outputs[0, -1].item()
        new_word = tokenizer.decode([new_token_id])


        # 计算新生成的词的重要性得分
        related_words = compute_word_importance(generated_text, model, tokenizer, new_word)
        print(f"Generated: {new_word}")
        print(f"Top 3 related words to '{new_word}':", related_words)

        # 如果生成的词是结束标记，则停止
        if new_token_id == tokenizer.eos_token_id:
            break

        generated_text += new_word
        # 更新输入
        input_ids = tokenizer(generated_text, return_tensors='pt').to("cuda")['input_ids']
    print("\nFinal generated text:", generated_text)

# Step 7: 主函数测试
def main():
    model, tokenizer = load_model_and_tokenizer()
    # Define the long text and the question
    long_text = """
    The Eiffel Tower, located in Paris, France, is a wrought-iron lattice tower that was completed in 1889. It was named after the engineer Gustave Eiffel, whose company designed and built the structure. The tower stands approximately 330 meters tall and was the tallest man-made structure in the world until the completion of the Empire State Building in 1931. The Eiffel Tower has become a global cultural icon of France and one of the most recognizable structures in the world.

    In 1969, the first human walked on the Moon as part of the Apollo 11 mission, a significant achievement in space exploration. The mission was led by astronauts Neil Armstrong, Edwin "Buzz" Aldrin, and Michael Collins. Armstrong's famous words upon landing were, "That's one small step for man, one giant leap for mankind."

    In 2000, the first human genome was sequenced, marking a milestone in genetic research. This project provided scientists with a detailed blueprint of human DNA, which has had profound implications for medicine and genetics.
    """
    question = "Who is the most famous person in the history of the Moon landing?"

    # Combine the text and question
    input_text = long_text + "\n\n" + question

    generate_and_analyze(model, tokenizer, input_text)

# 执行主函数
if __name__ == "__main__":
    main()
