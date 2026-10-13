from transformers import AutoTokenizer

# The stress-test text snippet
text = """
English and CAPITALIZATION
丞
😊
show_tokens False None elif == else: two tabs: "  " Three tabs: "   "
12.0*50=600
"""

# A rotating list of RGB background colors for visualization
colors_list = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence, tokenizer_name):
    print(f"\n================ MODEL: {tokenizer_name} ================")
    # Load only the tokenizer (no model weights / no GPU needed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Tokenize the text sequence
    token_ids = tokenizer(sentence).input_ids
    
    # Print each token with a rotating background color
    for idx, t in enumerate(token_ids):
        color = colors_list[idx % len(colors_list)]
        # Decode individual token ID back to text and wrap in ANSI color codes
        token_text = tokenizer.decode([t])
        print(f'\x1b[0;30;48;2;{color}m{token_text}\x1b[0m', end='')
    
    print("\n" + "=" * 50)

# List of different model tokenizers to compare
models_to_test = [
    "microsoft/Phi-3-mini-4k-instruct",
    "gpt2",
    "bert-base-uncased"
]

# Run the test across each model
for model_id in models_to_test:
    try:
        show_tokens(text, model_id)
    except Exception as e:
        print(f"Could not load {model_id}: {e}")