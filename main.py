from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import get_peft_model, LoraConfig

# Load base model and tokenizer
model_name = "facebook/bart-large-cnn"  # BART model for summarization
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Configure PEFT (Parameter Efficient Fine-Tuning) using LoRA
peft_config = LoraConfig(
    r=8,  # Rank for LoRA
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.1,
    bias="none",
)

# Apply PEFT to the model
model = get_peft_model(model, peft_config)

# Create a summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

def summarize_text(text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    # Example text
    text = """
    Artificial Intelligence (AI) is transforming the world as we know it. From autonomous driving to language translation, AI models are becoming increasingly sophisticated and capable. 
    The rise of deep learning and neural networks has made it possible to solve complex problems with greater accuracy and speed.
    """
    
    summary = summarize_text(text)
    print("\nOriginal Text:\n", text)
    print("\nSummary:\n", summary)
