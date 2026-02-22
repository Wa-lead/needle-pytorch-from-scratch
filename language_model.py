import needle as ndl
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb, generate_ptb

device = ndl.cuda() if ndl.cuda().enabled() else ndl.cpu()
print(f"Using device: {device}")

corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=64, device=device, dtype="float32")

model = LanguageModel(
    embedding_size=20,
    output_size=len(corpus.dictionary),
    hidden_size=32,
    num_layers=1,
    seq_model='transformer',
    seq_len=20,
    num_head=4,
    dim_head=5,
    device=device,
)
print(f"Vocab size: {len(corpus.dictionary)}, Train data shape: {train_data.shape}")

print("Training...")
train_acc, train_loss = train_ptb(model, train_data, seq_len=20, n_epochs=10, device=device, lr=0.003, optimizer=ndl.optim.Adam)
print(f"Train - accuracy: {float(train_acc):.4f}, loss: {float(train_loss):.4f}")

print("\nEvaluating...")
eval_acc, eval_loss = evaluate_ptb(model, train_data, seq_len=20, device=device)
print(f"Eval  - accuracy: {float(eval_acc):.4f}, loss: {float(eval_loss):.4f}")

print("\n--- Generation ---")
for prompt in ["the company", "we have", "in the"]:
    text = generate_ptb(model, corpus, prompt, max_len=30, temperature=0.8, device=device)
    print(f"  '{prompt}' -> {text}")