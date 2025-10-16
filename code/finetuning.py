from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses, InputExample
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load Corupus in
corpus_dataset = load_dataset("CoIR-Retrieval/cosqa", "corpus", split="corpus")
print(f"Corpus loaded with {len(corpus_dataset)} documents.")

# Load the queries
queries_dataset = load_dataset("CoIR-Retrieval/cosqa", "queries", split="queries")
print(f"Queries loaded with {len(queries_dataset)} queries.")


# Load in test set.
train_dataset = load_dataset("CoIR-Retrieval/cosqa", name="default", split="test")
print(f"Train 'test' split loaded with {len(train_dataset)} query-document pairs.")

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

print("Building corpus and query lookup maps...")
corpus_map = {item["_id"]: item["text"] for item in corpus_dataset}
query_map = {item["_id"]: item["text"] for item in queries_dataset}
print(f"Example corpus entry:\nID: {list(corpus_map.keys())[0]}\nText: {list(corpus_map.values())[0][:120]}...\n")
print(f"Example query entry:\nID: {list(query_map.keys())[0]}\nText: {list(query_map.values())[0]}\n")

print("Building positive (query, code) pairs...")
train_pairs = []
for item in train_dataset:
    qid = item["query-id"]
    did = item["corpus-id"]
    
    query_text = query_map[qid]
    code_text = corpus_map[did]
    
    train_pairs.append((query_text, code_text))
print(f"Built {len(train_pairs)} training pairs.")
print("Example pair:")
print("Query:", train_pairs[0][0])
print("Code:", train_pairs[0][1][:120], "...\n")
corpus_ids = list(corpus_map.keys())

print("Creating random negative samples for triplets...")
train_triplets = []

for query_text, pos_code in train_pairs:
    while True:
        neg_id = random.choice(corpus_ids)
        neg_code = corpus_map[neg_id]
        if neg_code != pos_code:
            break
    train_triplets.append((query_text, pos_code, neg_code))

print(f"Created {len(train_triplets)} sample triplets (showing 1 example):")
print("Query:", train_triplets[0][0])
print("Positive:", train_triplets[0][1][:120], "...")
print("Negative:", train_triplets[0][2][:120], "...\n")

train_samples = [
    InputExample(texts=[query, pos, neg])
    for query, pos, neg in train_triplets
]

print(f"Converted {len(train_samples)} triplets into InputExamples.")
print("Example InputExample:")
print(train_samples[0])

# Training loop
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)
print(f"DataLoader ready with {len(train_samples)} samples and batch size = 8.")

train_loss = losses.TripletLoss(model=model)
print("TripletLoss defined.")

loss_values = []

def loss_callback(score, epoch, step):
    loss_values.append(score)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path="fine_tuned_model",
    callback = loss_callback()
)

import matplotlib.pyplot as plt

plt.plot(loss_values)
plt.title("Training Loss per Step")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

print("Fine-tuning complete! Model saved to 'fine_tuned_model'.")