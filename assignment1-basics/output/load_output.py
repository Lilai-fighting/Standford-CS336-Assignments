import pickle

with open("output/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("output/merges.pkl", "rb") as t:
    merges = pickle.load(t)

print(vocab)
print(merges)