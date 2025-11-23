# sanity_check_vectors.py
from milvus_impl_gpt import VectorStore

print("\n=== SANITY CHECK: Embeddings Before Insertion ===\n")

# Load VectorStore but do NOT call _insert_data_if_needed()
# We override documents generation only
vs = VectorStore(auto_insert=False)
docs = vs.documents

print(f"Total chunks loaded: {len(docs)}")
if len(docs) == 0:
    print("❌ No documents or chunks found. Add .md files to the 'documents/' folder.")
    exit()

all_vectors = [d["vector"] for d in docs]

print("\nChecking vector list shape...")
print(f"- Example vector type: {type(all_vectors[0])}")
print(f"- Example vector length: {len(all_vectors[0])}")

# SUMMARY INFO
print("\n=== Vector Summary ===")
lengths = [len(v) if isinstance(v, list) else -1 for v in all_vectors]
unique_lengths = set(lengths)
print(f"- Unique lengths found: {unique_lengths}")

bad_length = [i for i, v in enumerate(all_vectors) if len(v) != vs.dim]
nested = [i for i, v in enumerate(all_vectors) if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list)]
non_numeric = [
    i for i, v in enumerate(all_vectors)
    if any(not isinstance(x, (float, int)) for x in v)
]

print(f"- Vectors with wrong length: {bad_length}")
print(f"- Suspected nested vectors ([[...]]) at indices: {nested}")
print(f"- Vectors with non-numeric elements: {non_numeric}")

# DETAILED PRINT FOR FIRST 3 VECTORS
print("\n=== SAMPLE VECTORS (first 3) ===")
for idx, v in list(enumerate(all_vectors[:3])):
    print(f"\nVector #{idx}")
    print(f"Type: {type(v)}")
    print(f"Length: {len(v)}")
    print(f"First 10 vals: {v[:10]}")
    if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
        print("⚠️  Nested vector detected: [[...]]")
    for element in v:
        if not isinstance(element, (float, int)):
            print(f"❌ Non-numeric element detected: {element} (type {type(element)})")

print("\n=== DONE ===\n")
