# Predict event time on temporal graphs using tgnn

A priminary idea.
combine harmonic encoding with temporal point proces, e.g., hawkes process.
devise tgnn aggregation scheme.
maximize  event sequence likelihood.


### Generate node2vec embeddings
```
cd tgnn/
python src/tge/software/node2vec/src/main.py --input ./data/CollegeMsg/CollegeMsg.edgelist --output ./data/CollegeMsg/CollegeMsg.emb
```

### Run with embeddings
```
python -m tge.main # default CollegeMsg dataset
```

## Ref


