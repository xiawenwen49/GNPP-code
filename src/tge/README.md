# Directly predict next event time on temporal graphs using tgnn

combining gnn framework with time-series inference framework
maximize event sequence likelihood.


<!-- ### Generate node2vec embeddings
```
cd tgnn/
python src/tge/software/node2vec/src/main.py --input ./data/CollegeMsg/CollegeMsg.edgelist --output ./data/CollegeMsg/CollegeMsg.emb
``` -->

<!-- ### Run with embeddings
```
python -m tge.main # default CollegeMsg dataset
``` -->





# Synthetic dataset generation
```
python -m tge.test
```

# Run tge model
```
python -m tge.main --epochs 100 --dataset Reddit
```

