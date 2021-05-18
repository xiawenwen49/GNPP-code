# Introduction

This is the code for *Graph neural point process for temporal interaction prediction*, i.e., the GNPP model.

## Notes
- Experimental results (partial) are in ./src/tge/notebooks/results.ipynb

- Other results (e.g., learned lambda scores, attention heatmap) are in http://10.4.1.95:8888/lab/tree/TGNN/src/tge/notebooks/exp_study.ipynb .

<!-- ### Generate node2vec embeddings
```
cd tgnn/
python src/tge/software/node2vec/src/main.py --input ./data/CollegeMsg/CollegeMsg.edgelist --output ./data/CollegeMsg/CollegeMsg.emb
``` -->

<!-- ### Run with embeddings
```
python -m tge.main # default CollegeMsg dataset
``` -->

# Run instruction
## Synthetic dataset generation
```
python -m tge.test
```

## Run the GNPP model
```
./run.sh
```

