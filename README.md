# Cattle Mounting Detection via Retrieval-Augmented Vision-Language Reasoning

Code and sample data for detecting cattle mounting behavior (Normal / Mounting) with a Retrieval-Augmented Generation (RAG) pipeline:

1. Retrieval — encode each video with VideoMAE (16 uniform frames) and find
   the most similar reference video by cosine similarity to retrieve its
   description.
2. Reasoning — feed the query video (16 uniform frames) and the retrieved
   description to Qwen3-VL-4B-thinking-FP8 (served with vLLM) to output reason and result

The `datasets/` folder contains only a small sample datas for running the demo, not the full dataset used in the paper.

# Structure

```
test_rag.py                               # inference pipeline
requirements.txt
datasets/
  train_video_reason_annotation.json      # reference DB annotations
  videos/                                  # reference videos      
  val/{normal,mounting}/                   # validation video clips    
result_rag.json, result_rag_normal.json   # output results
```

# Setup & run

Requires an NVIDIA GPU (CUDA 13, tested on RTX 4070 Ti SUPER 16 GB), Python 3.10+.


pip install -r requirements.txt
python test_rag.py
