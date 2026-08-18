# Cattle Mounting Detection via Retrieval-Augmented Vision-Language Reasoning

Code and sample data for detecting cattle mounting behavior (Normal / Mounting) with a Retrieval-Augmented Generation (RAG) pipeline:

1. Retrieval — encode each video with VideoMAE (16 uniform frames) and find
   the most similar reference video by cosine similarity to retrieve its
   description.
2. Reasoning — feed the query video (16 uniform frames) and the retrieved
   description to Qwen3-VL-4B-thinking-FP8 (served with vLLM) to output reason and result

The `datasets/` folder contains only a small sample datas for running the demo, not the full dataset used in the hub.

# Structure

```
├── datasets/
│   ├── train_video_reason_annotation.json  # reference DB annotations
│   ├── videos/                             # reference videos (332개)
│   └── val/                                # validation video clips
│       ├── mounting/                       
│       └── normal/                         
├── weights/
│   └── Qwen3-VL-4B-Thinking-FP8/          # VLM weight
├── test_rag.py                            # RAG inference pipeline
├── test.py                                # VLM Only inference
├── test_text_only.py                      
├── result.json                            # VLM Only mounting
├── result_rag.json                        # RAG VLM mounting
├── result_normal.json                     # VLM Only normal
├── result_rag_normal.json                 # RAG + VLM normal
└── requirements.txt                       

```

# Technical Implementation Details

1. VideoMAE-based video embedding

```
def get_video_embedding(video_path):
    """VideoMAE를 사용해 영상 특징 추출"""
    # 비디오 로드 및 16프레임 샘플링
    rgb, audio, info = read_video(video_path, pts_unit='sec')
    
    # 균등 간격으로 16개 프레임 선택
    indices = torch.linspace(0, rgb.size(0) - 1, 16).long()
    video_frames = rgb[indices]
    
    # VideoMAE로 특징 추출
    with torch.no_grad():
        outputs = retriever_model(**inputs.to(device))
        video_emb = outputs.last_hidden_state.mean(dim=1)
    
    # L2 정규화
    video_emb = video_emb / video_emb.norm(p=2, dim=-1, keepdim=True)
    return video_emb
```

2. RAG-based similar video search

```
def rag_inference(new_video_path, database_matrix):
    """가장 유사한 영상 검색"""
    query_emb = get_video_embedding(new_video_path)
    similarity = torch.mm(query_emb, database_matrix.t())
    scores, indices = torch.topk(similarity, k=1)
    return indices[0].item()
```

3. Context-Augmented 프롬프트 구성
```
# RAG 적용 시 프롬프트
user_msg = (
    f"Context: I found a similar historical video which was described as follows: "
    f"\"{retrieved_context}\"\n\n"
    "Task: Based on the visual content of the provided video and the context above, "
    "analyze whether the cow's behavior is Normal or mounting. "
    "Provide a reasoning and conclude with 'Status: Normal' or 'Status: Mounting'."
)
```

# Setup & run

Model: Qwen3-VL-4B-Thinking-FP8
Inference Settings: temperature=0.2, max_tokens=5120, repetition_penalty=1.1
Video Processing: 3.0 FPS sampling

NVIDIA GPU (CUDA 13, tested on RTX 4070 Ti SUPER 16 GB), Python 3.10+.

pip install torch torchvision transformers vllm qwen-vl-utils

python test_rag.py     # run RAG pipeline

python test.py         # run VLM only


# References
- Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
- VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

