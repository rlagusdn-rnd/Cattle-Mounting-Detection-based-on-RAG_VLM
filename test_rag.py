import torch
import numpy as np
import os
import json
from torchvision.io.video import read_video
from transformers import AutoProcessor, VideoMAEModel
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

device = "cuda" if torch.cuda.is_available() else "cpu"

# VideoMAE (Retriever)
retriever_model_id = "MCG-NJU/videomae-base" 
feature_extractor = AutoProcessor.from_pretrained(retriever_model_id)
retriever_model = VideoMAEModel.from_pretrained(retriever_model_id).to(device)

def get_video_embedding(video_path):
    """Feature extraction using VideoMAE"""
    try:
        rgb, audio, info = read_video(video_path, pts_unit='sec')
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return torch.zeros(1, 768).cpu()

    # frame sampling
    if rgb.size(0) > 16:
        indices = torch.linspace(0, rgb.size(0) - 1, 16).long()
        video_frames = rgb[indices]
    else:
        indices = torch.linspace(0, rgb.size(0) - 1, 16).long()
        video_frames = rgb[indices]

    video_frames_numpy = list(video_frames.numpy())
    inputs = feature_extractor(video_frames_numpy, return_tensors="pt")
    
    with torch.no_grad():
        outputs = retriever_model(**inputs.to(device))
        video_emb = outputs.last_hidden_state.mean(dim=1)
        
    video_emb = video_emb / video_emb.norm(p=2, dim=-1, keepdim=True)
    return video_emb.cpu()

def rag_inference(new_video_path, database_matrix):
    """Return the index of the most similar video"""
    query_emb = get_video_embedding(new_video_path)
    similarity = torch.mm(query_emb, database_matrix.t())
    scores, indices = torch.topk(similarity, k=1)
    return indices[0].item()


# vLLM
def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        "mm_processor_kwargs": video_kwargs,
    }

if __name__ == '__main__':
    # --- [Database Indexing (VideoMAE)] ---
    print("Indexing Database...")

    video_path = os.path.join(os.getcwd(), "datasets", "videos")
    video_path_list = os.listdir(video_path)
    # JSON load
    text_json = json.load(open("./datasets/train_video_reason_annotation.json"))

    # mapping
    video_to_conversations = {}
    for item in text_json:
        json_video_name = os.path.basename(item["video"])
        video_to_conversations[json_video_name] = item["conversations"]

    dataset_embeddings = []
    dataset_embeddings_video_name = []
    dataset_context_map = {} # index -> text mapping

    for i, video_name in enumerate(video_path_list):
        full_path = os.path.join(video_path, video_name)
        emb = get_video_embedding(full_path)
        dataset_embeddings.append(emb)
        dataset_embeddings_video_name.append(video_name)
        
        if video_name in video_to_conversations:
            convs = video_to_conversations[video_name]
            context_text = " ".join([turn['value'] for turn in convs if turn['from'] == 'gpt'])
            dataset_context_map[i] = context_text
        else:
            dataset_context_map[i] = "No description available."

    dataset_embeddings = torch.cat(dataset_embeddings)
    dataset_embeddings = dataset_embeddings / dataset_embeddings.norm(p=2, dim=-1, keepdim=True)

    # --- [vLLM (Qwen-VL) setting] ---
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    
    # model load
    checkpoint_path = os.path.join(os.getcwd(), "weights", "Qwen3-VL-4B-Thinking-FP8")
    # Processor load
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.80,
        max_model_len=12288,
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
    )

    tokenizer = processor.tokenizer

    stop_token_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "additional_special_tokens_ids"):
        stop_token_ids.extend(tokenizer.additional_special_tokens_ids)

    sampling_params = SamplingParams(
            temperature=0.2,          # more than 0
            repetition_penalty=1.1,   
            max_tokens=5120,
            top_k=20,                 
            stop_token_ids=stop_token_ids, 
        )

    # --- [4. RAG + vLLM Inference Pipeline] ---
    print("Start Inference...")
    # new_video_path = os.path.join(os.getcwd(), "datasets", "val", "mounting")
    new_video_path = os.path.join(os.getcwd(), "datasets", "val", "normal")

    video_list = os.listdir(new_video_path)

    FP = 0

    for video_name in video_list:
        target_full_path = os.path.join(new_video_path, video_name)
        
        # Find similar past videos
        idx = rag_inference(target_full_path, dataset_embeddings)
        
        # Context Extraction
        retrieved_video_name = dataset_embeddings_video_name[idx]
        retrieved_context = dataset_context_map[idx]
        
        # Configuring the prompt 
        user_msg = (
            f"Context: I found a similar historical video which was described as follows: \"{retrieved_context}\"\n\n"
            "Task: Based on the visual content of the provided video and the context above, "
            "analyze whether the cow's behavior is Normal or mounting. "
            "Provide a reasoning and conclude with 'Status: Normal' or 'Status: Mounting'."
        )
                
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant capable of visual reasoning. You must first output your thought process between <think> and </think> tags, and then provide the final answer."
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "video", 
                        "video": target_full_path,
                        # "max_pixels": 360 * 420,
                        # "fps": 3.0,
                        "nframes": 16, 
                    },
                    {"type": "text", "text": user_msg},
                ]
            }
        ]
        
        # vLLM Inference
        try:
            inputs = prepare_inputs_for_vllm(messages, processor)
            outputs = llm.generate([inputs], sampling_params=sampling_params)
            generated_text = outputs[0].outputs[0].text
            print(generated_text)
            
            # Extract only the actual response following </think>
            if "</think>" in generated_text:
                final_answer = generated_text.split("</think>")[-1].strip()
            else:
                final_answer = generated_text
            
            print(f"Video: {video_name}")
            print(f"Ref Video: {retrieved_video_name}")
            print(f"Output: {final_answer}\n" + "-"*30)


            # save the results
            result_dict = {
                "video": video_name,
                "retrieved_video": retrieved_video_name,
                "output": final_answer
            }
            
            result_json_path = "./result_rag_normal.json"
            # result_json_path = "./result_rag.json"

            if os.path.exists(result_json_path):
                with open(result_json_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            else:
                results = []
            
            results.append(result_dict)
            with open(result_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            if "Normal" in final_answer or "status: normal" in final_answer.lower():
                pass
                FP += 1
            else:
                pass
                # FP += 1
                
        except Exception as e:
            print(f"Inference Error on {video_name}: {e}")
            break


    print(f"Total FP Count: {FP}")
    
    # 정상 종료
    del llm
