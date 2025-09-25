# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List
import json
from scipy import stats
from tqdm import tqdm

def infer_stream(engine: 'InferEngine', infer_request: 'InferRequest'):
    request_config = RequestConfig(max_tokens=512, temperature=0, stream=True)
    metric = InferStats()
    gen = engine.infer([infer_request], request_config, metrics=[metric])
    query = infer_request.messages[0]['content']

    response_content = ""
    for resp_list in gen:
        if resp_list[0] is None:
            continue
        response_content += resp_list[0].choices[0].delta.content

    return response_content

#----------------------------------------------------------------------

if __name__ == '__main__':
    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
    from swift.plugin import InferStats

    model = r""
    dataset_path = r""
    infer_backend = 'pt'

    if infer_backend == 'pt':
        engine = PtEngine(model, max_batch_size=64)
    elif infer_backend == 'vllm':
        from swift.llm import VllmEngine
        engine = VllmEngine(model, max_model_len=32768)
    elif infer_backend == 'lmdeploy':
        from swift.llm import LmdeployEngine
        engine = LmdeployEngine(model)

    with open(dataset_path, 'r') as f:
        datas = json.load(f)
    
    # messages = [{'role': 'user', 'content': '<video>Please rate the quality of the human face in this video, considering factors such as resolution, clarity, smoothness, and overall visual quality.'}]
    for data in tqdm(datas):
        messages = [data["messages"][0]]
        videos = [data["videos"]]
        reply = infer_stream(engine, InferRequest(messages=messages, videos=videos))
        print(f"Query: {messages[0]['content']}")
        print(f"Response: {reply}")
