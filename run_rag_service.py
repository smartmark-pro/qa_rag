import sys
import logging
import torch
import json
from starlette.middleware.cors import CORSMiddleware


from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware


from typing import List, Union
from pydantic import BaseModel


from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
app = FastAPI(lifespan=lifespan, openapi_url=None, title="模型服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200, content="{}")


# https://huggingface.co/BAAI/bge-m3
# https://huggingface.co/BAAI/bge-reranker-v2-m3

from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)

max_size = 512
batch_size = 12


class Base(BaseModel):
    username: str=""
    token: str=""
    requestid: str=""

class EmbeddingsRequest(Base):
    model_name: str=""
    sentences: list
    embeddings_id: str


from enum import Enum
class ComputeScoreType(str, Enum):
    sparse = "sparse"
    dense = "dense"
    colbert ="colbert"
    custom = "custom"


class EmbeddingScoreRequest(Base):
    model_name: str=""
    query: list
    embeddings_id: str
    method: ComputeScoreType
    weight: list=None
    topk: int=10
    min_score: float=0.0


class RerankRequest(Base):
    model_name: str=""
    sentences: list
    topk: int=10
    min_score: float=0.0


class RemoveRequest(BaseModel):
    embeedings_id: str

# 简单测试使用存到变量， 暂不使用mlivus之类的向量数据库
embeddings_list = {}
@app.post("/app/rag/v1/embeddings")
async def create_embeddings(request: EmbeddingsRequest):
    if request.embeddings_id not in embeddings_list:

        embeddings = model.encode(request.sentences, 
                                batch_size=batch_size, 
                                max_length=max_size, 
                                return_dense=True, 
                                return_sparse=True, 
                                return_colbert_vecs=True) 
        embeddings_list[request.embeddings_id]=embeddings     
    return Response(status_code=200, content=json.dumps({}))


@app.post("/app/rag/v1/embeddings/score")
async def embeddings_score(request: EmbeddingScoreRequest):
    embeddings: dict= embeddings_list.get(request.embeddings_id)
    if request.method==ComputeScoreType.dense:
        embeddings_1 = model.encode(request.query, 
                                batch_size=batch_size, 
                            max_length=max_size, 
                            return_dense=True, 
                                )['dense_vecs']
        similarity = embeddings_1 @ embeddings.get("dense_vecs").T
        scores = [ float(sim[0]) for sim in similarity]
    elif request.method==ComputeScoreType.sparse:
        embeddings_1 = model.encode(request.query, 
                                batch_size=batch_size, 
                            max_length=max_size, 
                            return_sparse=True, 
                                )['lexical_weights']
        scores = []
        for e in embeddings["lexical_weights"]:
            lexical_score = model.compute_lexical_matching_score(embeddings_1[0], e)
            scores.append(lexical_score)
    elif request.method==ComputeScoreType.colbert:
        embeddings_1 = model.encode(request.query, max_length=max_size, return_dense=True, return_sparse=True, return_colbert_vecs=True)["colbert_vecs"]
        scores = []
        for e in embeddings["colbert_vecs"]:
            score = model.colbert_score(embeddings_1[0], e)
            scores.append(float(score))
    elif request.method==ComputeScoreType.custom:
        # TODO 
        pass
        # sentence_pairs = [[i,j] for i in sentences_1 for j in sentences_2]
        # model.compute_score(sentence_pairs, 
        #                   max_passage_length=128, # a smaller max length leads to a lower latency
        #                   weights_for_different_modes=[0.4, 0.2, 0.4])
        return []
    else:
        return Response(status_code=400, content="参数错误， 方法不存在")
    # 整理格式
    new = []
    for i, score in enumerate(scores):
        if score>request.min_score:
            new.append((i, score))
    new.sort(key=lambda x: -x[1])
    # 仅调试打印
    # print(str(score)[:100], len(scores), len(new))
    return Response(status_code=200, content=json.dumps(new[:request.topk]))


@app.post("/app/rag/v1/rerank")
async def rerank(request: RerankRequest):
    scores = reranker.compute_score(request.sentences, normalize=True) 
    new = []
    for i, score in enumerate(scores):
        if score>request.min_score:
            new.append((i, score))
    new.sort(key=lambda x: -x[1])
    # 仅调试打印
    # print(str(score)[:100], len(scores), len(new))               
    return Response(status_code=200, content=json.dumps(new[:request.topk]))


@app.post("/app/rag/v1/embeddings/remove")
async def remove(request: RemoveRequest):
    if request.embeedings_id in request:
        embeddings_list.pop(request.embeedings_id)
        return Response(status_code=200, content=json.dumps({}))
    else:
        return Response(status_code=400, content=json.dumps({"id": request.embeedings_id}))


@app.get("/app/rag/v1/embeddings/status")
async def embeddings_status():
    # 实际使用需要更多的状态
    status = {k: len(v["sparse"]) for k, v in embeddings_list}
    return Response(status_code=200, content=json.dumps(status))

if __name__ == "__main__":
    import logging
    import uvicorn

    logger = logging.getLogger("server")
    logger.info("start ...")

    uvicorn.run(app="run_rag_service:app", host="0.0.0.0", port=9046, reload=False, workers=1)