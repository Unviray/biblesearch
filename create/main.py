
from fastapi import FastAPI

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from peewee import *
from tqdm import tqdm
from environs import Env
from pprint import pprint
import json
app = FastAPI()


env = Env()

env.read_env()

database = SqliteDatabase("nwt.sqlite")

openai_embedding = embedding_functions.OpenAIEmbeddingFunction(
    api_key=env.str("OPENAI_API_KEY"), model_name="text-embedding-ada-002"
)


class BaseModel(Model):
    class Meta:
        database = database


class Link(BaseModel):
    identifier = IntegerField(unique=True)
    to = CharField()

    class Meta:
        table_name = "link"


class Verset(BaseModel):
    embedding = CharField()
    content = CharField()
    content_en = CharField(null=True)
    pointer_id = IntegerField(unique=True)

    class Meta:
        table_name = "verset"


print("initializing client")
client = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet", persist_directory=".chromadb/")
)

print("get collection")
collection = client.get_or_create_collection(
    name="bible",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=env.str("OPENAI_API_KEY"), model_name="text-embedding-ada-002"
    ),
)


def run_add():
    print("load to memory")
    data = []
    for row in Verset.select():
        data.append(
            {
                "pointer_id": row.pointer_id,
                "content_en": row.content_en,
                "embedding": row.embedding
            }
        )

    print("add collection")
    collection.add(
        documents=[item["content_en"] for item in data],
        embeddings=[json.loads(item["embedding"]) for item in data],
        ids=[str(item["pointer_id"]) for item in data],
    )


def run():
    result = collection.query(
        query_texts=[
            "worshipping God by following traditions or customs that displease him."
        ],
        n_results=10,
    )

    for n, id in enumerate(result["ids"][0]):
        print(
            round(result["distances"][0][n], 2),
            id,
            Verset.get(Verset.pointer_id == id).content,
        )


if __name__ == "__main__":
    run()
