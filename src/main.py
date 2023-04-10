from peewee import Model, IntegerField, CharField, SqliteDatabase
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from environs import Env


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mg-en")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mg-en")

env = Env()

env.read_env()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet", persist_directory=".chromadb/")
)

collection = client.get_or_create_collection(
    name="bible",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=env.str("OPENAI_API_KEY"), model_name="text-embedding-ada-002"
    ),
)


bookMap = [
    "Genesisy",
    "Eksodosy",
    "Levitikosy",
    "Nomery",
    "Deoteronomia",
    "Josoa",
    "Mpitsara",
    "Rota",
    "1 Samoela",
    "2 Samoela",
    "1 Mpanjaka",
    "2 Mpanjaka",
    "1 Tantara",
    "2 Tantara",
    "Ezra",
    "Nehemia",
    "Estera",
    "Joba",
    "Salamo",
    "Ohabolana",
    "Mpitoriteny",
    "Tononkiran’i Solomona",
    "Isaia",
    "Jeremia",
    "Fitomaniana",
    "Ezekiela",
    "Daniela",
    "Hosea",
    "Joela",
    "Amosa",
    "Obadia",
    "Jona",
    "Mika",
    "Nahoma",
    "Habakoka",
    "Zefania",
    "Hagay",
    "Zakaria",
    "Malakia",
    "Matio",
    "Marka",
    "Lioka",
    "Jaona",
    "Asan’ny Apostoly",
    "Romanina",
    "1 Korintianina",
    "2 Korintianina",
    "Galatianina",
    "Efesianina",
    "Filipianina",
    "Kolosianina",
    "1 Tesalonianina",
    "2 Tesalonianina",
    "1 Timoty",
    "2 Timoty",
    "Titosy",
    "Filemona",
    "Hebreo",
    "Jakoba",
    "1 Petera",
    "2 Petera",
    "1 Jaona",
    "2 Jaona",
    "3 Jaona",
    "Joda",
    "Apokalypsy",
]


class BaseModel(Model):
    class Meta:
        database = SqliteDatabase("nwt.sqlite")


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


def convert_pointer(num):
    divisors = [1000000, 1000, 1]
    result = []
    for divisor in divisors:
        quotient, num = divmod(num, divisor)
        result.append(quotient)
    return result


@app.get("/search")
def search(q: str):

    input_ids = tokenizer(q, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=1)

    q = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    query_result = collection.query(
        query_texts=[q],
        n_results=10,
    )

    results = []
    for n, id in enumerate(query_result["ids"][0]):
        results.append(
            {
                "distance": round(query_result["distances"][0][n], 2),
                "pointer": int(id),
                "content": Verset.get(Verset.pointer_id == id).content,
            }
        )

    return results


@app.get("/pointer")
def pointer(q: int):
    book_num, chapter, verset = convert_pointer(q)
    return f"{bookMap[book_num-1]} {chapter}:{verset}"
