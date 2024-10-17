from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
import os 
#ragas imports
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_relevancy
)

from datasets import DatasetDict, Dataset
import pandas as pd

load_dotenv()
os.environ["OPENAI_API_KEY"] = " "
os.environ["RAGAS_DO_NOT_TRACK"] = os.getenv("RAGAS_DO_NOT_TRACK")

def add_to_eval_dataset(df, question, context, answer):
    new_row = [question, [str(context)], answer]
    df.loc[len(df)] = new_row
    return df 


def dataframe_to_dict(df):
    dataset = Dataset.from_pandas(df)
    dataset_dict = DatasetDict({"eval":dataset})
    return dataset_dict 

def dict_to_dataframe(dict):
    dataframe = pd.DataFrame.from_dict(dict)
    return dataframe

def ragas_eval(data):
    print(os.environ["OPENAI_API_KEY"])
    result = evaluate(
    data["eval"],
    metrics=[
        faithfulness,
        answer_relevancy,
        context_relevancy
    ],
    llm = ChatOpenAI(model="gpt-4o"),
    )
    df = result.to_pandas()
    return (result, df.head())
