# examples:
#   python3 rag_llama_index.py -fs "docs/arduino/uno-rev3/arduino-uno-rev3.pdf" -u="https://store.arduino.cc/products/arduino-uno-rev3" -c="arduino-uno-rev3"
#   python3 rag_llama_index.py -fs "docs/technosoft/technosoft_ipos_233_canopen/technosoft_ipos_233_canopen.pdf" "docs/technosoft/technosoft_ipos_233_canopen/imot23xs.pdf" -u="https://technosoftmotion.com/en/intelligent-motors/\?SingleProduct\=174" -c="technosoft_ipos_233_canopen"
#   python3 rag_llama_index.py -fs "docs/raspberry/pi4/raspberry-pi-4-product-brief.pdf" "docs/raspberry/pi4/raspberry-pi-4-datasheet.pdf" -u="https://www.raspberrypi.com/products/raspberry-pi-4-model-b/" -c="raspberry-pi-4-product-brief"
#   python3 rag_llama_index.py -fs "docs/ur/ur5e/ur5e_user_manual_en_us.pdf" "docs/ur/ur5e/ur5e-fact-sheet.pdf" -u="https://www.universal-robots.com/products/ur5-robot/" -c="ur5e_user_manual_en_us"

# tbd:
#   add additional sources
#   improve sources parser
#       -> better parsing of the web pages and PDF
#   DB integration
#      integrate postgresql or another database
#       create simple devices representation in database, e.g. device_type table, and populate it
#       use sql requests to get data from the tables
#       use sql requests to put data to he tables
#        metadata filtering
#   improve openai requsts / consider chains
#       -> e.g. if device=robot get this and this data
#       -> if another type, e.g. inductive sensor ask for another data
#   test different query modes https://gpt-index.readthedocs.io/en/v0.5.27/reference/query.html#gpt_index.indices.query.schema.QueryMode


import argparse

parser = argparse.ArgumentParser(
    prog="RagLlamaindex",
    description="Retrieve information from different soures - PDFs and Web-Links",
)

parser.add_argument("-fs", "--filenames", nargs="+", default=[])
parser.add_argument("-u", "--url")
parser.add_argument("-c", "--collection")
parser.add_argument("-k", "--similarity_top_k", default=3)
parser.add_argument("-d", "--debug", action="store_true")

args = parser.parse_args()

import logging
import sys

logger = logging.getLogger("DefaultLogger")
streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
streamHandler.setFormatter(formatter)
if args.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
logger.addHandler(streamHandler)

from llama_index import QueryBundle
from llama_index.retrievers import BaseRetriever
from typing import Any, List

from llama_index import ServiceContext

import os
import openai

from llama_index import VectorStoreIndex

from llama_index.vector_stores import ChromaVectorStore

from llama_index.output_parsers import LangchainOutputParser
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    DEFAULT_REFINE_PROMPT_TMPL,
)
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from llama_index.llms import OpenAI

from llama_index.embeddings import HuggingFaceEmbedding

from llama_index.vector_stores import VectorStoreQuery

from llama_index.schema import NodeWithScore
from typing import Optional

# load open ai key
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# load embedding model
# model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embed_model_name = "sentence-transformers/all-MiniLM-L12-v2"
# model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

logger.info(
    "--------------------- Loading embedded model {} \n".format(embed_model_name)
)
embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

# define llm and its params
llm_temperature = 0.1
llm_model = "gpt-3.5-turbo"
# llm_model = "gpt-3.5-turbo-instruct" # not good - responces are too unprecise
# llm_model = "gpt-4" # good responces but way too expencive
logger.info("--------------------- Loading llm model {} \n".format(llm_model))
llm = OpenAI(temperature=llm_temperature, model=llm_model)

from llama_index.text_splitter import SentenceSplitter

from llama_hub.file.pymu_pdf.base import PyMuPDFReader

import re
import json
from llmsherpa.readers import LayoutPDFReader


# chroma_client = chromadb.EphemeralClient()
# chroma_collection = chroma_client.create_collection("quickstart")


def load_documents(filenames, url):
    """load documents from different sources"""

    # load from url
    from llama_index import download_loader
    from langchain.document_loaders import WebBaseLoader
    from llama_index import Document

    ReadabilityWebPageReader = download_loader("ReadabilityWebPageReader")
    # or set proxy server for playwright: loader = ReadabilityWebPageReader(proxy="http://your-proxy-server:port")
    # For some specific web pages, you may need to set "wait_until" to "networkidle". loader = ReadabilityWebPageReader(wait_until="networkidle")
    loader_url = ReadabilityWebPageReader()
    documents = []
    if url:
        logger.info("--------------------- Load urls \n")
        loader_url_lang = WebBaseLoader(url)
        data = loader_url_lang.load()
        doc = loader_url.load_data(url=url)
        doc.append(
            Document(text=data[0].page_content)
        )  # add information from different url reader
        documents = documents + doc

    # load from PDFs
    loader_pdf = PyMuPDFReader()
    llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    pdf_reader = LayoutPDFReader(llmsherpa_api_url)
    for file in filenames:
        logger.info("--------------------- Load document {} \n".format(file))
        doc = loader_pdf.load(file_path=file)
        documents = documents + doc
        # logger.info("--------------------- PDF read with Sherpa \n")
        # doc_sherpa = pdf_reader.read_pdf(file)

    # remove fields having value None -> cause error
    for doc in documents:
        for key in doc.metadata:
            if doc.metadata[key] is None:
                doc.metadata[key] = 0

    return documents


def load_documents_to_db(filenames, url, vector_store):
    """load data to vector database collection"""

    documents = load_documents(filenames, url)

    text_splitter = SentenceSplitter(
        chunk_size=1024,
        separator=" ",
    )
    text_chunks = []

    # old with k=10 was not good for different devices
    sentences = []
    window_size = 128
    step_size = 20

    # new - gives much better results with k=20
    sentences = []
    window_size = 96
    step_size = 76

    # maintain relationship with source doc index, to help inject doc metadata in (3)
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        # cur_text_chunks = text_splitter.split_text(" ".join(doc.text.split()))#text_splitter.split_text(doc.text)
        text_tokens = doc.text.split()
        for i in range(0, len(text_tokens), step_size):
            window = text_tokens[i : i + window_size]
            if len(window) < window_size:
                break
            sentences.append(window)
        paragraphs = [" ".join(s) for s in sentences]
        for i, p in enumerate(paragraphs):
            pp = re.sub(r"\.\.\.\.+", " ", p)  # remove dots
            paragraphs[i] = re.sub(r"\. \. \. \. +", " ", pp)  # remove dots
        # text_chunks.extend(paragraphs)
        text_chunks = paragraphs
        doc_idxs.extend([doc_idx] * len(paragraphs))

    from llama_index.schema import TextNode

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    # add from sherpas pdf rearder
    from llama_index.readers.schema.base import Document

    # for chunk in doc_sherpa.chunks():
    #    nodes.append(Document(text=chunk.to_context_text(), extra_info={}))

    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    logger.info("--------------------- Add data to the vector store \n")
    vector_store.add(nodes)


# prepare query engine for the llm request


def make_llm_request(responce_schema, llm, index, query_str, k):
    retriever = index.as_retriever(similarity_top_k=k)
    retrieved_nodes = retriever.retrieve(query_str)

    def generate_response(retrieved_nodes, responce_schema, query_str, llm):
        context_str = "\n\n".join([r.get_content() for r in retrieved_nodes])
        qa_prompt = PromptTemplate(
            """\
            Context information is below.
            ---------------------
            {context_str}
            ---------------------
            Given the context information and not prior knowledge, answer the query.
            The output should be a Markdown code snippet formatted in the following
            schema, including the leading and trailing "```json" and "```":

            {responce_schema}

            Query: {query_str}
            Answer: \
            """
        )
        fmt_qa_prompt = qa_prompt.format(
            context_str=context_str,
            responce_schema=responce_schema,
            query_str=query_str,
        )
        response = llm.complete(fmt_qa_prompt)
        return str(response), fmt_qa_prompt

    response, fmt_qa_prompt = generate_response(
        retrieved_nodes, responce_schema, query_str, llm
    )
    response_dict = json.loads(re.sub(r"json", "", re.sub(r"```", "", response)))

    return response, response_dict


# create vector store and get collection
import chromadb
from llama_index.storage.storage_context import StorageContext

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection(args.collection)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if len(chroma_collection.get()["ids"]) == 0:
    logger.info("--------------------- Load data to collection  \n")
    load_documents_to_db(args.filenames, args.url, vector_store)
else:
    logger.info("--------------------- Data already exist in collection  \n")

service_context = ServiceContext.from_defaults(
    chunk_size=1024, llm=llm, embed_model=embed_model
)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    service_context=service_context,
    storage_context=storage_context,
)

device_types = [
    "Motor",
    "Motor Drive",
    "PLC CPU",
    "PLC IO Module System",
    "PLC IO Module",
    "Robot Arm",
    "Microcontroller Board",
    "Inductive Sensor",
    "Computer",
]

interface_types = [
    "Ethernet",
    "EtherCAT",
    "RS-232",
    "CAN",
    "Bluetooth",
    "LTE",
    "USB",
    "Wireless LAN / WLAN",
]

protocol_types = ["CANopen", "Profinet", "Modbus", "EtherNet/IP", "Profibus", "IO-Link"]

motor_types = ["Stepper motor", "DC motor", "Brushless DC motor / BLDC", "Servomotor"]

serial_connection_types = ["I2C / IIC", "1-Wire", "SPI", "UART", "RS-232"]

from llama_index.prompts import PromptTemplate

################################################# ask product details ################################################

responce_schema = """    
    ```json
    {{
        "document_description": string  // What is this technical document about?
        "company_name": string  // What is company name?
        "product_name": string // What is the product name?
        "product_description" : string // What is this product about?
    }}
    ```
    """


query_str = "What is this technical document/manual/specification about? What is company name? What is the product name?"
response_device, response_device_dict = make_llm_request(
    responce_schema, llm, index, query_str, int(args.similarity_top_k)
)
print(response_device)


#####################################################################   #################################################

responce_schema = """    
```json
    {{
        "interfaces": list  // What interfaces is this device {device} supporting? Select zero, one or multiple from the list {interface_types}.
        "specific_information_interfaces": string  // "What specific about interfaces that this device supports?"
    }}
    ```
    """.format(
    device=response_device_dict, interface_types=interface_types
)


query_str = "What interfaces is this product {} supporting? Select zero, one or multiple from the list {}.".format(
    response_device_dict, interface_types
)
response_interface, response_interfaces_dict = make_llm_request(
    responce_schema, llm, index, query_str, int(args.similarity_top_k)
)
print(response_interface)


######################################################################################################################

responce_schema = """    
```json
    {{
        "protocols": list  // What interfaces is this device {device} supporting? Select zero, one or multiple from the list {protocol_types}.
        "specific_information_protocols": string  // "What specific about protocols that this device supports?"
    }}
    ```
    """.format(
    device=response_device_dict, protocol_types=protocol_types
)


query_str = "What protocols is this product {} supporting? Select zero, one or multiple from the list {}.".format(
    response_device_dict, protocol_types
)
response_protocols, response_protocols_dict = make_llm_request(
    responce_schema, llm, index, query_str, int(args.similarity_top_k)
)
print(response_protocols)

######################################################################################################################

responce_schema = """    
```json
    {{
        "operating_voltage_min": int  // "What is the recommended minimum operating supply voltage in [V] for the device {device}?".
        "operating_voltage_max": int  // "What is the recommended minimum operating supply voltage in [V] for the device {device}?".
    }}
    ```
    """.format(
    device=response_device_dict
)


query_str = "What are the minimum and maximum operating supply voltage for this device {}?".format(
    response_device_dict
)

response_protocols, response_protocols_dict = make_llm_request(
    responce_schema, llm, index, query_str, int(args.similarity_top_k)
)
print(response_protocols)

### operating_voltage_min = ResponseSchema(
###     name="operating_voltage_min",
###     description="What is the recommended operating supply voltage minimum?",
### )
###
### operating_voltage_max = ResponseSchema(
###     name="operating_voltage_max",
###     description="What is the recommended operating supply voltage maximum?",
### )

### ################################################# ask device type ################################################
###
### # define output schema
### device_type = ResponseSchema(
###     name="device_type",
###     description="What is the device type from the list {} on the following device description {}?".format(
###         device_types, response_device.response
###     ),
### )
###
### response_schemas = [device_type]
### query_engine = get_query_engine(response_schemas)
### query_str = "What is the device type from the list {} based on the following device description {}?".format(
###     device_types, response_device.response
### )
### response_device_type, response_device_type_dict = make_llm_request(
###     query_engine, query_str
### )
###
### ################################################# ask interfaces ################################################
###
### # define output schema
### interfaces = ResponseSchema(
###     name="interfaces",
###     description="What interfaces is this product {} supporting?".format(
###         response_device.response
###     ),
###     type="list",
### )
### interfaces_choices = ResponseSchema(
###     name="interfaces_choices",
###     description="Select zero, one or multiple only and only from this list {}".format(
###         interface_types
###     ),
###     type="list",
### )
### specific_information_interfaces = ResponseSchema(
###     name="specific_information_interfaces",
###     description="What specific about interfaces that this product supports {}?".format(
###         response_device.response
###     ),
### )
###
### response_schemas = [interfaces, interfaces_choices, specific_information_interfaces]
###
### query_engine = get_query_engine(response_schemas)
### query_str = "What interfaces is this product {} supporting? Select zero, one or multiple from the list {}.".format(
###     response_device.response, interface_types
### )
### response_interfaces, response_interfaces_dict = make_llm_request(
###     query_engine, query_str
### )
###
### ################################################# ask protocols ################################################
###
### protocols = ResponseSchema(
###     name="protocols",
###     description="What communication protocols is this product {} supporting? ".format(
###         response_device.response
###     ),
###     type="list",
### )
### specific_information_protocols = ResponseSchema(
###     name="specific_information_protocols",
###     description="What specific about communication protocols that this product supports {}?".format(
###         response_device.response
###     ),
### )
###
### protocols_choices = ResponseSchema(
###     name="protocols_choices",
###     description="Select zero, one or multiple only and only from this list {}".format(
###         protocol_types
###     ),
###     type="list",
### )
###
### response_schemas = [protocols, protocols_choices, specific_information_protocols]
###
### query_engine = get_query_engine(response_schemas)
### query_str = "What protocols is this product {} supporting? Select zero, one or multiple from the list {}.".format(
###     response_device.response, protocol_types
### )
### response_protocol, response_protocol_dict = make_llm_request(query_engine, query_str)
###
### ################################################# ask serial protocols ################################################
###
### serial_communication = ResponseSchema(
###     name="serial_connection",
###     description="What serial protocols is this product {} supporting?".format(
###         response_device.response
###     ),
###     type="list",
### )
### specific_information_serial_communication = ResponseSchema(
###     name="specific_information_protocols",
###     description="What specific about serial protocols that this product supports {}?".format(
###         response_device.response
###     ),
### )
###
### serial_communication_choices = ResponseSchema(
###     name="serial_communication_choices",
###     description="Select zero, one or multiple only and only from this list {}".format(
###         serial_connection_types
###     ),
###     type="list",
### )
###
### response_schemas = [
###     serial_communication,
###     serial_communication_choices,
###     specific_information_serial_communication,
### ]
###
### query_engine = get_query_engine(response_schemas)
###
### query_str = "What serial protocols is this product {} supporting? ".format(
###     response_device.response
### )
###
### response_serial_communication, response_serial_communication_dict = (
###     response_protocol,
###     response_protocol_dict,
### ) = make_llm_request(query_engine, query_str)
###
### ################################################# ask operating voltage ################################################
###
### # define output schema
### operating_voltage_min = ResponseSchema(
###     name="operating_voltage_min",
###     description="What is the recommended operating supply voltage minimum?",
### )
###
### operating_voltage_max = ResponseSchema(
###     name="operating_voltage_max",
###     description="What is the recommended operating supply voltage maximum?",
### )
###
### response_schemas = [operating_voltage_min, operating_voltage_max]
###
### query_engine = get_query_engine(response_schemas)
###
### query_str = "What are the minimum and maximum operating supply voltage for this device {}?".format(
###     response_device.response
### )
###
### response_voltage, response_voltage_dict = (
###     response_protocol,
###     response_protocol_dict,
### ) = make_llm_request(query_engine, query_str)
###
### ################################################# ask robot specs ################################################
###
###
### def ask_robot_specs():
###     payload = ResponseSchema(
###         name="payload", description="What is the robots maximum payload?"
###     )
###
###     reach = ResponseSchema(
###         name="reach", description="What is reach of the robots end-effector?"
###     )
###
###     workspace_coverage = ResponseSchema(
###         name="reach", description="What is the robots workspace_coverage?"
###     )
###
###     response_schemas = [payload, reach, workspace_coverage]
###
###     query_engine = get_query_engine(response_schemas)
###
###     query_str = "What are the specifications as payload, reach and workspace coverage for the device {} with the description?".format(
###         response_device_type_dict["device_type"], response_device.response
###     )
###
###     response_voltage, response_voltage_dict = make_llm_request(query_engine, query_str)
###
###
### if response_device_type_dict["device_type"] == "Robot Arm":
###     ask_robot_specs()
###
