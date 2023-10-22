import argparse

from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from text_generation import Client

PREPROMPT = "Below are a series of dialogues between various people and an AI assistant. The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable. The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed. It also tries to avoid giving false or misleading information, and it caveats when it isn't entirely sure about the right answer. That said, the assistant is practical and really does its best, and doesn't let caution get too much in the way of being useful.\n"
PROMPT = """"Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to
make up an answer. Don't make up new terms which are not available in the context.

{context}"""

END_7B = "\n<|prompter|>{query}<|endoftext|><|assistant|>"
#END_40B = "\nUser: {query}\nFalcon:"

PARAMETERS = {
    "temperature": 0.9,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "top_k": 50,
    "truncate": 1000,
    "max_new_tokens": 1024,
    "seed": 42,
    "stop_sequences": ["<|endoftext|>", "</s>"],
}
CLIENT_7B = Client("http://127.0.0.1:3000")  # Fill this part
#CLIENT_40B = Client("https://")  # Fill this part

import logging
import sys

def get_logger(debug):
    logger = logging.getLogger("DefaultLogger")
    streamHandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    streamHandler.setFormatter(formatter)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.addHandler(streamHandler)
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ws", "--window-size", type=int, default=128)
    parser.add_argument("-s", "--step-size", type=int, default=100)
    parser.add_argument("-f", "--fname", type=list, default=[])
    parser.add_argument("-u", "--url", type=list, default=[])
    parser.add_argument("-c", "--collection", type=str, default=[], required=True)
    parser.add_argument("-k", "--top-k", type=int, default=32)
    parser.add_argument("-d", "--debug", action="store_true")
    return parser.parse_args()


def load_documents(filenames, url, logger):
    """load documents from different sources"""

    # load from url
    from llama_index import download_loader
    from langchain.document_loaders import WebBaseLoader
    from llama_index import Document
    from llama_hub.file.pymu_pdf.base import PyMuPDFReader

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

    for file in filenames:
        logger.info("--------------------- Load document {} \n".format(file))
        doc = loader_pdf.load(file_path=file)
        documents = documents + doc

    # remove fields having value None -> cause error
    for doc in documents:
        for key in doc.metadata:
            if doc.metadata[key] is None:
                doc.metadata[key] = 0

    return documents

def embed(fname, url, window_size, step_size, logger):

    import re

    documents = load_documents(fname, url, logger)

    sentences = []
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
    #model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    model.max_seq_length = 512
    #cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    #cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    cross_encoder = CrossEncoder("corrius/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1")

    from llama_index.schema import TextNode
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    embeddings = []
    for node in nodes:
        embedding = model.encode(
            node.get_content(metadata_mode="all"),
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        node.embedding = embedding
        embeddings.append(embedding)

    return model, cross_encoder, embeddings, paragraphs, nodes

import chromadb
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext

args = parse_args()

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection(args.collection)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if len(chroma_collection.get()["ids"]) == 0:
    logger.info("--------------------- Load data to collection  \n")
    load_documents_to_db(args.filenames, args.url, vector_store)
else:
    logger.info("--------------------- Data already exist in collection  \n")

def load_to_db(vector_store, nodes, logger):
    logger.info("--------------------- Add data to the vector store \n")
    vector_store.add(nodes)

def search(query, model, cross_encoder, embeddings, paragraphs, top_k):
    query_embeddings = model.encode(query, convert_to_tensor=True)
    query_embeddings = query_embeddings.cuda()
    hits = util.semantic_search(
        query_embeddings,
        embeddings,
        top_k=top_k,
    )[0]
    cross_input = [[query, paragraphs[hit["corpus_id"]]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_input)
    for idx in range(len(cross_scores)):
        hits[idx]["cross_score"] = cross_scores[idx]
    results = []
    hits = sorted(hits, key=lambda x: x["cross_score"], reverse=True)
    for hit in hits:
        results.append(paragraphs[hit["corpus_id"]].replace("\n", " "))
    return results


if __name__ == "__main__":
    args = parse_args()
    model, cross_encoder, embeddings, paragraphs = embed(
        args.fname,
        args.window_size,
        args.step_size,
    )
    print(embeddings.shape)
    while True:
        print("\n")
        query = input("Enter query: ")
        results = search(
            query,
            model,
            cross_encoder,
            embeddings,
            paragraphs,
            top_k=args.top_k,
        )

        #query_7b = PREPROMPT + PROMPT.format(context="\n".join(results))
        #query_7b += END_7B.format(query=query)

        #query_40b = PREPROMPT + PROMPT.format(context="\n".join(results))
        #query_40b += END_40B.format(query=query)

        #text = ""
        #for response in CLIENT_7B.generate_stream(query_7b, **PARAMETERS):
        #    if not response.token.special:
        #        text += response.token.text

        #print("\n***7b response***")
        #print(text)

        #text = ""
        #for response in CLIENT_40B.generate_stream(query_40b, **PARAMETERS):
        #    if not response.token.special:
        #        text += response.token.text
#
        #print("\n***40b response***")
        #print(text)





fname = ["docs/ur/ur5e/ur5e_user_manual_en_us.pdf"]
url = "https://www.universal-robots.com/products/ur5-robot/"
window_size = 96
step_size = 76

logger = get_logger(False)

model, cross_encoder, embeddings, paragraphs, nodes = embed(
                    fname,
                    url,
                    window_size,
                    step_size,
                    logger,
              )
query = """What is this technical document/manual/specification about? What is company name? What is the product name?"""
results = search(
            query,
            model,
            cross_encoder,
            embeddings,
            paragraphs,
            top_k=10,
        )





#---------------------------------------------
fname="et200sp_cm_can.pdf"
fname="et200sp_system_manual.pdf"
fname="233.pdf"
window_size = 128
step_size = 100
model, cross_encoder, embeddings, paragraphs = embed(
        fname,
        window_size,
        step_size,
    )


fname="et200sp_cm_can.pdf"



query = """What is this technical document/manual/specification about? What is company name? What is the product name?"""
results = ['C Applied Standards This section describes relevant standards applied under the development of the robot arm and control box. Whenever a European Directive number is noted in brackets, it indicates that the standard is harmonized according to that Directive. A standard is not a law. A standard is a document developed by stakeholders within a given industry, deﬁning the normal safety and performance requirements for a product or product group. Abbreviations mean the following: ISO International Standardization Organization IEC International Electrotechnical Commission EN European Norm TS Technical Speciﬁcation TR Technical Report ANSI American National Standards Institute',
 'mentioned standards and/or normative documents is based on accredited tests and/or technical assessments carried out at DELTA – a part of FORCE Technology. Client Universal Robots A/S Energivej 25 5260 Odense Denmark Product identification (type(s), serial no(s).) UR robot generation 5, G5 for models UR3, UR5, and UR10 Manufacturer Universal Robots A/S Technical report(s) EMC test of UR robot generation 5, DELTA project no.117-29565-1 DANAK 19/18171 Standards/Normative documents EMC Directive 2014/30/EU, Article 6 EN 61326-3-1:2008 Industrial locations SIL 2 EN/(IEC) 61000-6-1:2007 EN/(IEC) 61000-6-2:2005 EN/(IEC) 61000-6-3:2007+A1 EN/(IEC) 61000-6-4:2007+A1 EN/(IEC) 61000-3-2:2014 EN/(IEC) 61000-3-3:2013 Version 5.7 Copyright © 2009–2020',
 'F I K A T C E R T I F I C A T E Fertigungsstätte Manufacturing plant Universal Robots A/S Energivej 25 5260 Odense S Denmark Beschreibung des Produktes (Details s. Anlage 1) Description of product (Details see Annex 1) Industrial robot UR16e, UR10e, UR5e and UR3e www.tuev-nord-cert.de Geprüft nach Tested in accordance with EN ISO 10218-1:2011 Registrier-Nr. / Registered No. 44 780 14097607 Prüfbericht Nr. / Test Report No. 3524 9416 Gültigkeit / Validity Aktenzeichen / File reference 8003008239 Bitte beachten Sie auch die umseitigen Hinweise Please also pay attention to the information',
 'B.2 Safety System Certiﬁcate Essen, 2019-07-16 Zertifizierungsstelle der TÜV NORD CERT GmbH Certification body of TÜV NORD CERT GmbH TÜV NORD CERT GmbH 45141 Essen Langemarckstraße 20 technology@tuev-nord.de berechtigt ist, das unten genannte Produkt mit dem abgebildeten Zeichen zu kennzeichnen. is authorized to provide the product described below with the mark as illustrated. Universal Robots A/S Energivej 25 5260 Odense S Denmark Hiermit wird bescheinigt, dass die Firma / This is to certify, that the company Z E R T I F I K A T C E R T I F I C A T',
 'B.2 Safety System Certiﬁcate B.2 Safety System Certiﬁcate Essen, 2019-07-16 Zertifizierungsstelle der TÜV NORD CERT GmbH TÜV NORD CERT GmbH 45141 Essen Langemarckstraße 20 technology@tuev-nord.de berechtigt ist, das unten genannte Produkt mit dem abgebildeten Zeichen zu kennzeichnen is authorized to provide the product mentioned below with the mark as illustrated Universal Robots A/S Energivej 25 5260 Odense S Denmark Hiermit wird bescheinigt, dass die Firma / This certifies that the company Z E R T I F I K A T C E R T I F I C A T E Fertigungsstätte Manufacturing plant Universal',
 'B.6 EMC Test Certiﬁcate The product identified above has been assessed and complies with the specified standards/normative docu- ments. The attestation does not include any market surveillance. It is the responsibility of the manufacturer that mass-produced apparatus have the same properties and quality. This attestation does not contain any statements pertaining to the requirements pursuant to other standards, directives or laws other than the above mentioned. Hørsholm, 15 August 2017 Michael Nielsen Specialist, Product Compliance DELTA – a part of FORCE Technology Venlighedsvej 4 2970 Hørsholm Denmark Tel. +45 72 19 40 00 Fax +45 72',
 'B.6 EMC Test Certiﬁcate B.6 EMC Test Certiﬁcate The product identified above has been assessed and complies with the specified standards/normative docu- ments. The attestation does not include any market surveillance. It is the responsibility of the manufacturer that mass-produced apparatus have the same properties and quality. This attestation does not contain any statements pertaining to the requirements pursuant to other standards, directives or laws other than the above mentioned. Hørsholm, 15 August 2017 Michael Nielsen Specialist, Product Compliance DELTA – a part of FORCE Technology Venlighedsvej 4 2970 Hørsholm Denmark Tel. +45 72 19 40',
 'Standardization Organization IEC International Electrotechnical Commission EN European Norm TS Technical Speciﬁcation TR Technical Report ANSI American National Standards Institute RIA Robotic Industries Association CSA Canadian Standards Association Conformity with the following standards is only guaranteed if all assembly instructions, safety instructions and guidance in this manual are followed. ISO 13849-1:2006 [PLd] ISO 13849-1:2015 [PLd] ISO 13849-2:2012 EN ISO 13849-1:2008 (E) [PLd – 2006/42/EC] EN ISO 13849-2:2012 (E) (2006/42/EC) Safety of machinery – Safety-related parts of control systems Part 1: General principles for design Part 2: Validation The safety control system is designed as Performance Level',
 'end-effector and intended use). Model: UR3e, UR5e, UR10e,UR16e (e-Series) Serial Number: Starting 20195000000 and higher — Effective 17 August 2019 Incorporation: Universal Robots UR3e, UR5e, UR10e and UR16e shall only be put into service upon being integrated into a ﬁnal complete machine (robot system, cell or application), which conforms with the provisions of the Machinery Directive and other ap- plicable Directives. It is declared that the above products, for what is supplied, fulﬁl the following Directives as Detailed Below: I Machinery Directive 2006/42/EC — The following essential requirements have been fulﬁlled: 1.1.2, 1.1.3, 1.1.5, 1.2.1, 1.2.4.3,',
 'must note the presence of directive 2009/127/EC. The declaration of incorporation according to 2006/42/EC annex II 1.B. is shown in appendix B. 2006/95/EC — Low Voltage Directive (LVD) 2004/108/EC — Electromagnetic Compatibility (EMC) 2011/65/EU — Restriction of the use of certain Hazardous Substances (RoHS) 2012/19/EU — Waste of Electrical and Electronic Equipment (WEEE) In the Declaration of Incorporation in appendix B, declarations of conformity with the above direc- tives are listed. A CE mark is afﬁxed according to the CE marking directives above. Information on both electric and electronic equipment waste is in chapter 7. Information']
ask = give a short answer of what this technical document/manual/specification about in form of 
{
"Document description",
"Company name",
"Product name",
"Product description"
}
based on the following information $results

answer:
{
"Document description": "Technical manual and documentation guide",
"Company name": "Siemens",
"Product name": "Communication Module CAN (6ES7137-6EA00-0BA0)",
"Product description": "The manual provides detailed information about the Communication Module CAN (6ES7137-6EA00-0BA0) for use in hazardous areas. It includes properties, wiring diagrams, characteristics, technical specifications, and general information about the SIMATIC ET 200SP distributed I/O system. The document also covers security information, recycling and disposal guidelines, and proper use of Siemens products. Additionally, it contains an overview of other SIMATIC documentation available for support and application examples."
}

query = """"What is specific about Communication Module CAN (6ES7137-6EA00-0BA0)? What is it for and what interfaces and protocols it is supporting?"""
results = ['the ET 200SP distributed I/O system. You connect the module directly to the ET 200SP CPU or to the ET 200SP interface module. You can find additional information on the configuration of the ET 200SP and the associated modules in the ET 200SP distributed I/O system (https://support.industry.siemens.com/cs/ww/en/view/58649293) system manual. 18 Communication Module CAN (6ES7137-6EA00-0BA0) Equipment Manual, 01/2023, A5E48404255-AD General properties Product overview 3.1 Properties The CAN communication module has the following properties: • CAN interface according to ISO 11898-2 (High Speed CAN) • The CAN protocol and CANopen protocol are implemented in the module. The module assumes the function of a CANopen slave or CANopen manager in the CANopen network. The module can operate in the following three modes: – CANopen manager – CANopen slave – CAN transparent',
 'You can operate the module in "CAN transparent" mode. • All CANopen functions are disabled. • Control and status information are exchanged cyclically between the module and the SIMATIC S7 controller. • Messages can be used in standard format and in extended CAN format. • Configured CAN messages with fixed message ID and fixed length can be used. • It is possible to use CAN messages in which the message ID and the length (max. 8 bytes) are changed at runtime. • The user can send and receive CAN messages in the user program. For this purpose, "Transmit proxy modules" and "Receive proxy modules" can be configured in the TIA Portal. Communication Module CAN (6ES7137-6EA00-0BA0) Equipment Manual, 01/2023, A5E48404255-AD 21 Product overview 3.2 System requirements 3.2 System requirements',
 'bit changes to "0", the module sets the transmit acknowledgment bit to "0" as soon as it has transmit- ted the message to be sent to the transmit buffer of the CAN controller. 0.1 0.0 Communication Module CAN (6ES7137-6EA00-0BA0) Equipment Manual, 01/2023, A5E48404255-AD 63 Functions 4.2 CAN transparent Receive proxy CAN messages can be received from the user program with the receive proxy. The module manages a list of message IDs ("filters") on the module side that it receives on the CAN bus. The list can be preassigned during configuration and changed at runtime from the user program by writing a corresponding data record. If a corresponding CAN message is received, it is entered in a receive buffer. The message is transmitted from this buffer to the SIMATIC',
 'The hardware identifier (HW identifier) is automatically assigned when the device configuration is defined in the TIA Portal. The HW identifier is included in diagnostics alarms so that the module involved can be located. The SIMATIC S7 controller requires the HW identifier in communication instructions in order to identify the module involved. See also Interface module (https://support.industry.siemens.com/cs/ww/en/ps/14034/man) CPU (https://support.industry.siemens.com/cs/ww/en/ps/13889/man) 72 Communication Module CAN (6ES7137-6EA00-0BA0) Equipment Manual, 01/2023, A5E48404255-AD Parameter assignment 7.1 Overview 7 The following section contains compact descriptions of the configuration steps for the 3 operating modes of the CAN communication module: • CANopen manager • CANopen slave • CAN transparent The descriptions are not based on a concrete example configuration. They are intended to illustrate the basic method for configuring the module and provide a quick',
 'status bit, to the SIMATIC S7 controller. As long as there is at least one configured message ID from which a CAN message has not yet been received, a zero is transmitted to the SIMATIC S7 controller in the status bit. Communication Module CAN (6ES7137-6EA00-0BA0) Equipment Manual, 01/2023, A5E48404255-AD 59 Functions 4.2 CAN transparent 4.2.6 Cyclical data exchange between the SIMATIC S7 controller and the module with programmed CAN messages (proxy) "Transmit proxies" and "Receive proxies" can be set up in the TIA Portal. These proxies serve as a "communication channel" that enables the user program to send and receive any CAN messages not configured beforehand. The use of multiple transmit / receive proxies facilitates programming of the user program because they are independent of the communication channels.']
ask = What is specific about Communication Module CAN (6ES7137-6EA00-0BA0)? What is it for and what interfaces and protocols it is supporting? Give a short answer in form of 
{
"Interfaces",
"protocols",
"Specific information",
}
based on the following information $results

{
"Interfaces": "CAN interface according to ISO 11898-2 (High Speed CAN)",
"Protocols": "CAN protocol and CANopen protocol",
"Specific information": "The module can operate as a CANopen slave or CANopen manager in the CANopen network. It also supports 'CAN transparent' mode, enabling cyclic exchange of control and status information between the module and the SIMATIC S7 controller. It can use both standard format and extended CAN format messages. The user can configure CAN messages with fixed or variable message IDs and lengths at runtime. Transmit and receive proxy modules can be set up for sending and receiving CAN messages in the user program."
}








fname="233.pdf"
query = """What is this technical document/manual/specification about? What is company name? What is the product name?"""
results = ['may be reproduced or transmitted in any form or by any means, electrical or mechanical including photocopying, recording or by any information-retrieval system without permission in writing from Technosoft S.A. The information in this document is subject to change without notice. About This Manual This manual describes how to program Technosoft iPOS family of intelligent drives using CANopen protocol. 1 The iPOS drives are conforming to CiA 301 v4.2 application layer and communication profile, CiA WD 305 v.2.2.130F Layer Setting Services and to CiA (DSP) 402 v4.0 device profile for drives and motion control, now included in IEC 61800-7-1 Annex A, IEC 61800-7-201 and IEC 61800-7-301 standards. The manual presents the object dictionary associated with these three profiles. It also explains how to combine the Technosoft Motion Language',
 '2023 10 iPOS CANopen Programming 22.6 Customizing the Drive Reaction to Fault Conditions ......................................... 231 Read This First Whilst Technosoft believes that the information and guidance given in this manual is correct, all parties must rely upon their own skill and judgment when making use of it. Technosoft does not assume any liability to anyone for any loss or damage caused by any error or omission in the work, whether such error or omission is the result of negligence or any other cause. Any and all such liability is disclaimed. All rights reserved. No part or parts of this document may be reproduced or transmitted in any form or by any means, electrical or mechanical including photocopying, recording or by any information-retrieval system without permission in writing from',
 'Name Object code Data type Access PDO mapping Value range Default value 100Ah Manufacturer software version VAR Visible String Const No No Product dependent 5.8.5 Object 2060h: Software version of a TML application By inspecting this object, the user can find out the software version of the TML application (drive setup plus motion setup and eventually cam tables) that is stored in the EEPROM memory of the drive. The object shows a string of the first 4 elements written in the TML application field, grouped in a 32-bit variable. If more character are written, only the first 4 will be displayed. Each byte represents an ASCII character. Object description: Entry description: Example: Index Name Object code Data type Access PDO mapping Units Value range Default value 2060h Software',
 '1018h: Identity Object This object provides general information about the device. Sub-index 01h shows the unique Vendor ID allocated to Technosoft (1A3h). Sub-index 02h contains the Technosoft drive product ID. It can be found physically on the drive label or in Drive Setup/ Drive info button under the field product ID. If the Technosoft product ID is P027.214.E121, sub-index 02h will be read as the number 27214121 in decimal. Sub-index 03h shows the Revision number. Sub-index 04h shows the drives Serial number. For example the number 0x4C451158 will be 0x4C (ASCII L); 0x45 (ASCII E); 0x1158 --> the serial number will be LE1158. Object description: Entry description: Index Name Object code Data type Sub-index Description Access PDO mapping Value range Default value Sub-index Description Access PDO mapping Value',
 'Bit Timing Parameters ............................................................................................................. 24 2.2.6 ...... Store Configuration Protocol ................................................................................................................. 25 2.2.7 ...... Inquire Identity Vendor ID ..................................................................................................................... 25 2.2.8 ...... Inquire Identity Product Code ................................................................................................................ 25 2.2.9 ...... Inquire Identity Revision Number .......................................................................................................... 25 2.2.10..... Inquire Identity Serial Number ............................................................................................................... 26 2.2.11..... Inquire Identity Node ID ........................................................................................................................ 26 2.2.12..... Identify Remote Slave ........................................................................................................................... 26 2.2.13..... Identify non-configured Remote Slave .................................................................................................. 26 © Technosoft 2023 2 iPOS CANopen Programming 3 CAN and the CANopen protocol ........................................................................... 27 3.1 CAN Architecture ....................................................................................................... 27 3.2 Accessing CANopen devices .................................................................................... 27 3.2.1 ...... Object dictionary ................................................................................................................................... 27 3.2.2 ...... Object access using index and sub-index ............................................................................................. 27 3.2.3 ...... Service Data Objects (SDO) ................................................................................................................. 28 3.2.4 ...... Process Data Objects (PDO) ................................................................................................................ 28 3.3 Objects that define SDOs and PDOs']
ask = give a short answer of what this technical document/manual/specification about in form of 
{
"Document description",
"Company name",
"Product name",
"Product description"
}
based on the following information $results
{
"Document description": "Technical manual for Technosoft iPOS family of intelligent drives using CANopen protocol",
"Company name": "Technosoft S.A.",
"Product name": "iPOS drives",
"Product description": "The manual describes how to program Technosoft iPOS family of intelligent drives using CANopen protocol. The iPOS drives conform to CiA 301 v4.2 application layer and communication profile, CiA WD 305 v.2.2.130F Layer Setting Services, and CiA (DSP) 402 v4.0 device profile for drives and motion control, included in IEC 61800-7-1 Annex A, IEC 61800-7-201, and IEC 61800-7-301 standards. The manual presents the object dictionary associated with these profiles and explains how to combine the Technosoft Motion Language. It also provides information about Bit Timing Parameters and various Inquire Identity details for the drives."
}


query = """"What is specific about iPOS drives? What is it for and what interfaces and protocols it is supporting?"""
results=['iPOS family of Technosoft intelligent drives. Notational Conventions This document uses the following conventions: TML – Technosoft Motion Language iPOS – a Technosoft drive family, the code is usually iPOSxx0x xx-CAN GUI – Graphical User Interface IU – drive/motor internal units IP – Interpolated Position RegisterY.x- bit x or register Y; Example: Controlword.5 – bit 5 of Controlword data cs – command specifier CSP – Cyclic Synchronous Position CSV – Cyclic Synchronous Velocity CST – Cyclic Synchronous Torque Axis ID or CAN ID or COB ID – the unique number allocated to each drive in a network. RO – read only RW – read and write SW – software H/W or HW - hardware 1 Available only with the firmware F514x. © Technosoft 2023 11 iPOS CANopen Programming',
 'may be reproduced or transmitted in any form or by any means, electrical or mechanical including photocopying, recording or by any information-retrieval system without permission in writing from Technosoft S.A. The information in this document is subject to change without notice. About This Manual This manual describes how to program Technosoft iPOS family of intelligent drives using CANopen protocol. 1 The iPOS drives are conforming to CiA 301 v4.2 application layer and communication profile, CiA WD 305 v.2.2.130F Layer Setting Services and to CiA (DSP) 402 v4.0 device profile for drives and motion control, now included in IEC 61800-7-1 Annex A, IEC 61800-7-201 and IEC 61800-7-301 standards. The manual presents the object dictionary associated with these three profiles. It also explains how to combine the Technosoft Motion Language',
 'Access PDO mapping Value range Default value 6502h Supported drive modes VAR UNSIGNED32 RO Possible UNSIGNED32 001F0065h for iPOS family The modes of operation supported by the Technosoft drives, and their corresponding bits, are the following: Data description: MSB LSB x 0 0 Manufacturer specific 31 21 20 … x … 16 0 rsvd 15 0 1 ip ... 7 6 1 hm 5 0 rsvd 4 0 tq 3 1 pv 2 0 vl 1 1 pp 0 Data description – manufacturer specific: Bit 31 … 21 20 19 18 Description Reserved External Reference Torque Mode External Reference Speed Mode External Reference Position Mode © Technosoft 2023 76 iPOS CANopen Programming 17 16 Electronic Gearing Position Mode Electronic Camming Position Mode 5.8.3 Object 1008h: Manufacturer Device Name',
 'CANopen predefined connection set (as defined by CiA DS301 v4.2.0). Thus, TechnoCAN protocol and CANopen protocol can co-exist and communicate simultaneously on the same physical CAN bus, without disturbing each other. 1.2 Changing the drive Axis ID (Node ID) The axis ID of an iPOS drive can be set in 3 ways: ▪ ▪ ▪ Hardware (H/W) – depending on the drive type, it can be via H/W pins or switches. Software (via Setup)– any value between 1 and 255, stored in the setup table. Software (via CANopen master) – using CiA-3051F 1 protocol Remark: • ▪ If the drive is in CANopen mode, a Node ID value above 127 is automatically converted into 255 and the drive is set with CAN communication “non-configured” mode waiting for a',
 'iPOS CANopen Programming 1.1.3 Establishing serial communication with the drive EasySetup communicates with the drive via an RS-232 serial link or CAN interface. If your PC has no serial port, use an USB to RS232 adapter. For the serial connections, refer to the drive Technical Reference manual. If the drive or the Starter Kit board accompanying the drive has a 9-pin serial port, use a standard 9-wire, non-inverting (one to one) serial cable. Figure 1.1.1. EasySetup - Opening window All Technosoft drives with CAN interface have a unique AxisID (address) for serial communication. The AxisID value is by default 255 or it is set by the levels of the AxisID selection inputs, when these exist. Remark: When first started, EasySetup tries to communicate via RS-232 and COM1 with']
ask = What is specific about Communication Module CAN (6ES7137-6EA00-0BA0)? What is it for and what interfaces and protocols it is supporting? Give a short answer in form of 
{
"Interfaces",
"protocols",
"Specific information",
}
based on the following information $results
{
"Interfaces": "CAN interface, RS-232 serial link",
"Protocols": "TechnoCAN protocol, CANopen protocol",
"Specific information": "The Communication Module CAN (6ES7137-6EA00-0BA0) allows for serial communication with Technosoft iPOS family of intelligent drives. It supports both TechnoCAN protocol and CANopen protocol simultaneously on the same physical CAN bus without interference. The module also enables communication via an RS-232 serial link, and the drive's Axis ID can be set either through hardware pins/switches or via software using CiA-3051F protocol."
}

ask = 
based on follwoing two queries from different documentation sources for different devices, can this devices communicate with each other? And with each communication protocol?

answer
Based on the provided information, it appears that both devices support communication via the CAN protocol. However, it is important to note that they are using different implementations of the CAN protocol:

Device 1:

    Device name: ET 200SP distributed I/O system
    Communication module: CAN (6ES7137-6EA00-0BA0)
    Supported CAN protocols: CAN protocol and CANopen protocol
    Operation modes: CANopen manager, CANopen slave, and CAN transparent mode

Device 2:

    Device name: iPOS family of Technosoft intelligent drives
    Communication protocol: CANopen protocol
    Additional Information: TechnoCAN protocol and CANopen protocol can co-exist and communicate simultaneously on the same physical CAN bus.

While both devices support CAN communication, they may not be directly compatible due to their different implementations of the CAN protocol. However, if both devices support the CANopen protocol, they should be able to communicate with each other using that protocol.




fname="nanotec.pdf"
query = """What is this technical document/manual/specification about? What is company name? What is the product name?"""
resutl= ['Technical Manual PD6-E Fieldbus: CANopen For use with the following variants: PD6-E891S95-E-65-2, PD6-E891M95-E-65-2, PD6-E891L95-E-65-2, PD6-EB80SD- E-65-2, PD6-EB80MD-E-65-2, PD6-EB80LD-E-65-2, PD6-EB80CD-E-65-2 Valid with firmware version FIR-v2213 and since hardware version W003 Technical Manual Version: 1.0.0 Contents Contents 1 Introduction...................................................................................................10 1.1 Version information................................................................................................................................... 10 1.2 Copyright, marking and contact................................................................................................................10 1.3 Intended use............................................................................................................................................. 10 1.4 Warranty and disclaimer........................................................................................................................... 10 1.5 Target group and qualification..................................................................................................................11 1.6 EU directives for product safety............................................................................................................... 11 1.7 Other applicable regulations..................................................................................................................... 11 1.8 Used icons................................................................................................................................................ 11 1.9 Emphasis in the text................................................................................................................................. 12 1.10 Numerical values.................................................................................................................................... 12 1.11 Bits.......................................................................................................................................................... 12 1.12 Counting direction (arrows).....................................................................................................................12 2 Safety and warning notices........................................................................ 14 3 Technical details and pin assignment....................................................... 15 3.1 Environmental conditions.......................................................................................................................... 15 3.2 Dimensioned drawings..............................................................................................................................15 3.2.1 PD6-E891S95-E-65-2..................................................................................................................... 15 3.2.2 PD6-E891M95-E-65-2.....................................................................................................................16 3.2.3 PD6-E891L95-E-65-2......................................................................................................................16 3.2.4 PD6-EB80SD-E-65-2...................................................................................................................... 16 3.2.5 PD6-EB80MD-E-65-2......................................................................................................................17 3.2.6 PD6-EB80LD-E-65-2.......................................................................................................................17 3.2.7',
 'the object dictionary. In the following, the owner of the object dictionary is referred to as the "server"; the CAN node – which wants to request or write the data – is referred to as the "client". An "upload" refers to the reading of a value of an object from the object dictionary; a "download" refers to the writing of a value in the object dictionary. In addition, the following abbreviations are used in the diagrams: ■ <IDX>: Index of the object that is to be read or written in the object dictionary; the LSB of the index is in byte 1 here. Example: The statusword of the controller has index 6041h; byte 1 is then written with 41h and byte 2 with 60h. With Expedited Transfer, the',
 'Technical details and pin assignment....................................................... 15 3.1 Environmental conditions.......................................................................................................................... 15 3.2 Dimensioned drawings..............................................................................................................................15 3.2.1 PD6-E891S95-E-65-2..................................................................................................................... 15 3.2.2 PD6-E891M95-E-65-2.....................................................................................................................16 3.2.3 PD6-E891L95-E-65-2......................................................................................................................16 3.2.4 PD6-EB80SD-E-65-2...................................................................................................................... 16 3.2.5 PD6-EB80MD-E-65-2......................................................................................................................17 3.2.6 PD6-EB80LD-E-65-2.......................................................................................................................17 3.2.7 PD6-EB80CD-E-65-2...................................................................................................................... 17 3.3 Electrical properties and technical data....................................................................................................18 3.3.1 Electrical data................................................................................................................................. 18 3.3.2 Technical data................................................................................................................................ 18 3.4 Overtemperature protection...................................................................................................................... 19 3.5 LED signaling............................................................................................................................................23 3.5.1 Power LED......................................................................................................................................23 3.6 Pin assignment......................................................................................................................................... 24 3.6.1 Overview......................................................................................................................................... 24 3.6.2 X1 — voltage supply...................................................................................................................... 24 3.6.3 X2 — Inputs/outputs and external logic supply..............................................................................26 3.6.4 X3 – CANopen IN and logic supply............................................................................................... 28 3.6.5 X4 – CANopen OUT and logic supply........................................................................................... 29 3.6.6 X5 – external holding brake........................................................................................................... 29 3.6.7 X6 — Mini USB.............................................................................................................................. 30 3.6.8 S1 switch – Hex coding switch for node-ID and baud rate............................................................ 30 3.6.9 S2 switch – 120 ohm termination resistor......................................................................................31 4',
 "form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 3. Neither the name of the Institute nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE INSTITUTE AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE INSTITUTE OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR",
 'reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. ■ Neither the name of ARM nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS']
answer=
{
"Document description": "Technical Manual",
"Company name": "Unknown",
"Product name": "PD6-E Fieldbus: CANopen",
"Product description": "This technical manual provides information and instructions for the PD6-E Fieldbus: CANopen, designed for use with various product variants including PD6-E891S95-E-65-2, PD6-E891M95-E-65-2, PD6-E891L95-E-65-2, PD6-EB80SD-E-65-2, PD6-EB80MD-E-65-2, PD6-EB80LD-E-65-2, and PD6-EB80CD-E-65-2. The manual covers version 1.0.0 of the product and includes details about environmental conditions, dimensioned drawings, electrical properties, pin assignment, safety and warning notices, and more. It also explains the use of the object dictionary and CANopen communication."
}
query = """"What is specific about PD6-E Fieldbus: CANopen? What is it for and what interfaces and protocols it is supporting?"""
result=['interfaces, such as PDO configuration etc. ■ Application: Parameters related to operating modes. ■ Customer: Parameters that are written and read by the customer/user only and are ignored by the controller firmware. ■ Drive: Parameters related to the motor and the sensors (BLDC/Stepper, closed/open-loop…). Some are set and saved by auto setup. ■ Tuning: Parameters related to motor and encoder that are set either by auto setup or that can be found in the data sheets, e.g., pole pairs and maximum current. ■ CANopen: Parameters related to CANopen communication If an object is not assigned one of these categories, it cannot be saved, e.g., statusword and all objects whose value is dependent on the current state of the controller. The objects in each category are listed below. In',
 'contains all physical interfaces/protocols that can be activated/deactivated. These can be switched in this object (2102h). The current status of the activated fieldbuses is in object 2103h:2h. The following distribution of the bits applies here: 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 E-IP MTCP MRTU 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 SPI E-CAT E-NET CAN RS232 RS485 USB Version: 1.0.0 / FIR-v2213 273 10 Description of the object dictionary USB USB interface RS-485 RS-485 interface RS-232 RS-232 interface CAN CANopen interface E-NET EtherNet interface E-CAT EtherCAT interface SPI SPI interface MRTU Modbus RTU protocol MTCP E-IP Modbus TCP protocol EtherNet/IP™ protocol 2103h Fieldbus Module Status Function Shows the active',
 'to set a value. PDO mapping Some bus systems, such as CANopen or EtherCAT, support PDO mapping. Described in this table entry is whether the object can be inserted into a mapping and, if so, into which. The following designations are available here: ■ "no": The object may not be entered in a mapping. ■ "TX-PDO": The object may be entered in an RX mapping. ■ "RX-PDO": The object may be entered in a TX mapping. Allowed values In some cases, only certain values may be written in the object. If this is the case, these values are listed here. If there are no restrictions, the field is empty. Preset value To bring the controller to a secured state when switching on, it is necessary to preset a',
 'exchange data during running operation, use CANopen messages of type PDO (see subsection Process Data Object). The SDO transfers are divided into three types of access: ■ "expedited transfer" for transferring objects with up to four bytes. ■ "normal transfer" for transferring any number of bytes, whereby each CAN message is confirmed individually. ■ "block transfer" is also for any number of bytes; here, a given block of CAN tickets is confirmed at once. An SDO message is sent to CAN-ID 600h + node-ID, the answer comes from CAN-ID 580h + node-ID. 8.2.4.1 Expedited Transfer This method is used to write (download) or read (upload) values in objects of type (UN)SIGNED8, INTEGER16 oder INTEGER32 in the object dictionary. This service is confirmed, i.e., each access is answered with',
 'referred to as a "Process Data Object" (PDO). The PDO is intended for data that need to be exchanged cyclically. The idea behind a PDO message is to remove all additional information (index, subindex and data length) from a CAN message and to only fill the CAN message with data. The source and target information for the PDO are stored separately in the so-called PDO mapping. PDOs can only be used if the NMT state machine is in the "Operational" state (see section Network Management (NMT)); the PDOs must be configured in the "Pre-Operational" NMT state. The controller supports a total of 8 independent PDO mappings; each corresponding PDO message can have a maximum of eight bytes (=64 bit) of user data. It is thereby possible to, for']
ask = What is specific about PD6-E Fieldbus: CANopen? What is it for and what interfaces and protocols it is supporting? Give a short answer in form of 
{
"Interfaces",
"protocols",
"Specific information",
}
based on the following information $results
answer=
{
"Interfaces": "CAN",
"Protocols": "CANopen",
"Specific information": "The PD6-E Fieldbus: CANopen supports the CAN interface and the CANopen protocol. It allows communication through Process Data Objects (PDOs) and Service Data Objects (SDOs) and provides various parameter categories for configuration, such as Application, Customer, Drive, Tuning, and CANopen parameters. The controller also offers 8 independent PDO mappings, each capable of transmitting up to 8 bytes of user data."
}





















results = search(
    query,
    model,
    cross_encoder,
    embeddings,
    paragraphs,
    top_k=32,
)


what is CM CAN based on the following text: a CANopen node with its entire configuration remains in the project but is not downloaded to the CM CAN module. The node does not exist in the configuration that is used by the CM CAN module. No errors or warnings are generated by the module. Figure 7-20 Disabling a node temporarily When such a node appears in the menu, it is marked as "(disabled)". Communication Module CAN (6ES7137-6EA00-0BA0) Equipment Manual, 01/2023, A5E48404255-AD 99 Parameter assignment 7.2 Configuring CANopen manager Figure 7-21 Node disabled Checking data consistency You can check the consistency of the assignments for the receive data and transmit data as well as the data types used with a compilation. Setting the module to "Operational" with the user program To enable data to be transferred between the', 'Communication Module CAN (6ES7137-6EA00-0BA0) Equipment Manual, 01/2023, A5E48404255-AD 13 New properties/functions 2.2 Changes compared to previous version 2.2 Changes compared to previous version What\'s new as of firmware version V1.2 There are now three entries in the HW catalog for the CM CAN module with firmware versions V1.0, V1.1 and V1.2. Version V1.2 includes all functions of Version V1.1 plus a new "Block Error passive alarm" function. The OD index entry range has been extended to 0x6FFF. These functions are available as of TIA Portal V17 Update 6. Note Backward compatibility An existing configuration of CM CAN module V1.2 cannot be downgraded from V1.2 to V1.1 within a TIA Portal project. The same applies to a downgrade to Version V1.0 from a higher version (V1.1 and higher). If', 'of the control bits under Control and Status Information (Page 34). 100 Communication Module CAN (6ES7137-6EA00-0BA0) Equipment Manual, 01/2023, A5E48404255-AD Parameter assignment 7.3 Configuring CANopen slave 7.3 Configuring CANopen slave 7.3.1 Overview Configuration with the HSP in the TIA Portal (CM CAN is NMT slave). Import the HSP of the CM CAN module to the TIA Portal. The module is configured in the TIA Portal. The configuration for "CANopen slave" operating mode mainly consists of the following steps: • • Drag the module from the HW catalog to the project. • Set the "CANopen slave" operating mode. • Set bus-specific parameters for the "CANopen slave" module: – Node ID – Transmission rate • Create OD entries for the process data to be exchanged between the SIMATIC S7 controller', 'alarm This option can be selected in any operating mode. If you select this option and the CM CAN module receives an Error Passive alarm, the error passive state of the CAN bus is not reported as an alarm in the diagnostic buffer. The ERROR LED remains off. If you deselect this option and the CM CAN module receives an error passive alarm, the Error Passive state of the CAN bus is reported as an alarm in the diagnostic buffer. The ERROR LED on the device flashes red. This option is enabled by default. Note If the "Enable additional diagnostic alarms" check box is deselected, the "Block Error passive alarm" check box is disabled. Under "General > Module parameters > Diagnostics" • Check box Enable diagnostic alarms for', 'are control and status information. The control and status information is exchanged with the user program via these addresses. It is important for startup of the module that the control information transferred here from the user program is set correctly. 98 Communication Module CAN (6ES7137-6EA00-0BA0) Equipment Manual, 01/2023, A5E48404255-AD Parameter assignment 7.2 Configuring CANopen manager Disabling a node temporarily (for configuration with HSP_V16_0310_003_ET200SP_CM_CAN_1.0) You have configured a series of nodes in the TIA Portal, but one node does not physically exist in the CAN network. To work with the "reduced" network, you can temporarily disable one or more nodes. Such a CANopen node with its entire configuration remains in the project but is not downloaded to the CM CAN module. The node does not exist in the configuration




























for i in range(0, len(text_tokens), 100):
        window = text_tokens[i : i + 128]
        if len(window) < 128:
            break
        sentences.append(window)


text = extract_text(fname)
text = " ".join(text.split())
text_tokens = text.split()
sentences = []
for i in range(0, len(text_tokens), step_size):
    window = text_tokens[i : i + window_size]
    if len(window) < window_size:
        break
    sentences.append(window)
paragraphs = [" ".join(s) for s in sentences]
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
model.max_seq_length = 512
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
embeddings = model.encode(
    paragraphs,
    show_progress_bar=True,
    convert_to_tensor=True,
)