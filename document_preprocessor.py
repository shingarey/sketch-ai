import re

from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index import Document, download_loader
from llama_index.prompts import PromptTemplate
from llama_index.schema import TextNode
from llmsherpa.readers import LayoutPDFReader


class DocumentPreprocessor:

    """
    The DocumentPreprocessor class is responsible for processing different
    types of documents, including normal PDFs and Sherpa table PDFs.
    """

    def __init__(self, logger, url, pdf_filenames, collection_name):
        """
        Initializes a DocumentPreprocessor object.
        Args:
        - logger (Logger): a logger object.
        - url (str): a URL string.
        - pdf_filenames (list): a list of PDF file names.
        - collection_name (str): the name of the collection.
        Attributes:
        - logger (Logger): a logger object.
        - nodes (list): a list of TextNode objects.
        - url (str): a URL string.
        - pdf_filenames (list): a list of PDF file names.
        - collection_name (str): the name of the collection.
        """
        self.logger = logger
        self.nodes = []
        self.url = url
        self.pdf_filenames = pdf_filenames
        self.collection_name = collection_name

    def get_nodes(self):
        return self.nodes

    def load_url_documents(self):
        """
        Loads documents from URLs.
        Returns:
        - url_docs (list): a list of Document objects.
        """
        self.logger.info("--------------------- Load urls \n")
        ReadabilityWebPageReader = download_loader("ReadabilityWebPageReader")
        loader_url = ReadabilityWebPageReader()
        if self.url:
            docs = loader_url.load_data(url=self.url)
            doc = docs[0]
            doc.metadata["file_path"] = self.url
            t = re.sub("\n\n", " ", doc.text)
            doc.text = t
            return [doc]
        return []

    def load_pdf_documents(self):
        """
        This function is responsible for loading PDF documents and returning
        two lists of `Document` objects: `pdf_docs` and `pdf_docs_sherpa`.
        Returns:
        - pdf_docs (list): a list of Document objects.
        - pdf_docs_sherpa (list): a list of Document objects.
        """
        loader_pdf = PyMuPDFReader()
        llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
        pdf_reader = LayoutPDFReader(llmsherpa_api_url)
        pdf_docs = []
        pdf_docs_sherpa = []
        for file in self.pdf_filenames:
            self.logger.info(
                "--------------------- Load local PDF document {} \n".format(file)
            )
            pdf_docs.append(loader_pdf.load(file_path=file))
            self.logger.info(
                "--------------------- Ask Sherpa to analyze PDF document\n"
            )
            pdf_docs_sherpa.append(pdf_reader.read_pdf(file))
        return pdf_docs, pdf_docs_sherpa

    def remove_none_fields(self, docs):
        """
        Removes fields with None values from a list of documents.
        Args:
        - docs (list): a list of Document objects.
        """
        for doc in docs:
            for key in doc.metadata:
                if doc.metadata[key] is None:
                    doc.metadata[key] = 0

    def load_documents(self):
        """
        Loads documents from different sources.
        Returns:
        - url_docs (list): a list of Document objects.
        - pdf_docs (list): a list of Document objects.
        - pdf_docs_sherpa (list): a list of Document objects.
        """
        url_docs = self.load_url_documents()
        pdf_docs, pdf_docs_sherpa = self.load_pdf_documents()
        self.remove_none_fields(url_docs)
        return url_docs, pdf_docs, pdf_docs_sherpa

    def process_sherpa_pdf(self, documents):
        """
        Processes sherpa PDF documents.
        Args:
        - documents (Document): a Document object.
        """
        for chunk in documents.chunks():
            self.nodes.append(Document(text=chunk.to_context_text(), extra_info={}))

    def clear_nodes(self):
        """
        Clears the nodes list.
        """
        self.nodes = []

    def process_normal_pdf(self, llm, documents):
        """
        The function uses the OpenAI object to generate a detailed summary of
        each PDF page and adds the summary to the `nodes` list as a `TextNode`
        object.

        Args:
        - llm (OpenAI): an OpenAI object.
        - documents (list): a list of Document objects.
        """
        qa_prompt2 = PromptTemplate(
            """\
            Read this PDF page and summarize semantically in form of sentences. \
            Start each sentence with a new line. \
            Retain all the technical specification data. \
            Translate to english before if required. \
            PDF page: '{pdf_page}'
            Answer: \
            """
        )
        qa_prompt = PromptTemplate(
            """\
            Read this PDF page and prepare a detailed summary of it. Start each sentence with new line. \
            Retain all the technical specification data. \
            Translate to english before if required. \
            PDF page: '{pdf_page}'
            Answer: \
            """
        )

        for doc_idx, doc in enumerate(documents):
            if len(doc.text) > 200:
                self.logger.info(
                    "--------------------- Ask LLM to summarize page {page} from PDF"
                    " {pdf} \n".format(page=doc_idx, pdf=doc.metadata["file_path"])
                )
                fmt_qa_prompt = qa_prompt.format(pdf_page=doc.text)
                response = llm.complete(fmt_qa_prompt)
                for line in response.text.splitlines():
                    src_doc = documents[doc_idx]
                    node = TextNode(
                        text=line,
                    )
                    node.metadata = src_doc.metadata
                    self.nodes.append(node)
                    self.logger.debug("text: {}".format(line))

    def process_sherpa_table(self, llm, documents):
        """
        Processes sherpa table PDF documents.

        Args:
        - llm (OpenAI): an OpenAI object.
        - documents (Document): a Document object.
        """
        from llama_index.prompts import PromptTemplate
        from llama_index.readers.schema.base import Document

        qa_prompt = PromptTemplate(
            """\
            read this table and prepare a detailed summary of it:
            Table: '{table}'
            Answer: \
            """
        )

        self.logger.info("--------------------- Process sherpa table PDF \n")
        table_text = documents.to_context_text()
        fmt_qa_prompt = qa_prompt.format(table=table_text)
        self.logger.info("--------------------- Ask LLM to summarize table\n")
        response = llm.complete(fmt_qa_prompt)
        lines = str(response.text).splitlines()
        for i in lines:
            if not i:
                lines.remove(i)
        for i in range(len(lines)):
            if lines[i]:
                text = lines[i]
                self.nodes.append(
                    Document(
                        text=text,
                        extra_info={},
                    )
                )
                self.logger.debug("text: {}".format(text))

    def create_collection_dict(self) -> dict:
        """
        This function creates a dictionary of collections from the loaded documents.

        Returns:
        - collection_dict (dict): a dictionary of collections.
        """

        url_docs, pdf_docs, pdf_docs_sherpa = self.load_documents()
        collection_dict = {}
        if url_docs:
            collection_dict[self.collection_name + "_url1"] = [url_docs[0]]
        # collection_dict[self.collection_name + "_url2"] = [url_docs[1]]
        for idx, name in enumerate(self.pdf_filenames):
            for i in range(len(pdf_docs[0])):
                collection_dict[self.collection_name + "_pdf_" + str(idx) + str(i)] = [
                    pdf_docs[idx][i]
                ]

            collection_dict[
                self.collection_name + "_pdf_sherpa_" + str(idx)
            ] = pdf_docs_sherpa[idx]

            for table_id, table in enumerate(pdf_docs_sherpa[idx].tables()):
                collection_dict[
                    self.collection_name
                    + "_pdf_sherpa_table_"
                    + str(idx)
                    + str(table_id)
                ] = pdf_docs_sherpa[idx].tables()[table_id]

        return collection_dict
