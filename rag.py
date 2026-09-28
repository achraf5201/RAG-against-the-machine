from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from rank_bm25 import BM25Okapi

# print(RecursiveCharacterTextSplitter)

class RAG:
    def __init__(self):
        self.content = ""

    def load_data(self, filename):
        with open(filename, "r") as f:
            self.content = f.read()
    
    def chunker(self):
        chunker_td = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size = 200,
            chunk_overlap = 20,
            add_start_index = True

        )

        data = self.load_data("data.MD")
        chunks = chunker_td.create_documents([self.content])
        for i, chunk in enumerate(chunks):
            print("chunk ", i)
            lstinx = chunk.metadata["start_index"] + len(chunk.page_content)
            print(chunk.metadata["start_index"], lstinx)
            print(chunk.page_content)
            print("\n\n")
        
        # print(len(chunks))
        return chunks

    # def bm25_handler(self, query):
    #     tokenized_chunk = [chunk.lower().split() for chunk in self.chunker()]

    #     bm25 = BM25Okapi(tokenized_chunk)
    #     tokenized_query = query.split(" ")
    #     top_k = bm25.get_top_n(tokenized_query, tokenized_chunk, n=1)[0]



d = RAG()
# d.bm25_handler("The Code of Criminal Procedure")
d.chunker()

