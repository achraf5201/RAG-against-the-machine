from src.chunking import Chunking







r = Chunking()

r.chunking_markdown()
r.chunking_python()

with open("file.txt", "w") as f:
    for chunk in r.chunks:
        print(str(chunk.metadata))
        # print(chunk.page_content)
        # print(chunk.metadata)
        f.write(chunk.page_content)
        f.write('\n')
        f.write(str(chunk.metadata["source"]))
        f.write("\n\n\n")