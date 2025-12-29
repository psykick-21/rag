from src.ai.rag.models import RetrievalResult
from src.ai.rag.models import DocumentChunk, DocumentChunkEmbedding
from src.db.connection import conn

from typing import List
from openai import OpenAI
from pgvector.psycopg import Vector
from dotenv import load_dotenv

load_dotenv()


class Retriever:
    """
    Responsible only for retrieving relevant document chunks.
    """

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Retrieves the relevant document chunks using the OpenAI API."""

        client = OpenAI()

        query_embedding = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        ).data[0].embedding

        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM file_chunks
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (Vector(query_embedding), top_k)
            )
            fetched_chunks = cursor.fetchall()
            chunks = [DocumentChunk(
                content=chunk[3],
                source=chunk[1],
                metadata={"chunk_index": chunk[2]}
            ) for chunk in fetched_chunks]

        return RetrievalResult(chunks=chunks)


    def _chunk_document_text(
        self, document_text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[DocumentChunk]:
        """Chunks the document text into smaller chunks of the given size."""

        chunks = []
        idx = 0

        for i in range(0, len(document_text), chunk_size - overlap):
            chunk_text = document_text[i:i + chunk_size]
            chunks.append(
                DocumentChunk(
                    content=chunk_text,
                    source="document",
                    metadata={"chunk_index": idx}
                )
            )
            idx += 1
        return chunks

    
    def _embed_chunks(
        self,
        chunks: List[DocumentChunk]
    ) -> List[DocumentChunkEmbedding]:
        """Embeds the chunks using the OpenAI API."""

        client = OpenAI()

        embeded_chunks = []

        for chunk in chunks:
            embedding = client.embeddings.create(
                input=chunk.content,
                model="text-embedding-3-small"
            )
            embeded_chunk = DocumentChunkEmbedding(
                document_chunk=chunk,
                embedding=embedding.data[0].embedding
            )
            embeded_chunks.append(embeded_chunk)
        
        return embeded_chunks


    def _save_embeddings_to_db(
        self,
        embeddings: List[DocumentChunkEmbedding]
    ):
        """Saves the embeddings to the database."""

        with conn.cursor() as cursor:
            for embedding in embeddings:
                cursor.execute(
                    "INSERT INTO file_chunks (file_name, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                    (embedding.document_chunk.source, embedding.document_chunk.metadata["chunk_index"], embedding.document_chunk.content, embedding.embedding)
                )
            conn.commit()


if __name__ == "__main__":
#     document_text = """# Text Summarization API Using HuggingFace, FAST API and Deployment to AWS with CI/CD Pipeline

# >***A short video demonstrating the working of the API***<br>
# https://www.youtube.com/watch?v=QG-pj9tV81M

# ## A brief workflow of the project
# 1. Setup logging
# 1. Getting the dataset and model from HuggingFace
# 2. Build modules and pipelines (mentioned below in pipelines section) right from data gathering to model training and inferencing
# 3. Use FAST API and Uvicorn to create training and inferencing endpoints.
# 5. Deployment of the dockerized app to ECR and ECS on AWS
# 6. Build CI/CD pipeline using GitHub Actions
# 5. Postman for inferencing


# ## Pipelines
# 1. Data Ingestion
# 2. Data Transformation
# 3. Data Validation
# 4. Model Training
# 5. Model Evaluation
# 6. Inferencing


# ### References
# **Model**: Hugging Face [Google Pegasus](https://huggingface.co/google/pegasus-cnn_dailymail)<br>
# Citations:Jingqing Zhang, Yao Zhao, Mohammad Saleh, & Peter J. Liu. (2019). PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization.<br>
 
# **Dataset**: Hugging Face [Samsum dataset](https://huggingface.co/datasets/samsum)<br>
# License: CC BY-NC-ND 4.0<br>
# Citations: Gliwa, B., Mochol, I., Biesek, M., & Wawer, A. (2019). SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization. In Proceedings of the 2nd Workshop on New Frontiers in Summarization (pp. 70–79). Association for Computational Linguistics.<br>"""
    
#     retriever = Retriever()
#     chunks = retriever._chunk_document_text(document_text)
#     embeded_chunks = retriever._embed_chunks(chunks)
#     retriever._save_embeddings_to_db(embeded_chunks)

    retriever = Retriever()
    retrieval_result = retriever.retrieve("What is the model used in the project?")
    for i, chunk in enumerate(retrieval_result.chunks):
        print(f"Chunk {i+1}:")
        print(chunk.content)
        print("-" * 100)
        print("\n")