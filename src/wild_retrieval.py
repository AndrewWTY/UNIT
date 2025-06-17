import os
import sqlite3
import json
import pickle as pkl
from transformers import GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rank_bm25 import BM25Okapi
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class WildRetrieval:
    def __init__(self, cache_path, batch_size, embed_cache_path, db_path='factcheck_cache/wildhallu.db', retrieval_type="gtr-t5-large"):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.db_path = db_path
        self.retrieval_type = retrieval_type
        self.batch_size = batch_size
        assert retrieval_type == "bm25" or retrieval_type.startswith("gtr-")

        if not os.path.exists(self.db_path):
            data = load_dataset("wentingzhao/WildHallucinations", split="train")
            self._initialize_database()
            self._tokenize_and_store(data)
        else:
            print('Database already exists. Skipping initialization.')

        self.encoder = None
        self.cache_path = cache_path
        self.embed_cache_path = embed_cache_path
        self.cache = {}
        self.embed_cache = {}
        # self.load_cache()
        self.add_n = 0
        self.add_n_embed = 0

    def load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        if os.path.exists(self.embed_cache_path):
            with open(self.embed_cache_path, "rb") as f:
                self.embed_cache = pkl.load(f)
        else:
            self.embed_cache = {}

    def save_cache(self):
        if self.add_n > 0:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    new_cache = json.load(f)
                self.cache.update(new_cache)

            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)

        if self.add_n_embed > 0:
            if os.path.exists(self.embed_cache_path):
                with open(self.embed_cache_path, "rb") as f:
                    new_cache = pkl.load(f)
                self.embed_cache.update(new_cache)

            with open(self.embed_cache_path, "wb") as f:
                pkl.dump(self.embed_cache, f)

    def _initialize_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity TEXT,
                passage TEXT
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity ON passages (entity)')
        conn.commit()
        conn.close()

    def _tokenize_and_store(self, data):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for item in tqdm(data, desc='Tokenizing and storing data'):
            entity = item['entity']
            for webpage in item['info']:
                tokens = self.tokenizer.tokenize(webpage['text'])
                for i in range(0, len(tokens), 256):
                    passage = self.tokenizer.convert_tokens_to_string(tokens[i:i + 256])
                    cursor.execute('INSERT INTO passages (entity, passage) VALUES (?, ?)', (entity, passage))
        conn.commit()
        conn.close()

    def load_encoder(self):
        encoder = SentenceTransformer("sentence-transformers/gtr-t5-large", cache_folder='/'.join(self.cache_path.split('/')[:-1]) if '/' in self.cache_path else '.')
        encoder = encoder.cuda()
        encoder = encoder.eval()
        self.encoder = encoder
        assert self.batch_size is not None

    def get_all_passages(self, entity):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT passage FROM passages WHERE entity = ?', (entity,))
        passages = [row[0] for row in cursor.fetchall()]
        conn.close()

        return passages

    def get_bm25_passages(self, entity, query_sentence, top_k=5):
        passages = self.get_all_passages(entity)
        print("Passages retrieved: ", len(passages))
        if not passages:
            return []
        if entity in self.embed_cache:
            bm25 = self.embed_cache[entity]
        else:
            tokenized_passages = [self.tokenizer.tokenize(p) for p in passages]
            bm25 = BM25Okapi(tokenized_passages)
            self.embed_cache[entity] = bm25
            self.add_n_embed += 1
        tokenized_query = self.tokenizer.tokenize(query_sentence)
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1]

        return [passages[i] for i in top_indices[:top_k]]

    def get_gtr_passages(self, entity, query_sentence, top_k=5):
        passages = self.get_all_passages(entity)
        print("Passages retrieved: ", len(passages))

        if not passages:
            return []

        if self.encoder is None:
            print("Loading encoder...")
            self.load_encoder()

        if entity in self.embed_cache:
            passage_embeddings = self.embed_cache[entity]
        else:
            passage_embeddings = self.encoder.encode(passages, device=self.encoder.device, batch_size=self.batch_size)
            self.embed_cache[entity] = passage_embeddings
            self.add_n_embed += 1

        query_embedding = self.encoder.encode([query_sentence], device=self.encoder.device, batch_size=self.batch_size)[0]

        scores = np.inner(query_embedding, passage_embeddings)

        top_indices = np.argsort(scores)[::-1]

        return [passages[i] for i in top_indices[:top_k]]

    def get_passages(self, topic, question, k):
        retrieval_query = topic + " " + question.strip()
        cache_key = topic + "#" + retrieval_query

        if cache_key not in self.cache:
            if self.retrieval_type == "bm25":
                self.cache[cache_key] = self.get_bm25_passages(topic, retrieval_query, k)
            else:
                self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, k)
            self.add_n += 1

        return self.cache[cache_key]


# Example usage:
if __name__ == '__main__':
    cache_path = 'factcheck_cache/retrieval-wildhalu-bm25.json'
    embed_cache_path = 'factcheck_cache/retrieval-wildhalu-bm25.pkl'
    retriever = WildRetrieval(batch_size=256, cache_path=cache_path, embed_cache_path=embed_cache_path)

    entity = 'Marta Batmasian'
    top_k = 2

    query_sentences = ['Who is Marta Batmasian']

    print("\nGTR Relevant Passages:")
    for query_sentence in query_sentences:
        relevant_passages_gtr = retriever.get_passages(entity, query_sentence, top_k)
        for passage in relevant_passages_gtr:
            print(passage)
            print("%" * 50)