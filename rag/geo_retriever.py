# rag/geo_retriever.py
import geopandas as gpd
from shapely.geometry import Point
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import openai

class GeoRetriever:
    def __init__(self, geojson_path="data/geo/sample.geojson"):
        if os.path.exists(geojson_path):
            self.gdf = gpd.read_file(geojson_path)
        else:
            self.gdf = gpd.GeoDataFrame()
        if not self.gdf.empty:
            self.gdf["geometry"] = self.gdf["geometry"].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
            self.gdf = self.gdf.set_crs(epsg=4326)
        else:
            print("No geographic data loaded.")
        # Embedding model and ChromaDB setup
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection("geo_features")
        self._index_features()

    def _index_features(self):
        if self.gdf.empty:
            return
        docs = []
        metadatas = []
        ids = []
        for idx, row in self.gdf.iterrows():
            desc = f"{row.get('name', '')} ({row.get('type', '')}) at {row.geometry.wkt}"
            docs.append(desc)
            metadatas.append({"name": row.get("name", ""), "type": row.get("type", ""), "wkt": row.geometry.wkt})
            ids.append(str(idx))
        embeddings = self.model.encode(docs).tolist()
        self.collection.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def query(self, question, location=None, radius_km=5):
        # If location is provided, find features within radius
        if location and not self.gdf.empty:
            lat, lon = location
            pt = Point(lon, lat)
            buffer_deg = radius_km / 111  # 1 deg ~ 111km
            nearby = self.gdf[self.gdf.distance(pt) <= buffer_deg]
            if not nearby.empty:
                return f"Found {len(nearby)} features near {location}:\n" + str(nearby[["name", "type"]])
            else:
                return f"No features found within {radius_km}km of {location}."
        return "No location provided or no data loaded."

    def semantic_search(self, query, top_k=3):
        query_emb = self.model.encode([query]).tolist()[0]
        results = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
        hits = results['documents'][0] if results['documents'] else []
        return hits

    def format_context(self, features):
        # Format features as context for LLM
        return "\n".join(features)

    def rag_answer(self, user_query, top_k=3, openai_api_key=None):
        features = self.semantic_search(user_query, top_k=top_k)
        context = self.format_context(features)
        prompt = f"You are a geographic information assistant. Use the following context to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return "OpenAI API key not set. Please set OPENAI_API_KEY environment variable."
        openai.api_key = openai_api_key
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error from OpenAI API: {e}"