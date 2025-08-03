# Geographic Information RAG with Spatial Queries

A Retrieval-Augmented Generation (RAG) system that combines geographic data, satellite imagery, and location-based information to answer spatial queries and provide location-specific insights.

## Features
- Geographic data processing and spatial indexing
- Satellite imagery analysis and interpretation
- Location-based information retrieval
- Spatial query processing and optimization
- Multi-scale geographic analysis capabilities

## Tech Stack
- Streamlit (UI)
- ChromaDB (Vector DB)
- HuggingFace Sentence Transformers (Embeddings)
- GeoPandas, Shapely, rasterio (Geospatial)
- OpenAI or HuggingFace LLM (Generation)

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```

## Deployment
### Streamlit Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io/ and deploy a new app from your repo.
3. Set your OpenAI API key in the app secrets as `OPENAI_API_KEY`.

### HuggingFace Spaces
1. Create a new Space (Streamlit).
2. Upload your code and requirements.txt.
3. Set your OpenAI API key as a secret.
4. Deploy!

### Local
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...  # or set in your .env
streamlit run app.py
```

## Sample Queries
- What cities are in south India?
- Describe the region near 12.97, 77.59
- Find polygons near Mumbai
- What features are within 10km of Bangalore?

## Project Structure
```
GEO/
├── app.py                # Streamlit app
├── rag/
│   ├── __init__.py
│   ├── geo_retriever.py  # Handles spatial queries, vector search, RAG
│   ├── image_analyzer.py # Satellite image feature extraction
│   └── utils.py
├── data/
│   ├── geo/              # Sample geographic datasets
│   └── images/           # Sample satellite images
├── requirements.txt
└── README.md
```

## License
MIT