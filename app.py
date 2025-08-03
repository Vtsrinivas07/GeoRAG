import streamlit as st
from rag.geo_retriever import GeoRetriever
from rag.image_analyzer import ImageAnalyzer
import os
import time

st.set_page_config(page_title="Geographic RAG Demo", layout="wide")

st.sidebar.title("Geographic RAG System")
st.sidebar.markdown("Upload data, select query type, and ask spatial questions.")

st.title("Geographic Information RAG with Spatial Queries")

geo = GeoRetriever()
img_analyzer = ImageAnalyzer()

# Tabs: Spatial Query, Map View, Image Analysis, Semantic Search, RAG Answer, Evaluation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Spatial Query", "Map View", "Image Analysis", "Semantic Search", "RAG Answer", "Evaluation"])

with tab1:
    st.header("Ask a Spatial Question")
    query = st.text_input("Enter your question:")
    lat = st.number_input("Latitude", value=12.9716, format="%f")
    lon = st.number_input("Longitude", value=77.5946, format="%f")
    radius = st.slider("Radius (km)", 1, 50, 5)
    if st.button("Submit"):
        result = geo.query(query, location=(lat, lon), radius_km=radius)
        st.success(result)
    else:
        st.info("Results will appear here.")

with tab2:
    st.header("Map Visualization")
    try:
        import geopandas as gpd
        import folium
        from streamlit_folium import st_folium
        if not geo.gdf.empty:
            m = folium.Map(location=[12.9716, 77.5946], zoom_start=5)
            folium.GeoJson(geo.gdf).add_to(m)
            st_folium(m, width=700, height=500)
        else:
            st.info("No geographic data loaded.")
    except ImportError:
        st.map()  # Fallback
        st.info("Map and spatial data will be shown here.")

with tab3:
    st.header("Satellite Image Analysis")
    image_files = [f for f in os.listdir('data/images') if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    if image_files:
        selected_image = st.selectbox("Select an image to analyze:", image_files)
        img_path = os.path.join('data/images', selected_image)
        st.image(img_path, caption=selected_image, use_column_width=True)
        if st.button("Analyze Image"):
            result = img_analyzer.analyze(img_path)
            st.success(result)
    else:
        st.info("No satellite images found in data/images/. Please add a sample image.")

with tab4:
    st.header("Semantic Search over Geographic Data")
    sem_query = st.text_input("Enter a semantic query:", key="sem_query")
    top_k = st.slider("Top K Results", 1, 5, 3)
    if st.button("Semantic Search"):
        results = geo.semantic_search(sem_query, top_k=top_k)
        if results:
            st.write("Top relevant features:")
            for i, res in enumerate(results, 1):
                st.write(f"{i}. {res}")
        else:
            st.info("No relevant features found.")

with tab5:
    st.header("RAG Answer (LLM-augmented)")
    rag_query = st.text_input("Enter your question for RAG:", key="rag_query")
    rag_top_k = st.slider("Top K Context Features", 1, 5, 3, key="rag_top_k")
    openai_api_key = st.text_input("OpenAI API Key (optional)", type="password")
    if st.button("Get RAG Answer"):
        answer = geo.rag_answer(rag_query, top_k=rag_top_k, openai_api_key=openai_api_key or None)
        st.success(answer)
    else:
        st.info("Enter a question and click 'Get RAG Answer' to see the LLM-augmented response.")

with tab6:
    st.header("Evaluation: Latency & Transparency")
    eval_query = st.text_input("Enter a query to evaluate:", key="eval_query")
    eval_top_k = st.slider("Top K Context Features", 1, 5, 3, key="eval_top_k")
    eval_mode = st.radio("Evaluation Mode", ["Semantic Search", "RAG Answer"])
    openai_api_key_eval = st.text_input("OpenAI API Key (optional, for RAG)", type="password", key="eval_api")
    if st.button("Run Evaluation"):
        start = time.time()
        if eval_mode == "Semantic Search":
            results = geo.semantic_search(eval_query, top_k=eval_top_k)
            latency = time.time() - start
            st.write(f"Semantic Search Latency: {latency:.3f} seconds")
            st.write("Retrieved Context:")
            for i, res in enumerate(results, 1):
                st.write(f"{i}. {res}")
        else:
            answer = geo.rag_answer(eval_query, top_k=eval_top_k, openai_api_key=openai_api_key_eval or None)
            latency = time.time() - start
            st.write(f"RAG Answer Latency: {latency:.3f} seconds")
            st.write("Generated Answer:")
            st.success(answer)
            st.write("Retrieved Context:")
            context = geo.format_context(geo.semantic_search(eval_query, top_k=eval_top_k))
            st.code(context)
    else:
        st.info("Enter a query and click 'Run Evaluation' to measure latency and see context.")