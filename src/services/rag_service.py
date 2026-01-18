"""
RAG (Retrieval-Augmented Generation) service for video recommendations.
"""
import os
import json
import pandas as pd
from typing import List, Tuple, Optional
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from sklearn.model_selection import train_test_split

from config.settings import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_EMBEDDING_MODEL,
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    RAG_N_RESULTS,
    RAG_TOP_K,
    TRAIN_TEST_SPLIT_RATIO,
    TRAIN_TEST_RANDOM_STATE,
    YOUTUBE_TITLES_CSV,
    OUTPUT_JSON
)
from src.models.video_recommendation import VideoRecommendation
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGService:
    """Service for RAG-based video recommendations."""
    
    def __init__(self):
        """Initialize RAG service with OpenAI and ChromaDB."""
        self.client: Optional[OpenAI] = None
        self.collection: Optional[chromadb.Collection] = None
        self.data: Optional[pd.DataFrame] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize OpenAI client and ChromaDB collection."""
        try:
            # Initialize OpenAI client
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Initialize embedding function
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name=OPENAI_EMBEDDING_MODEL
            )
            
            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
            
            # Delete existing collection if it exists (for fresh start)
            try:
                chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
                logger.info(f"Deleted existing collection: {CHROMA_COLLECTION_NAME}")
            except Exception:
                pass  # Collection doesn't exist, which is fine
            
            # Create new collection
            self.collection = chroma_client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                embedding_function=openai_ef
            )
            
            # Load data and populate collection
            self._load_and_populate_collection()
            
            self._initialized = True
            logger.info("RAGService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAGService: {e}")
            raise
    
    def _load_and_populate_collection(self) -> None:
        """Load data from JSON and populate ChromaDB collection."""
        try:
            # Load embeddings from JSON if exists
            if OUTPUT_JSON.exists():
                logger.info(f"Loading embeddings from {OUTPUT_JSON}")
                ids, documents, metadata, vectors = self._load_from_json()
                
                if ids and documents:
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadata,
                        embeddings=vectors
                    )
                    logger.info(f"Loaded {len(ids)} embeddings into collection")
                    return
            
            # If JSON doesn't exist, create it from CSV
            logger.info("Creating embeddings from CSV file")
            self._create_embeddings_from_csv()
            
        except Exception as e:
            logger.error(f"Error loading/populating collection: {e}")
            raise
    
    def _load_from_json(self) -> Tuple[List[str], List[str], List[dict], List[List[float]]]:
        """Load embeddings data from JSON file."""
        try:
            with open(OUTPUT_JSON, "r") as f:
                data = json.load(f)
            
            ids = data.get('ids', [])
            documents = data.get('documents', [])
            metadata = data.get('metadata', [])
            vectors = data.get('vectors', [])
            
            return ids, documents, metadata, vectors
            
        except Exception as e:
            logger.error(f"Error loading from JSON: {e}")
            return [], [], [], []
    
    def _create_embeddings_from_csv(self) -> None:
        """Create embeddings from YouTube titles CSV and save to JSON."""
        try:
            if not YOUTUBE_TITLES_CSV.exists():
                raise FileNotFoundError(f"CSV file not found: {YOUTUBE_TITLES_CSV}")
            
            # Load CSV
            self.data = pd.read_csv(YOUTUBE_TITLES_CSV)
            
            # Split into train/test
            train, test = train_test_split(
                self.data,
                test_size=TRAIN_TEST_SPLIT_RATIO,
                random_state=TRAIN_TEST_RANDOM_STATE
            )
            
            # Create embeddings
            ids = [f"doc_{i}" for i in range(len(train))]
            documents = [train['title'].iloc[i] for i in range(len(train))]
            metadatas = [
                {
                    "category": train['category_1'].iloc[i],
                    "Vid_id": train['vid_id'].iloc[i]
                }
                for i in range(len(train))
            ]
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            vectors = []
            for i, title in enumerate(documents):
                try:
                    response = self.client.embeddings.create(
                        input=title,
                        model=OPENAI_EMBEDDING_MODEL,
                    )
                    vectors.append(response.data[0].embedding)
                    if (i + 1) % 10 == 0:
                        logger.info(f"Generated {i + 1}/{len(documents)} embeddings")
                except Exception as e:
                    logger.error(f"Error generating embedding for document {i}: {e}")
                    vectors.append([])
            
            # Save to JSON
            data = {
                "ids": ids,
                "documents": documents,
                "metadata": metadatas,
                "vectors": vectors
            }
            
            with open(OUTPUT_JSON, "w") as f:
                json.dump(data, f, indent=4)
            
            logger.info(f"Saved embeddings to {OUTPUT_JSON}")
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=vectors
            )
            
            logger.info(f"Added {len(ids)} documents to ChromaDB collection")
            
        except Exception as e:
            logger.error(f"Error creating embeddings from CSV: {e}")
            raise
    
    def find_similar_videos(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Find similar videos using vector similarity search.
        
        Args:
            query: Search query text
            
        Returns:
            Tuple of (titles, video_ids)
        """
        if not self._initialized:
            raise RuntimeError("RAGService not initialized. Call initialize() first.")
        
        try:
            # Generate embedding for query
            response = self.client.embeddings.create(
                input=query,
                model=OPENAI_EMBEDDING_MODEL
            )
            query_embedding = response.data[0].embedding
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=RAG_N_RESULTS
            )
            
            titles = results['documents'][0] if results['documents'] else []
            video_ids = [
                meta['Vid_id'] 
                for meta in results['metadatas'][0] 
                if 'Vid_id' in meta
            ] if results['metadatas'] else []
            
            return titles, video_ids
            
        except Exception as e:
            logger.error(f"Error finding similar videos: {e}")
            return [], []
    
    def get_recommendations(self, query: str) -> str:
        """
        Get video recommendations using RAG and GPT.
        
        Args:
            query: User query text
            
        Returns:
            Formatted response with recommendations
        """
        if not self._initialized:
            raise RuntimeError("RAGService not initialized. Call initialize() first.")
        
        try:
            # Find similar videos
            titles, video_ids = self.find_similar_videos(query)
            
            if not titles:
                return "No recommendations found."
            
            # Create prompt for GPT
            system_message = (
                "You estimate which video below is suitable for the keyword that user wants. "
                "Only return video title and Video ID."
            )
            
            user_prompt = "Here are 5 choices that you can choose from:\n"
            for i, (title, vid_id) in enumerate(zip(titles, video_ids), 1):
                user_prompt += f"{i}. Title: {title}\n   Video ID: {vid_id}\n"
            
            user_prompt += "\nAnd now the question for you:\n\n"
            user_prompt += f"Analyze and determine three titles suitable for the keyword from 5 options above: {query}\n"
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "Title is , video id is"}
            ]
            
            # Get response from GPT
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages
            )
            
            result = response.choices[0].message.content
            logger.info(f"Generated recommendations for query: {query}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return f"Error generating recommendations: {str(e)}"
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.client = None
        self.collection = None
        logger.info("RAGService cleaned up")
