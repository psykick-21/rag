"""Ingestions endpoint router."""

from fastapi import APIRouter
from src.db.connection import conn
from typing import List, Dict
from datetime import datetime

router = APIRouter(prefix="/api/v1", tags=["ingestions"])


@router.get("/ingestions")
async def get_ingestions() -> List[Dict]:
    """Get all ingestion records with ingestion_id, timestamp, and number of chunks."""
    
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT ingestion_id, ingested_at, chunks_processed FROM ingestion_metadata ORDER BY ingested_at DESC"
        )
        rows = cursor.fetchall()
        
        ingestions = []
        for row in rows:
            ingestion_id, ingested_at, chunks_processed = row
            ingestions.append({
                "ingestion_id": str(ingestion_id),
                "timestamp": ingested_at.isoformat() if isinstance(ingested_at, datetime) else ingested_at,
                "number_of_chunks": chunks_processed
            })
        
        return ingestions

