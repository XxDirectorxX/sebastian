from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class QuantumState(Base):
    __tablename__ = "quantum_states"
    
    id = Column(Integer, primary_key=True, index=True)
    field_id = Column(String, unique=True, index=True)
    cache_offset = Column(Integer)
    field_strength = Column(Float)
    reality_coherence = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class FieldMetrics(Base):
    __tablename__ = "field_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    field_id = Column(String, index=True)
    coherence_level = Column(Float)
    reality_alignment = Column(Float)
    stability_factor = Column(Float)
    quantum_stability = Column(Float)
    field_uniformity = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
