#!/usr/bin/env python3
"""
Structural Intelligence using YOUR .asf_core2.py PersistentCatalog
"""
import sys
sys.path.append('/Volumes/StagbrookField/stagbrook_field')
from asf_core2 import PersistentCatalog, parse, key
import hashlib
import time

class BruceStructuralIntelligence:
    def __init__(self, catalog_path: str = "bruce_si.db"):
        self.catalog = PersistentCatalog(catalog_path)
        self.owner_id = "bruce_stagbrook"
        
    def record_satisfaction(self, nl_request: str, shape_dyck: str, satisfaction: float):
        """Record satisfaction using YOUR catalog label system"""
        shape = parse(shape_dyck) if shape_dyck.startswith(('A', 'S', '(')) else None
        if not shape:
            return
            
        # Use YOUR label system
        confidence = satisfaction
        notes = f"satisfaction:{satisfaction:.2f} at {time.time()}"
        self.catalog.set_label(shape, nl_request, namespace="bruce_requests", 
                              confidence=confidence, notes=notes)
        
        # Also label satisfaction level
        if satisfaction >= 0.9:
            self.catalog.set_label(shape, "gold", namespace="bruce_ratings", 
                                  confidence=satisfaction)
        elif satisfaction >= 0.7:
            self.catalog.set_label(shape, "good", namespace="bruce_ratings", 
                                  confidence=satisfaction)
    
    def predict_satisfaction(self, nl_request: str) -> float:
        """Predict satisfaction using YOUR find_by_label"""
        shape_keys = self.catalog.find_by_label(nl_request, namespace="bruce_requests")
        if not shape_keys:
            return 0.5
            
        # Get the highest confidence match
        best_key = max(shape_keys, key=lambda x: x[1])  # x[1] is confidence
        return best_key[1]  # Return confidence as satisfaction prediction
    
    def get_gold_exemplars(self) -> list:
        """Get YOUR gold-rated shapes"""
        gold_keys = self.catalog.find_by_label("gold", namespace="bruce_ratings")
        exemplars = []
        for shape_key, confidence in gold_keys:
            entry = self.catalog.get(parse("A"))  # Dummy to get entry structure
            if entry:
                exemplars.append({
                    'shape_key': shape_key,
                    'satisfaction': confidence,
                    'access_count': entry.access_count
                })
        return exemplars
    
    def close(self):
        self.catalog.close()

if __name__ == "__main__":
    si = BruceStructuralIntelligence()
    
    # Test with dummy data
    si.record_satisfaction("double numbers", "A", 0.95)
    pred = si.predict_satisfaction("double numbers")
    print(f"Predicted satisfaction: {pred:.0%}")
    
    gold = si.get_gold_exemplars()
    print(f"Gold exemplars: {len(gold)}")
    
    si.close()