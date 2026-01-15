import json
import os
from collections import deque

FILE_PATH = 'location_hierarchy.json'

class HierarchyEditor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {}
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                self.data = json.load(f)
            print(f"Loaded {len(self.data)} locations from {self.filepath}")
        else:
            print(f"File {self.filepath} not found. Starting with empty data.")
            self.data = {}

    def save(self):
        # Sort keys for cleaner JSON
        sorted_data = {k: self.data[k] for k in sorted(self.data)}
        with open(self.filepath, 'w') as f:
            json.dump(sorted_data, f, indent=4)
        print(f"Saved changes to {self.filepath}")

    def ensure_loc(self, loc):
        """Internal helper to create a key if it doesn't exist."""
        if loc not in self.data:
            print(f"Creating new entry for '{loc}'")
            self.data[loc] = {"parents": [], "siblings": [], "grandparents": []}

    # --- NEW FUNCTIONALITY HERE ---
    def add_location(self, loc, parents=None, siblings=None, grandparents=None):
        """
        Explicitly creates a new location and sets up its initial connections.
        Safe to call if location already exists (it will just add the new connections).
        """
        self.ensure_loc(loc)
        
        if parents:
            for p in parents:
                self.add_parent(loc, p)
        
        if siblings:
            for s in siblings:
                self.add_sibling(loc, s)
                
        if grandparents:
            for g in grandparents:
                self.add_grandparent(loc, g)

    def delete_location(self, loc):
        """Completely removes a location and cleans up references to it in other nodes."""
        if loc not in self.data:
            print(f"Location '{loc}' not found, nothing to delete.")
            return

        # 1. Remove this loc from everyone else's lists
        print(f"Removing references to '{loc}' from other nodes...")
        for other_loc, relations in self.data.items():
            if loc in relations.get('parents', []):
                relations['parents'].remove(loc)
            if loc in relations.get('siblings', []):
                relations['siblings'].remove(loc)
            if loc in relations.get('grandparents', []):
                relations['grandparents'].remove(loc)
        
        # 2. Delete the node itself
        del self.data[loc]
        print(f"Deleted '{loc}' from database.")

    # -----------------------------

    def add_sibling(self, loc_a, loc_b):
        """Adds a sibling relationship (Bidirectional)"""
        self.ensure_loc(loc_a)
        self.ensure_loc(loc_b)
        
        if loc_b not in self.data[loc_a]['siblings']:
            self.data[loc_a]['siblings'].append(loc_b)
        if loc_a not in self.data[loc_b]['siblings']:
            self.data[loc_b]['siblings'].append(loc_a)
        
        print(f"Ensured SIBLING: {loc_a} <-> {loc_b}")

    def add_parent(self, child, parent):
        """Adds a parent relationship (Directional)"""
        self.ensure_loc(child)
        self.ensure_loc(parent)
        
        if parent not in self.data[child]['parents']:
            self.data[child]['parents'].append(parent)
            print(f"Added PARENT: {child} -> {parent}")
        else:
            print(f"Relation already exists: {child} -> {parent}")

    def add_grandparent(self, child, g_parent):
        """Adds a grandparent relationship (Directional)"""
        self.ensure_loc(child)
        self.ensure_loc(g_parent)
        
        if g_parent not in self.data[child]['grandparents']:
            self.data[child]['grandparents'].append(g_parent)
            print(f"Added GRANDPARENT: {child} -> {g_parent}")

    def remove_relation(self, loc_a, loc_b):
        """Removes ANY direct link between A and B (Parent, Sibling, or Grandparent)"""
        found = False
        
        def clean(source, target):
            changed = False
            for key in ['parents', 'siblings', 'grandparents']:
                if key in self.data[source] and target in self.data[source][key]:
                    self.data[source][key].remove(target)
                    changed = True
            return changed

        if loc_a in self.data and clean(loc_a, loc_b): found = True
        if loc_b in self.data and clean(loc_b, loc_a): found = True
        
        if found:
            print(f"Removed connection between '{loc_a}' and '{loc_b}'")
        else:
            print(f"No direct connection found between '{loc_a}' and '{loc_b}'")

    def check_neighbors(self, loc):
        if loc not in self.data:
            print(f"Location '{loc}' not found.")
            return
        
        d = self.data[loc]
        print(f"\n--- {loc} ---")
        print(f"  Parents:      {d.get('parents', [])}")
        print(f"  Siblings:     {d.get('siblings', [])}")
        print(f"  Grandparents: {d.get('grandparents', [])}")


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    editor = HierarchyEditor(FILE_PATH)

    # 1. ADD NEW LOCATION ("Southeastern United States")
    # You can now define everything in one go:
    editor.add_location(
        "Western United States",
        parents=["Western US", "Western CONUS"],
        siblings=["The Southwest", "The Northwest", "Northwestern US", "Southwestern US", "Southwestern United States", "Northwestern United States"]
    )
    #editor.add_location(
    #    "Rwanda"
    #)

    # 2. CHECK YOUR WORK
    editor.check_neighbors("Southeastern United States")

    # 3. SAVE CHANGES 
    editor.save()