"""
STEP 1: Dataset Generation
Synthetic data generation for supply chain hypergraph:
- Nodes (suppliers/components)
- Hyperedges (subassemblies)
- Incidence relationships
- Multi-echelon dependencies
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

class SupplyChainDataGenerator:
    """Generate synthetic supply chain data with realistic distributions"""
    
    def __init__(self, 
                 n_suppliers: int = 150,
                 n_assemblies: int = 80,
                 seed: int = 42):
        """
        Initialize data generator
        
        Args:
            n_suppliers: Number of nodes (suppliers/components)
            n_assemblies: Number of hyperedges (subassemblies)
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.n_suppliers = n_suppliers
        self.n_assemblies = n_assemblies
        self.suppliers = {}
        self.assemblies = {}
        self.incidence = []
        self.echelon_deps = []
        
    def generate_nodes(self) -> pd.DataFrame:
        """
        Generate supplier/component nodes with realistic properties
        
        Node features:
        - node_id: Unique identifier
        - node_type: 'supplier' or 'component'
        - tier: Supply chain tier (1-3, lower = closer to raw materials)
        - lead_time: Days to deliver/produce (exponential distribution)
        - reliability: Probability of successful delivery [0,1] (Beta dist)
        - substitutability: How easily replaceable [0,1] (Uniform)
        - cost: Relative cost (Pareto distribution)
        """
        nodes_data = []
        
        for i in range(self.n_suppliers):
            node_id = f"S{i:04d}"
            
            # Mix of supplier types
            node_type = "supplier" if np.random.random() < 0.7 else "component"
            
            # Tier distribution (most at tier 2-3)
            tier = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
            
            # Lead time: exponential (most quick, some slow)
            lead_time = int(np.random.exponential(scale=15)) + 3
            
            # Reliability: Beta(8, 2) skewed towards high reliability
            reliability = np.random.beta(a=8, b=2)
            
            # Substitutability: uniform [0.1, 1.0]
            substitutability = np.random.uniform(0.1, 1.0)
            
            # Cost: Pareto distribution (few expensive, many cheap)
            cost = np.random.pareto(a=1.5) + 1.0
            
            nodes_data.append({
                'node_id': node_id,
                'node_type': node_type,
                'tier': tier,
                'lead_time': lead_time,
                'reliability': reliability,
                'substitutability': substitutability,
                'cost': cost
            })
            
            self.suppliers[node_id] = nodes_data[-1]
        
        return pd.DataFrame(nodes_data)
    
    def generate_hyperedges(self) -> pd.DataFrame:
        """
        Generate subassemblies (hyperedges) with engineering properties
        
        Hyperedge features:
        - hyperedge_id: Unique identifier
        - bom_weight: Bill of Materials importance [0,1] (Pareto)
        - tolerance: Engineering tolerance sensitivity [0.2, 1.0]
        - critical_path: Is on critical path to final product [0,1]
        - tier_level: Assembly tier (1=raw materials, N=final assembly)
        """
        hyperedges_data = []
        
        # Determine tier distribution (multi-echelon)
        tier_levels = np.random.choice([1, 2, 3, 4, 5], 
                                      size=self.n_assemblies, 
                                      p=[0.25, 0.3, 0.25, 0.15, 0.05])
        
        for i in range(self.n_assemblies):
            hyperedge_id = f"H{i:04d}"
            
            # BOM weight: Pareto (few critical, many non-critical)
            bom_weight = np.random.pareto(a=1.5) / 10 + 0.1
            bom_weight = min(bom_weight, 1.0)
            
            # Tolerance: uniform [0.2, 1.0]
            tolerance = np.random.uniform(0.2, 1.0)
            
            # Critical path: higher for tier 1 (raw materials)
            tier = tier_levels[i]
            critical_prob = 0.8 if tier == 1 else (0.5 if tier == 2 else 0.2)
            critical_path = 1.0 if np.random.random() < critical_prob else 0.0
            
            hyperedges_data.append({
                'hyperedge_id': hyperedge_id,
                'bom_weight': bom_weight,
                'tolerance': tolerance,
                'critical_path': critical_path,
                'tier_level': tier
            })
            
            self.assemblies[hyperedge_id] = hyperedges_data[-1]
        
        return pd.DataFrame(hyperedges_data)
    
    def generate_incidence(self) -> pd.DataFrame:
        """
        Generate incidence matrix (hyperedge-node relationships)
        
        This is the CORE of the hypergraph:
        - Each hyperedge connects 2-6 suppliers (logical AND)
        - Simulates real subassembly structure
        """
        incidence_data = []
        
        for hyperedge_id in self.assemblies.keys():
            # Suppliers per assembly: between 2-6 (realistic for subassemblies)
            n_suppliers_in_assembly = np.random.randint(2, 7)
            
            # Random suppliers for this assembly
            suppliers_in_assembly = np.random.choice(
                list(self.suppliers.keys()),
                size=n_suppliers_in_assembly,
                replace=False
            )
            
            for supplier_id in suppliers_in_assembly:
                incidence_data.append({
                    'hyperedge_id': hyperedge_id,
                    'node_id': supplier_id
                })
        
        return pd.DataFrame(incidence_data)
    
    def generate_echelon_dependencies(self) -> pd.DataFrame:
        """
        Generate multi-echelon dependencies
        
        Models how subassemblies depend on each other:
        - Higher tier depends on lower tier subassemblies
        - Creates assembly hierarchy
        """
        dependencies = []
        
        # Group hyperedges by tier level
        assemblies_by_tier = {}
        for hid, props in self.assemblies.items():
            tier = props['tier_level']
            if tier not in assemblies_by_tier:
                assemblies_by_tier[tier] = []
            assemblies_by_tier[tier].append(hid)
        
        # Create tier-to-tier dependencies
        tiers = sorted(assemblies_by_tier.keys())
        
        for i in range(len(tiers) - 1):
            parent_tier = tiers[i + 1]
            child_tier = tiers[i]
            
            parent_assemblies = assemblies_by_tier[parent_tier]
            child_assemblies = assemblies_by_tier[child_tier]
            
            # Each parent depends on 1-3 children
            for parent_id in parent_assemblies:
                n_children = np.random.randint(1, 4)
                children = np.random.choice(child_assemblies, 
                                           size=min(n_children, len(child_assemblies)),
                                           replace=False)
                
                for child_id in children:
                    dependencies.append({
                        'parent_hyperedge': parent_id,
                        'child_hyperedge': child_id
                    })
        
        return pd.DataFrame(dependencies)
    
    def generate_all(self) -> dict:
        """
        Generate complete synthetic dataset
        
        Returns:
            Dictionary with all tables:
            - nodes: supplier/component features
            - hyperedges: subassembly features
            - incidence: hyperedge-node relationships (CORE)
            - echelon_dependencies: multi-tier structure
        """
        print("Generating supply chain dataset...")
        
        nodes = self.generate_nodes()
        hyperedges = self.generate_hyperedges()
        incidence = self.generate_incidence()
        echelon_deps = self.generate_echelon_dependencies()
        
        print(f"  ✓ Nodes: {len(nodes)}")
        print(f"  ✓ Hyperedges: {len(hyperedges)}")
        print(f"  ✓ Incidence relationships: {len(incidence)}")
        print(f"  ✓ Echelon dependencies: {len(echelon_deps)}")
        
        return {
            'nodes': nodes,
            'hyperedges': hyperedges,
            'incidence': incidence,
            'echelon_dependencies': echelon_deps
        }


def load_real_datasets(data_path: str) -> dict:
    """
    Load and extract features from real supply chain datasets
    
    Integrates actual data with synthetic structure
    """
    data_files = {
        'bom': f"{data_path}/BOM/train_set.csv",
        'dataco': f"{data_path}/DataCo/DataCoSupplyChainDataset.csv",
        'maintenance': f"{data_path}/Maintenance/ai4i2020.csv"
    }
    
    datasets = {}
    
    # Load BOM data
    print("Loading BOM data...")
    bom = pd.read_csv(data_files['bom'])
    print(f"  ✓ BOM: {bom.shape}")
    datasets['bom'] = bom
    
    # Load DataCo data
    print("Loading DataCo data...")
    try:
        dataco = pd.read_csv(data_files['dataco'], encoding='latin-1')
        print(f"  ✓ DataCo: {dataco.shape}")
        datasets['dataco'] = dataco
    except Exception as e:
        print(f"  ✗ DataCo loading error: {e}")
    
    # Load Maintenance data
    print("Loading Maintenance data...")
    maintenance = pd.read_csv(data_files['maintenance'])
    print(f"  ✓ Maintenance: {maintenance.shape}")
    datasets['maintenance'] = maintenance
    
    return datasets


def save_datasets(datasets: dict, output_dir: str):
    """Save all generated datasets to CSV files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, df in datasets.items():
        filepath = output_path / f"{name}.csv"
        df.to_csv(filepath, index=False)
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    # Generate synthetic dataset
    generator = SupplyChainDataGenerator(
        n_suppliers=150,
        n_assemblies=80,
        seed=42
    )
    
    synthetic_data = generator.generate_all()
    
    # Save synthetic data
    output_dir = "outputs/datasets"
    save_datasets(synthetic_data, output_dir)
    
    # Load real datasets
    real_data = load_real_datasets("Data set")
    
    print("\n✓ Dataset generation complete!")
