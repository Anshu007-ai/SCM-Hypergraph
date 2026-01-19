"""
Real Data Loader Module
Loads and processes real datasets from the Data set/ folder
Integrates BOM, DataCo Supply Chain, and Predictive Maintenance data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List


class RealDataLoader:
    """
    Load and process real supply chain datasets.
    
    Datasets:
    - BOM (Bill of Materials): Manufacturing component relationships
    - DataCo: Supply chain logistics and transactions
    - Maintenance: Predictive maintenance and failure data
    """
    
    def __init__(self, data_dir: str = "Data set"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to Data set folder (default: "Data set")
        """
        self.data_dir = Path(data_dir).resolve()  # Convert to absolute path
        self.bom_dir = self.data_dir / "BOM"
        self.dataco_dir = self.data_dir / "DataCo"
        self.maintenance_dir = self.data_dir / "Maintenance"
        
        # Storage for loaded data
        self.bom_train = None
        self.bom_test = None
        self.dataco = None
        self.maintenance = None
        
    def load_bom_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load Bill of Materials data (train and test sets).
        
        Returns:
            Tuple of (train_df, test_df)
        """
        train_path = self.bom_dir / "train_set.csv"
        test_path = self.bom_dir / "test_set.csv"
        
        self.bom_train = pd.read_csv(train_path)
        self.bom_test = pd.read_csv(test_path)
        
        print(f"✓ BOM Train: {self.bom_train.shape[0]} rows × {self.bom_train.shape[1]} columns")
        print(f"  Columns: {list(self.bom_train.columns)}")
        print(f"✓ BOM Test: {self.bom_test.shape[0]} rows × {self.bom_test.shape[1]} columns")
        
        return self.bom_train, self.bom_test
    
    def load_dataco_data(self) -> pd.DataFrame:
        """
        Load DataCo Supply Chain dataset.
        WARNING: File is 180MB+ so we'll use chunks if needed.
        
        Returns:
            DataFrame with supply chain data
        """
        dataco_path = self.dataco_dir / "DataCoSupplyChainDataset.csv"
        
        # Load with limited columns to manage memory
        print("Loading DataCo dataset (this may take a moment)...")
        self.dataco = pd.read_csv(dataco_path)
        
        print(f"✓ DataCo: {self.dataco.shape[0]} rows × {self.dataco.shape[1]} columns")
        print(f"  Columns: {list(self.dataco.columns)[:10]}... ({len(self.dataco.columns)} total)")
        
        return self.dataco
    
    def load_maintenance_data(self) -> pd.DataFrame:
        """
        Load Predictive Maintenance AI4I 2020 dataset.
        
        Returns:
            DataFrame with maintenance and failure data
        """
        maintenance_path = self.maintenance_dir / "ai4i2020.csv"
        
        self.maintenance = pd.read_csv(maintenance_path)
        
        print(f"✓ Maintenance: {self.maintenance.shape[0]} rows × {self.maintenance.shape[1]} columns")
        print(f"  Columns: {list(self.maintenance.columns)}")
        
        return self.maintenance
    
    def extract_supplier_nodes(self) -> pd.DataFrame:
        """
        Extract supplier/vendor nodes from BOM data.
        
        Returns:
            DataFrame with supplier information:
            - node_id: Unique supplier ID
            - node_type: "supplier"
            - cost: Average invoice amount
            - reliability: Derived from failure patterns (placeholder: 0.8)
            - lead_time: Days to deliver (placeholder: 14)
            - substitutability: Can be replaced? (placeholder: 0.7)
            - tier: Supply chain tier (placeholder: 1)
        """
        if self.bom_train is None:
            self.load_bom_data()
        
        df = self.bom_train.copy()
        
        # Extract unique vendors/suppliers
        suppliers = df.groupby('Vendor_Code').agg({
            'Inv_Amt': ['count', 'mean', 'std', 'min', 'max'],
            'GL_Code': 'nunique',
            'Product_Category': 'nunique'
        }).reset_index()
        
        suppliers.columns = ['vendor_code', 'invoice_count', 'avg_cost', 'std_cost', 'min_cost', 'max_cost', 'gl_codes', 'categories']
        suppliers['node_id'] = 'SUPPLIER_' + suppliers.index.astype(str)
        suppliers['node_type'] = 'supplier'
        suppliers['tier'] = 1  # BOM suppliers are tier 1
        
        # Reliability: Higher invoice count = more reliable (normalized)
        max_invoices = suppliers['invoice_count'].max()
        suppliers['reliability'] = 0.6 + 0.35 * (suppliers['invoice_count'] / max_invoices)
        
        # Lead time: Inverse of invoice frequency (more frequent = shorter lead time)
        suppliers['lead_time'] = 30 - (20 * suppliers['invoice_count'] / max_invoices)
        suppliers['lead_time'] = suppliers['lead_time'].clip(lower=5)
        
        # Substitutability: Based on how many GL codes they supply
        suppliers['substitutability'] = suppliers['gl_codes'] / suppliers['gl_codes'].max()
        
        # Cost: Average invoice amount
        suppliers['cost'] = suppliers['avg_cost']
        
        # Select relevant columns
        nodes_df = suppliers[[
            'node_id', 'node_type', 'tier', 'lead_time', 
            'reliability', 'substitutability', 'cost', 'vendor_code'
        ]].copy()
        
        print(f"\n✓ Extracted {len(nodes_df)} supplier nodes from BOM data")
        print(f"  Reliability range: [{nodes_df['reliability'].min():.3f}, {nodes_df['reliability'].max():.3f}]")
        print(f"  Lead time range: [{nodes_df['lead_time'].min():.1f}, {nodes_df['lead_time'].max():.1f}] days")
        print(f"  Cost range: ${nodes_df['cost'].min():.2f} - ${nodes_df['cost'].max():.2f}")
        
        return nodes_df
    
    def extract_product_categories_as_hyperedges(self) -> pd.DataFrame:
        """
        Extract product categories as hyperedges (assemblies).
        Each category represents a product that uses multiple suppliers.
        
        Returns:
            DataFrame with hyperedge information:
            - hyperedge_id: Unique product category ID
            - bom_weight: Importance weight (based on total cost)
            - tolerance: Quality tolerance (placeholder)
            - critical_path: Is in critical path? (placeholder)
            - tier_level: Supply tier level
        """
        if self.bom_train is None:
            self.load_bom_data()
        
        df = self.bom_train.copy()
        
        # Extract unique product categories
        products = df.groupby('Product_Category').agg({
            'Inv_Amt': ['sum', 'count', 'mean'],
            'Vendor_Code': 'nunique',
            'GL_Code': 'nunique'
        }).reset_index()
        
        products.columns = ['product_category', 'total_cost', 'invoice_count', 'avg_cost', 'supplier_count', 'gl_codes']
        products['hyperedge_id'] = 'ASSEMBLY_' + products.index.astype(str)
        
        # BOM weight: Normalized total cost
        max_cost = products['total_cost'].max()
        products['bom_weight'] = products['total_cost'] / max_cost
        
        # Tolerance: Inverse of supplier count (few suppliers = tighter tolerance)
        products['tolerance'] = products['supplier_count'] / products['supplier_count'].max()
        
        # Critical path: Products with high cost are more critical
        products['critical_path'] = products['total_cost'] > products['total_cost'].quantile(0.75)
        products['critical_path'] = products['critical_path'].astype(int)
        
        # Tier level: Based on supplier diversity
        products['tier_level'] = 1  # BOM is tier 1
        
        # Select relevant columns
        hyperedges_df = products[[
            'hyperedge_id', 'bom_weight', 'tolerance', 'critical_path', 'tier_level', 'product_category'
        ]].copy()
        
        print(f"\n✓ Extracted {len(hyperedges_df)} product categories as hyperedges")
        print(f"  BOM weight range: [{hyperedges_df['bom_weight'].min():.3f}, {hyperedges_df['bom_weight'].max():.3f}]")
        print(f"  Critical products: {hyperedges_df['critical_path'].sum()}")
        
        return hyperedges_df
    
    def extract_incidence_matrix_data(self) -> pd.DataFrame:
        """
        Extract incidence relationships (supplier -> product category).
        
        Returns:
            DataFrame with columns: hyperedge_id, node_id
        """
        if self.bom_train is None:
            self.load_bom_data()
        
        df = self.bom_train.copy()
        
        # Create mapping: vendor -> product category
        incidence = df[['Vendor_Code', 'Product_Category']].drop_duplicates()
        
        # Map to our IDs
        vendor_mapping = self.extract_supplier_nodes()[['vendor_code', 'node_id']].set_index('vendor_code')['node_id'].to_dict()
        product_mapping = self.extract_product_categories_as_hyperedges()[['product_category', 'hyperedge_id']].set_index('product_category')['hyperedge_id'].to_dict()
        
        incidence['node_id'] = incidence['Vendor_Code'].map(vendor_mapping)
        incidence['hyperedge_id'] = incidence['Product_Category'].map(product_mapping)
        
        incidence_df = incidence[['hyperedge_id', 'node_id']].drop_duplicates()
        
        print(f"\n✓ Extracted {len(incidence_df)} incidence relationships")
        
        return incidence_df
    
    def extract_reliability_from_maintenance(self) -> Dict[str, float]:
        """
        Extract equipment reliability scores from maintenance data.
        
        Returns:
            Dictionary mapping product types to reliability scores
        """
        if self.maintenance is None:
            self.load_maintenance_data()
        
        df = self.maintenance.copy()
        
        # Calculate failure rate by product type
        reliability = {}
        
        if 'Type' in df.columns:
            type_stats = df.groupby('Type').agg({
                'Machine failure': ['sum', 'count']
            }).reset_index()
            
            type_stats.columns = ['Type', 'failures', 'total']
            type_stats['reliability'] = 1 - (type_stats['failures'] / type_stats['total'])
            
            reliability = dict(zip(type_stats['Type'], type_stats['reliability']))
            
            print(f"\n✓ Extracted reliability from {len(reliability)} equipment types")
            for eq_type, rel in reliability.items():
                print(f"  {eq_type}: {rel:.3f}")
        
        return reliability
    
    def get_supply_chain_statistics(self) -> Dict:
        """
        Get high-level statistics about the supply chain.
        
        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'bom_suppliers': len(self.extract_supplier_nodes()) if self.bom_train is not None else 0,
            'bom_products': len(self.extract_product_categories_as_hyperedges()) if self.bom_train is not None else 0,
            'bom_relationships': len(self.extract_incidence_matrix_data()) if self.bom_train is not None else 0,
            'dataco_records': len(self.dataco) if self.dataco is not None else 0,
            'maintenance_records': len(self.maintenance) if self.maintenance is not None else 0,
        }
        return stats
    
    def load_all(self):
        """Load all available datasets."""
        print("=" * 70)
        print("LOADING REAL DATA FROM DATA SET FOLDER")
        print("=" * 70)
        
        self.load_bom_data()
        try:
            self.load_dataco_data()
        except Exception as e:
            print(f"⚠ Could not load DataCo (file may be too large): {str(e)[:100]}")
        
        self.load_maintenance_data()
        
        print("\n" + "=" * 70)
        print("DATA EXTRACTION")
        print("=" * 70)
        
        nodes = self.extract_supplier_nodes()
        hyperedges = self.extract_product_categories_as_hyperedges()
        incidence = self.extract_incidence_matrix_data()
        reliability = self.extract_reliability_from_maintenance()
        
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        for key, value in self.get_supply_chain_statistics().items():
            print(f"  {key}: {value}")
        
        return {
            'nodes': nodes,
            'hyperedges': hyperedges,
            'incidence': incidence,
            'reliability': reliability
        }
