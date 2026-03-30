import pandas as pd
import numpy as np

df = pd.read_csv('Data set/IndiGo/indigo_disruption.csv')

print('=== INDIGO DATASET STATISTICS ===')
print(f'Total records: {len(df):,}')
print(f'Columns: {list(df.columns)}')
print('')

print('=== SAMPLE ROWS ===')
print(df.head(3).to_string())
print('')

print('=== DATA TYPES ===')
print(df.dtypes)
print('')

print('=== UNIQUE VALUES ===')
for col in df.columns:
    unique_count = df[col].nunique()
    print(f'{col}: {unique_count} unique values')

print('')
print('=== CLASS DISTRIBUTION ===')
if 'criticality_class' in df.columns:
    labels = df['criticality_class'].values
    unique, counts = np.unique(labels, return_counts=True)
    print('criticality_class distribution:')
    print(dict(zip(unique, counts)))
elif 'Criticality_Class' in df.columns:
    labels = df['Criticality_Class'].values
    unique, counts = np.unique(labels, return_counts=True)
    print('Criticality_Class distribution:')
    print(dict(zip(unique, counts)))
else:
    print('No criticality class column found')
    print('Available columns:', df.columns.tolist())
