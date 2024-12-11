# %%
import pandas as pd 
import numpy as np
import json
import pickle
import yaml
from scipy.io import mmread
import scipy.io
import gzip
import logging
import argparse
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore', category=DeprecationWarning)

def parse_args():
    parser = argparse.ArgumentParser(description='Process BCR data using allele information')
    parser.add_argument('--config', type=str, help='Path to config YAML file', default='config.yaml')
    return parser.parse_args()

def consensus(sequences, nucleotides):
    """
    Generate a consensus sequence from a list of sequences.
    """
    if len(sequences) <= 1:
        return sequences[0]
    
    max_len = max(len(seq) for seq in sequences)
    cons_df = pd.DataFrame(0, index=range(max_len), columns=nucleotides)
    
    for seq in sequences:
        for i, nt in enumerate(seq):
            if nt in cons_df.columns:
                cons_df.loc[i, nt] += 1

    return ''.join(cons_df.idxmax(axis=1))

def process_data(config):
    """Main processing function"""
    # Load input files
    base = config['dataset']['base_path']
    all_genes = pd.read_csv(f"{base}/{config['dataset']['all_genes_file']}")
    metadata = pd.read_csv(f"{base}/{config['dataset']['metadata_file']}")
    clonotype = pd.read_csv(f"{base}/{config['dataset']['clonotype_file']}", sep='\t')
    annotation = pd.read_csv(f"{base}/{config['dataset']['annotation_file']}", sep='\t')
    barcodes = pd.read_csv(f"{base}/{config['dataset']['barcodes_file']}", sep='\t')
    
    # Filter data
    barcode_rm_multiplet = barcodes[barcodes[config['barcodes']['multiplet_column']] == 0]
    annotation_rm_multiplet = annotation[annotation[config['annotation']['cell_barcode_column']].isin(
        barcode_rm_multiplet[config['barcodes']['barcode_column']])]
    # annotation_full_length = annotation[annotation[config['barcodes']['full_length_column']] == 1]

    # Filter for full-length chains
    barcodes_full_length = barcodes[
        (barcodes[config['barcodes']['chain_columns']['heavy_full_length']] == 1) & 
        ((barcodes[config['barcodes']['chain_columns']['kappa_full_length']] == 1) | 
            (barcodes[config['barcodes']['chain_columns']['lambda_full_length']] == 1))
    ]

    barcodes_rm_multiplet_full_length = barcodes_full_length[barcodes_full_length[config['barcodes']['multiplet_column']] == 0]
    barcodes_rm_multiplet_full_length_2 = barcode_rm_multiplet[
        (barcode_rm_multiplet[config['barcodes']['chain_columns']['heavy_full_length']] == 1) & 
        ((barcode_rm_multiplet[config['barcodes']['chain_columns']['kappa_full_length']] == 1) | 
            (barcode_rm_multiplet[config['barcodes']['chain_columns']['lambda_full_length']] == 1))
    ]

    assert len(barcodes_rm_multiplet_full_length) == len(barcodes_rm_multiplet_full_length_2), \
        "Number of barcodes with full-length chains does not match"

    annotation_rm_multiplet_full_length = annotation[(annotation[config['annotation']['cell_barcode_column']].isin(
        barcodes_rm_multiplet_full_length_2[config['barcodes']['barcode_column']])) & (annotation['full_length'] == 1)]

    print(f'Number of cells with full-length chains: {len(barcodes_rm_multiplet_full_length_2)}')

    # Fill missing values
    call_columns = [config['annotation']['v_call_column'], 
                    config['annotation']['d_call_column'], 
                    config['annotation']['j_call_column']]
    annotation_rm_multiplet_full_length.loc[:, call_columns] = \
        annotation_rm_multiplet_full_length[call_columns].fillna('NA')

    # Group annotations
    annotation_group = annotation_rm_multiplet_full_length.groupby(config['annotation']['cell_barcode_column']).agg({
        config['annotation']['locus_column']: list,
        config['annotation']['v_call_column']: lambda x: ','.join(map(str, sorted(x))),
        config['annotation']['d_call_column']: lambda x: ','.join(map(str, sorted(x))),
        config['annotation']['j_call_column']: lambda x: ','.join(map(str, sorted(x))),
    })

    # Select groups with exactly two loci
    annotation_group_2_locus = annotation_group[
        annotation_group[config['annotation']['locus_column']].str.len() == 2]

    # Group by alleles (fixed deprecation warning)
    clonotype_by_allele = annotation_group_2_locus.groupby(
        [
            config['annotation']['v_call_column'],
            config['annotation']['d_call_column'],
            config['annotation']['j_call_column']
        ],
        as_index=False
    ).apply(lambda x: pd.Series({
        'count': len(x), 
        'indices': list(x.index)
    }))

    # Filter for multiple cells
    clonotype_by_allele_more = clonotype_by_allele[clonotype_by_allele['count'] >= config['parameters']['min_cells_per_clonotype']]

    # Process clonotypes
    records = []
    clono_records = []
    clonotype_map = {}

    # Add progress bar
    pbar = tqdm(clonotype_by_allele_more.iterrows(), 
                total=len(clonotype_by_allele_more),
                desc="Processing clonotypes")
    
    for i, t in pbar:
        try:
            consensus_H = []
            consensus_L = []
            
            # Create clonotype identifier
            clonotype = (f"{t[config['annotation']['v_call_column']].replace(',', '_')},"
                        f"{t[config['annotation']['d_call_column']].replace(',', '_')},"
                        f"{t[config['annotation']['j_call_column']].replace(',', '_')}")
            clonotype_map[i] = clonotype
            
            pbar.set_postfix({'clonotype': i})  # Show current clonotype in progress bar
            
            for cell in t['indices']:
                try:
                    temp = annotation_rm_multiplet_full_length[
                        annotation_rm_multiplet_full_length[config['annotation']['cell_barcode_column']] == cell]
                    
                    if len(temp) == 0:
                        logging.error(f"Cell {cell} not found in filtered annotation data")
                        continue
                    
                    try:
                        # Check for heavy chain presence
                        heavy_chain_data = temp[temp[config['annotation']['locus_column']] == config['annotation']['chain_types']['heavy']]
                        if len(heavy_chain_data) == 0:
                            logging.error(f"Cell {cell} is missing heavy chain sequence")
                            continue

                        # Check for light chain presence
                        light_chain_data = temp[temp[config['annotation']['locus_column']].isin([
                            config['annotation']['chain_types']['lambda'],
                            config['annotation']['chain_types']['kappa']
                        ])]
                        if len(light_chain_data) == 0:
                            logging.error(f"Cell {cell} is missing light chain sequence")
                            continue

                        # Get sequences and v_calls
                        try:
                            IGH = heavy_chain_data[config['annotation']['sequence_column']].values[0]
                            H_v = heavy_chain_data[config['annotation']['v_call_column']].values[0]
                        except (IndexError, KeyError):
                            logging.error(f"Cell {cell} has invalid heavy chain sequence data")
                            continue

                        try:
                            IGL = light_chain_data[config['annotation']['sequence_column']].values[0]
                            L_v = light_chain_data[config['annotation']['v_call_column']].values[0]
                        except (IndexError, KeyError):
                            logging.error(f"Cell {cell} has invalid light chain sequence data")
                            continue
                        
                        # Get isotype
                        try:
                            IGH_C = barcode_rm_multiplet.loc[
                                barcode_rm_multiplet[config['barcodes']['barcode_column']] == cell, 
                                config['barcodes']['chain_columns']['heavy_isotype']].values[0]
                        except IndexError:
                            logging.error(f"Cell {cell} is missing isotype information in barcode report")
                            IGH_C = None
                        
                        # Get germline alignments
                        try:
                            H_germ = heavy_chain_data[config['annotation']['germline_alignment_column']].values[0]
                        except (IndexError, KeyError):
                            logging.error(f"Cell {cell} is missing heavy chain germline alignment")
                            continue

                        try:
                            L_germ = light_chain_data[config['annotation']['germline_alignment_column']].values[0]
                        except (IndexError, KeyError):
                            logging.error(f"Cell {cell} is missing light chain germline alignment")
                            continue
                        
                        if not isinstance(H_germ, str) or not isinstance(L_germ, str):
                            logging.error(f"Cell {cell} has non-string germline alignment data")
                            continue
                        
                        consensus_H.append(H_germ)
                        consensus_L.append(L_germ)
                        
                        records.append({
                            config['columns']['records'][0]: cell,
                            config['columns']['records'][1]: i,
                            config['columns']['records'][2]: IGH_C,
                            config['columns']['records'][3]: IGH,
                            config['columns']['records'][4]: H_v,
                            config['columns']['records'][5]: IGL,
                            config['columns']['records'][6]: L_v
                        })
                        
                    except Exception as e:
                        logging.error(f"Error processing sequence data for cell {cell}: {str(e)}")
                        continue
                    
                except Exception as cell_error:
                    logging.error(f"Error accessing annotation data for cell {cell}: {str(cell_error)}")
                    continue
                
            # Create consensus record
            if len(consensus_H) < config['parameters']['min_consensus_sequences']:
                logging.error(f"Clonotype {i} has insufficient heavy chain sequences for consensus ({len(consensus_H)} < {config['parameters']['min_consensus_sequences']})")
                continue
                
            if len(consensus_L) < config['parameters']['min_consensus_sequences']:
                logging.error(f"Clonotype {i} has insufficient light chain sequences for consensus ({len(consensus_L)} < {config['parameters']['min_consensus_sequences']})")
                continue
                
            clono_records.append({
                config['columns']['clono_records'][0]: i,
                config['columns']['clono_records'][1]: consensus(consensus_H, config['consensus']['nucleotides']),
                config['columns']['clono_records'][2]: consensus(consensus_L, config['consensus']['nucleotides'])
            })
                
        except Exception as row_error:
            logging.error(f"Error processing clonotype {i}: {str(row_error)}")
            continue

    # Save results
    seq_data = pd.DataFrame.from_records(records)
    root_data = pd.DataFrame.from_records(clono_records)
    
    return seq_data, root_data, clonotype_map

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Configure logging
    logging.basicConfig(level=config['logging']['level'], format=config['logging']['format'])
    
    logging.info(f"Starting processing with config file: {config_path}")
    
    try:
        # Process the data
        seq_data, root_data, clonotype_map = process_data(config)
        
        # Log processing statistics
        logging.info(f"Processing completed:")
        logging.info(f"- Total sequences processed: {len(seq_data)} entries in seq_data")
        logging.info(f"- Total clonotypes processed: {len(root_data)} entries in root_data")
        logging.info(f"- Average sequences per clonotype: {len(seq_data)/len(root_data):.2f}")
        
        # Save results
        seq_data.to_csv(config['output']['seq_data_file_script2'], index=False)
        root_data.to_csv(config['output']['root_data_file_script2'], index=False)
        
        with open(config['output']['clonotype_map_file_script2'], 'w') as f:
            json.dump(clonotype_map, f)
            
        logging.info("Results saved successfully")
        
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
