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

    # Group by cell barcode and count loci
    cell_barcode_by_locus_num = annotation_rm_multiplet_full_length.groupby(
        config['annotation']['cell_barcode_column'])[config['annotation']['locus_column']].apply(list).reset_index()
    cell_barcode_by_locus_num.loc[:, 'len'] = cell_barcode_by_locus_num[config['annotation']['locus_column']].apply(len)

    # Select cells with exactly two loci
    cell_barcode_by_locus_num_more = cell_barcode_by_locus_num[
        cell_barcode_by_locus_num['len'] == config['parameters']['required_loci']]

    # Filter barcodes for selected cells
    barcodes_full_length_more = barcodes_rm_multiplet_full_length[
        barcodes_rm_multiplet_full_length[config['barcodes']['barcode_column']].isin(
            cell_barcode_by_locus_num_more[config['annotation']['cell_barcode_column']].to_list()
        )
    ]

    # Create clonotype mapping from CDR3 sequences
    barcodes_full_length_more.loc[:, config['clonotype_columns']['clonotype_column']] = barcodes_full_length_more[
        config['barcodes']['cdr3_columns']
    ].apply(lambda row: '_'.join(row.dropna()), axis=1)

    # Map clonotypes to IDs
    clonotype.loc[:, config['clonotype_columns']['clonotype_column']] = clonotype[
        [config['clonotype_columns']['clonotype_mapping_columns']['light_chain'],
         config['clonotype_columns']['clonotype_mapping_columns']['heavy_chain']]
    ].apply(lambda row: '_'.join(row.dropna()), axis=1)
    
    clonotype_map = dict(zip(
        clonotype[config['clonotype_columns']['clonotype_column']],
        clonotype[config['clonotype_columns']['clonotype_id_column']]
    ))

    # Add clonotype IDs to barcodes
    barcodes_full_length_more.loc[:, config['clonotype_columns']['clonotype_id_column']] = \
        barcodes_full_length_more[config['clonotype_columns']['clonotype_column']].map(clonotype_map)

    # Group by clonotype ID (fixed deprecation warning)
    barcodes_group = barcodes_full_length_more.groupby(
        config['clonotype_columns']['clonotype_id_column'],
        as_index=False
    ).agg({
        config['barcodes']['barcode_column']: list
    })

    # Filter for clonotypes with multiple cells
    barcodes_group = barcodes_group[
        barcodes_group[config['barcodes']['barcode_column']].str.len() >= 
        config['parameters']['min_cells_per_clonotype']]
    # Add clonotype information to the grouped DataFrame
    barcodes_group.loc[:, config['clonotype_columns']['clonotype_column']] = barcodes_group.index


    print(f'{len(barcodes_group)} clonotypes with multiple cells')

    # Process clonotypes
    records = []
    clono_records = []
    repeat_locus = []

    # Add progress bar
    pbar = tqdm(barcodes_group.iterrows(), 
                total=len(barcodes_group),
                desc="Processing clonotypes")
    
    for i, t in pbar:
        try:
            consensus_H = []
            consensus_L = []
            clonotype_id = t[config['clonotype_columns']['clonotype_column']]
            
            pbar.set_postfix({'clonotype_id': clonotype_id})  # Show current clonotype in progress bar
            
            for cell in t[config['barcodes']['barcode_column']]:
                try:
                    temp = annotation_rm_multiplet_full_length[
                        annotation_rm_multiplet_full_length[config['annotation']['cell_barcode_column']] == cell]
                    
                    if len(temp[config['annotation']['locus_column']].unique()) < 2:
                        logging.error(f"Cell {cell} does not have both heavy and light chains")
                        repeat_locus.append(cell)
                        continue

                    try:
                        # Check for heavy chain presence
                        heavy_chain_data = temp[temp[config['annotation']['locus_column']] == 
                                              config['annotation']['chain_types']['heavy']]
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
                            if not isinstance(H_germ, str):
                                logging.error(f"Cell {cell} has non-string heavy chain germline alignment")
                                continue
                        except (IndexError, KeyError):
                            logging.error(f"Cell {cell} is missing heavy chain germline alignment")
                            continue

                        try:
                            L_germ = light_chain_data[config['annotation']['germline_alignment_column']].values[0]
                            if not isinstance(L_germ, str):
                                logging.error(f"Cell {cell} has non-string light chain germline alignment")
                                continue
                        except (IndexError, KeyError):
                            logging.error(f"Cell {cell} is missing light chain germline alignment")
                            continue
                        
                        consensus_H.append(H_germ)
                        consensus_L.append(L_germ)
                        
                        records.append({
                            config['columns']['records'][0]: cell,
                            config['columns']['records'][1]: clonotype_id,
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
                logging.error(f"Clonotype {clonotype_id} has insufficient heavy chain sequences for consensus ({len(consensus_H)} < {config['parameters']['min_consensus_sequences']})")
                continue
                
            if len(consensus_L) < config['parameters']['min_consensus_sequences']:
                logging.error(f"Clonotype {clonotype_id} has insufficient light chain sequences for consensus ({len(consensus_L)} < {config['parameters']['min_consensus_sequences']})")
                continue
                
            clono_records.append({
                config['columns']['clono_records'][0]: clonotype_id,
                config['columns']['clono_records'][1]: consensus(consensus_H, config['consensus']['nucleotides']),
                config['columns']['clono_records'][2]: consensus(consensus_L, config['consensus']['nucleotides'])
            })

        except Exception as row_error:
            logging.error(f"Error processing clonotype ID {i}: {str(row_error)}")
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
        seq_data.to_csv(config['output']['seq_data_file_script1'], index=False)
        root_data.to_csv(config['output']['root_data_file_script1'], index=False)
        
        with open(config['output']['clonotype_map_file_script1'], 'w') as f:
            json.dump(clonotype_map, f)
            
        logging.info("Results saved successfully")
        
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
