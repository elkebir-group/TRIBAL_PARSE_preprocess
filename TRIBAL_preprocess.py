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
import itertools

warnings.filterwarnings('ignore', category=DeprecationWarning)

def parse_args():
    parser = argparse.ArgumentParser(description='Process BCR data using allele information')
    parser.add_argument('--config', type=str, help='Path to config YAML file', default='config.yaml')
    parser.add_argument('--multiplets', type=bool, help='Use mutliplets in data', default=False)
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

def process_data(config, multiplet):
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

    

    clonotype_map = {}
    for i, t in clonotype_by_allele.iterrows():
        try:            
            # Create clonotype identifier
            clonotype = (f"{t[config['annotation']['v_call_column']].replace(',', '_')},"
                        f"{t[config['annotation']['d_call_column']].replace(',', '_')},"
                        f"{t[config['annotation']['j_call_column']].replace(',', '_')}")
            clonotype_map[i] = clonotype

        except Exception as e:
            logging.error(f"Error processing clonotype: {e}")
            continue

    print(f'Number of clonotypes before multiplet filtering: {len(clonotype_by_allele)}')

    if multiplet:
        barcode_multiplet = barcodes[barcodes['isMultiplet'] == 1]
        # get full length multiplet
        barcode_multiplet_full_length = barcode_multiplet[
            (barcode_multiplet[config['barcodes']['chain_columns']['heavy_full_length']] == 1) & 
            ((barcode_multiplet[config['barcodes']['chain_columns']['kappa_full_length']] == 1) | 
                (barcode_multiplet[config['barcodes']['chain_columns']['lambda_full_length']] == 1))
        ]
        annotation_multiplet_full_length = annotation[(annotation['cell_barcode'].isin(barcode_multiplet_full_length['Barcode'])) & (annotation['full_length'] == 1)]

        annotation_multiplet_full_length_grouped = annotation_multiplet_full_length.groupby('cell_barcode').agg({
            'locus': lambda x: list(x), 
            'full_length': 'sum'
        })

        annotation_multiplet_full_length_grouped['IGH_count'] = annotation_multiplet_full_length_grouped['locus'].apply(lambda x: x.count("IGH"))
        annotation_multiplet_full_length_grouped['IGK_count'] = annotation_multiplet_full_length_grouped['locus'].apply(lambda x: x.count('IGK'))
        annotation_multiplet_full_length_grouped['IGL_count'] = annotation_multiplet_full_length_grouped['locus'].apply(lambda x: x.count('IGL'))

        annotation_multiplet_full_length_grouped_filtered = annotation_multiplet_full_length_grouped[
            (annotation_multiplet_full_length_grouped['full_length'] >= 2) & 
            ((annotation_multiplet_full_length_grouped['IGH_count'] == annotation_multiplet_full_length_grouped['IGK_count']) | 
             (annotation_multiplet_full_length_grouped['IGH_count'] == annotation_multiplet_full_length_grouped['IGL_count']))
        ].copy()

        annotation_multiplet_full_length_grouped_filtered['IGH_IGK'] = (
            annotation_multiplet_full_length_grouped_filtered['IGH_count'] == 
            annotation_multiplet_full_length_grouped_filtered['IGK_count']
        )
        annotation_multiplet_full_length_grouped_filtered['IGH_IGL'] = (
            annotation_multiplet_full_length_grouped_filtered['IGH_count'] == 
            annotation_multiplet_full_length_grouped_filtered['IGL_count']
        )

        annotation_multiplet_full_length_valid = annotation_multiplet_full_length[
            annotation_multiplet_full_length['cell_barcode'].isin(
                annotation_multiplet_full_length_grouped_filtered.index)].copy()

        mapping_dict = {
            'IGH_count': annotation_multiplet_full_length_grouped_filtered['IGH_count'].to_dict(),
            'IGK_count': annotation_multiplet_full_length_grouped_filtered['IGK_count'].to_dict(),
            'IGL_count': annotation_multiplet_full_length_grouped_filtered['IGL_count'].to_dict(),
            'IGH_IGK': annotation_multiplet_full_length_grouped_filtered['IGH_IGK'].to_dict(),
            'IGH_IGL': annotation_multiplet_full_length_grouped_filtered['IGH_IGL'].to_dict()
        }

        # Update all columns at once
        for col, mapping in mapping_dict.items():
            annotation_multiplet_full_length_valid[col] = annotation_multiplet_full_length_valid['cell_barcode'].map(mapping)

        # For the call columns
        call_columns = [
            config['annotation']['v_call_column'], 
            config['annotation']['d_call_column'], 
            config['annotation']['j_call_column']
        ]
        annotation_multiplet_full_length_valid[call_columns] = \
            annotation_multiplet_full_length_valid[call_columns].fillna('NA')

        multiplet_cells = annotation_multiplet_full_length_valid['cell_barcode'].unique()

        clonotype_by_allele.loc[:, 'clonotype'] = clonotype_by_allele['v_call'].str.replace(',','_') + ',' + clonotype_by_allele['d_call'].str.replace(',','_') + ',' + clonotype_by_allele['j_call'].str.replace(',','_')
        clonotype_map_reverse = {v: k for k, v in clonotype_map.items()}


        clonotype_by_allele.loc[:, 'clonotype_id'] = clonotype_by_allele['clonotype'].map(clonotype_map_reverse)

        # Initialize new_dataframe with explicit dtypes for all columns
        column_dtypes = {col: annotation_multiplet_full_length_valid[col].dtype 
                        for col in annotation_multiplet_full_length_valid.columns}
        new_dataframe = pd.DataFrame(columns=annotation_multiplet_full_length_valid.columns).astype(column_dtypes)

        def process_dataframe(df, clonotype_map, clonotype_map_reverse, clonotype_by_allele):
            temp = pd.DataFrame(columns=df.columns)  # Temporary dataframe to store results for each cell barcode

            # Check the unique heavy and light chains
            heavy_chains = df[df['locus'] == 'IGH']
            light_chains = df[df['locus'].isin(['IGK', 'IGL'])]
            
            n_heavy = len(heavy_chains)
            n_light = len(light_chains)
            
            # Case 1: n == 1
            if n_heavy == 1:
                heavy = heavy_chains.iloc[0]
                
                if n_light == 1:
                    # Pair the heavy chain with the light chain
                    light = light_chains.iloc[0]
                    pair = combine_pairs(heavy, light)
                    temp = pair
                    # temp = pd.concat([temp, pair], ignore_index=True)
                    update_clonotype(heavy, light, clonotype_map, clonotype_map_reverse, clonotype_by_allele)
                
                elif n_light > 1:
                    # Check IGH_IGK vs IGH_IGL
                    light = select_best_light(light_chains)
                    pair = combine_pairs(heavy, light)
                    temp = pair
                    # temp = pd.concat([temp, pair], ignore_index=True)
                    update_clonotype(heavy, light, clonotype_map, clonotype_map_reverse, clonotype_by_allele)

            # Case 2: n > 1
            elif n_heavy > 1:
                pairs = []
                heavy_chains_list = []
                light_chains_list = []
                clonotype_list = {}
                for heavy, light in itertools.product(heavy_chains.iterrows(), light_chains.iterrows()):
                    heavy = heavy[1]
                    light = light[1]
                    clonotype_key = f"{heavy['v_call']}_{light['v_call']},{heavy['d_call']}_{light['d_call']},{heavy['j_call']}_{light['j_call']}"
                    clonotype_id = clonotype_map_reverse.get(clonotype_key, None)

                    # print(clonotype_key, clonotype_id)

                    if clonotype_id:
                        pair = combine_pairs(heavy, light)
                        pairs.append(pair)
                        # print(clonotype_key, clonotype_id)
                        row_idx = clonotype_by_allele[clonotype_by_allele['clonotype_id'] == clonotype_id].index[0]
                        # print(clonotype_by_allele.at[clonotype_id, 'count'])
                        # print(0)
                        clonotype_list[clonotype_key] = (clonotype_by_allele.at[row_idx, 'count'] + 1, pair)
                        # print(1)
                        
                        # update_clonotype(heavy, light, clonotype_map, clonotype_map_reverse, clonotype_by_allele)
                    else:
                        heavy_chains_list.append(heavy.to_dict())
                        light_chains_list.append(light.to_dict())

                    # Check if pair exists in clonotype_map
                    # if not is_clonotype_match(heavy, light, clonotype_map, clonotype_map_reverse):
                    # update_clonotype(heavy, light, clonotype_map, clonotype_map_reverse, clonotype_by_allele)

                heavy_chains_list = pd.DataFrame.from_records(heavy_chains_list)
                light_chains_list = pd.DataFrame.from_records(light_chains_list)
                # print(cell)
                # print(heavy_chains_list.columns)
                # print(len(pairs), len(heavy_chains_list), len(heavy_chains))

                if len(pairs) > len(heavy_chains):
                    # Sort the dictionary by the first term in the value (ascending or descending)
                    sorted_keys = sorted(clonotype_list, key=lambda k: clonotype_list[k][0], reverse=True)
                    # Keep only the second term in the value for the sorted result
                    pairs = [clonotype_list[key][1] for key in sorted_keys][:len(heavy_chains)]

                elif len(pairs) < len(heavy_chains):
                    for heavy, light in pair_by_transcript_count(heavy_chains_list, light_chains_list):
                        pair = combine_pairs(heavy, light)
                        pairs.append(pair)
                        # temp = pd.concat([temp, pair], ignore_index=True)
                        # update_clonotype(heavy, light, clonotype_map, clonotype_map_reverse, clonotype_by_allele)
                if pairs:
                    for i, pair in enumerate(pairs):
                        pair['cell_barcode'] = pair['cell_barcode'] + f"_{i}"
                        update_clonotype(pair.iloc[0], pair.iloc[1], clonotype_map, clonotype_map_reverse, clonotype_by_allele)
                        # Update the original pairs list
                        pairs[i] = pair
                        # print(pairs[i]['cell_barcode'])
                        

                temp = pd.concat(pairs, ignore_index=True)
                # print(temp['cell_barcode'])
            
            return temp

        def select_best_light(light_chains):
            """Select light chain based on IGH_IGK/IGH_IGL or transcript count."""
            # Create a copy of the DataFrame to avoid the warning
            light_chains = light_chains.copy()
            
            # Use loc to set values
            light_chains.loc[:, 'IGH_IG'] = light_chains.apply(lambda x: x['IGH_IGK'] or x['IGH_IGL'], axis=1)
            
            # Sort without modifying in place
            light_chains = light_chains.sort_values(by=['IGH_IG', 'transcript_count'], ascending=[False, False])
            
            return light_chains.iloc[0]

        def is_clonotype_match(heavy, light, clonotype_map, clonotype_map_reverse):
            """Check if the given pair exists in the clonotype map."""
            clonotype_key = f"{heavy['v_call']}_{light['v_call']},{heavy['d_call']}_{light['d_call']},{heavy['j_call']}_{light['j_call']}"
            return clonotype_key in clonotype_map_reverse.keys()

        def combine_pairs(heavy, light):
            """Combine heavy and light chains into a single row."""
            
            pair = pd.DataFrame.from_records([heavy, light])
            return pair

        def pair_by_transcript_count(heavy_chains, light_chains):
            """
            Pair heavy and light chains based on transcript count, matching the highest-transcript heavy chain
            with the highest-transcript light chain and continuing until one set is exhausted.
            """
            # Sort both heavy and light chains by transcript count (descending)
            heavy_sorted = heavy_chains.sort_values(by='transcript_count', ascending=False).reset_index(drop=True)
            light_sorted = light_chains.sort_values(by='transcript_count', ascending=False).reset_index(drop=True)
            
            # Initialize pairs list
            pairs = []
            
            # Iterate through the smaller of the two sorted lists
            for i in range(min(len(heavy_sorted), len(light_sorted))):
                heavy = heavy_sorted.iloc[i]
                light = light_sorted.iloc[i]
                pairs.append((heavy, light))
            
            return pairs

        def update_clonotype(heavy, light, clonotype_map, clonotype_map_reverse, clonotype_by_allele):
            """Update clonotype maps and clonotype-to-cell relationships."""
            global new_dataframe
            # print(type(heavy), type(light))
            clonotype_key = f"{heavy['v_call']}_{light['v_call']},{heavy['d_call']}_{light['d_call']},{heavy['j_call']}_{light['j_call']}"
            clonotype_id = clonotype_map_reverse.get(clonotype_key, None)

            if clonotype_id is None:
                # Create a new clonotype ID
                clonotype_id = max(clonotype_map.keys(), default=0) + 1
                clonotype_map[clonotype_id] = clonotype_key
                clonotype_map_reverse[clonotype_key] = clonotype_id
            if clonotype_id in clonotype_by_allele['clonotype_id'].unique():
                row_idx = clonotype_by_allele[clonotype_by_allele['clonotype_id'] == clonotype_id].index[0]
                clonotype_by_allele.at[row_idx, 'count'] += 1
                # clonotype_by_allele.at[row_idx, 'len'] += 1
                clonotype_by_allele.at[row_idx, 'indices'] = list(set(clonotype_by_allele.at[row_idx, 'indices'] + [heavy['cell_barcode']]))
            else:
                new_row = {
                    'v_call': f"{heavy['v_call']},{light['v_call']}",
                    'd_call': f"{heavy['d_call']},{light['d_call']}",
                    'j_call': f"{heavy['j_call']},{light['j_call']}",
                    'count': 1,
                    'indices': [heavy['cell_barcode']],
                    # 'len': 1,
                    'clonotype': clonotype_key,
                    'clonotype_id': clonotype_id
                }
                clonotype_by_allele.loc[len(clonotype_by_allele)] = new_row
                # clonotype_by_allele = pd.concat([clonotype_by_allele, pd.DataFrame([new_row])], ignore_index=True)
                # print(f'length of clonotype_by_allele has been changed to: {len(clonotype_by_allele)}')

        # Example usage:
        multiplet_cells = annotation_multiplet_full_length_valid['cell_barcode'].unique()

        for cell in multiplet_cells:
            df = annotation_multiplet_full_length_valid[
                annotation_multiplet_full_length_valid['cell_barcode'] == cell].copy()
            temp = process_dataframe(df, clonotype_map, clonotype_map_reverse, clonotype_by_allele)
            
            # Only concatenate if temp is not empty
            if not temp.empty:
                # Ensure temp has the same columns and dtypes as new_dataframe
                temp = temp.reindex(columns=new_dataframe.columns)
                for col in new_dataframe.columns:
                    temp[col] = temp[col].astype(new_dataframe[col].dtype)
                new_dataframe = pd.concat([new_dataframe, temp], ignore_index=True)

    print(f'Number of clonotypes after multiplet filtering: {len(clonotype_by_allele)}')
    # Process clonotypes
    records = []
    clono_records = []

    # Filter for multiple cells
    clonotype_by_allele_more = clonotype_by_allele[clonotype_by_allele['count'] >= config['parameters']['min_cells_per_clonotype']]
    

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
                        if multiplet:
                            temp = new_dataframe[
                                new_dataframe[config['annotation']['cell_barcode_column']] == cell]
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
                        # Get isotype
                        try:
                            IGH_C = barcode_rm_multiplet.loc[
                                barcode_rm_multiplet[config['barcodes']['barcode_column']] == cell, 
                                config['barcodes']['chain_columns']['heavy_isotype']].values[0]
                        except:
                            try:
                                ids = cell.split('_')
                                if len(ids) > 5:
                                    id = '_'.join(cell.split('_')[:-1])
                                else:
                                    id = '_'.join(cell.split('_'))
                                IGH_C = barcode_multiplet.loc[
                                barcode_multiplet[config['barcodes']['barcode_column']] == id, 
                                config['barcodes']['chain_columns']['heavy_isotype']].values[0]
                            except:
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

def main(config_path, multiplet):
    # Load configuration
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Configure logging
    logging.basicConfig(level=config['logging']['level'], format=config['logging']['format'])
    
    logging.info(f"Starting processing with config file: {config_path}")
    
    try:
        # Process the data
        seq_data, root_data, clonotype_map = process_data(config, multiplet)
        
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
    main(args.config, args.multiplets)
