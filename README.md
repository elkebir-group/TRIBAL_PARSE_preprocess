# BCR Data Processing Pipeline

This repository contains a pipeline for processing B-cell receptor (BCR) sequencing data with a focus on clonotype analysis and consensus sequence generation.

## Overview

The pipeline processes BCR sequencing data by:
1. Filtering multiplets and non-full-length chains
2. Grouping sequences by V(D)J allele combinations
3. Generating consensus sequences for clonotypes
4. Mapping cell barcodes to clonotypes

## Prerequisites

The required dependencies are specified in the `environment.yml` file. You can create a conda environment using: 

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate TRIBAL_preprocess
```

## Usage

To run the pipeline, use the following command:

```bash
python TRIBAL_preprocess.py --config <input_file> [--multiplets <bool>]
```

### Command Line Arguments

- `--config`: Path to the configuration YAML file (required)
- `--multiplets`: Boolean flag to include multiplet analysis (optional, default=False)
  - When set to `True`, the pipeline will:
    - Process cells marked as multiplets
    - Analyze cells with multiple heavy/light chain pairs
    - Attempt to match multiplet chains based on transcript counts and existing clonotypes
    - Add multiplet-derived sequences to the clonotype pool
  - When set to `False`, multiplets are filtered out

Example usage with multiplets:
```bash
python TRIBAL_preprocess.py --config config.yaml --multiplets True
```

## Preprocessing Approaches

The repository implements two different preprocessing approaches, each with its own configuration and command:

### 1. Allele-Based Preprocessing (Current Implementation)

This approach, implemented in `preprocess.py`, groups BCR sequences based on their V(D)J allele combinations:

- **Grouping Strategy**: Sequences are grouped by exact V, D, and J allele matches
- **Advantages**:
  - More precise clonotype definition
  - Better handling of somatic hypermutation
  - Maintains allelic information
- **Use Case**: Preferred when studying allele-specific responses or when high precision in clonotype definition is required

**Run with:**
```bash
python preprocess.py --config config.yaml
```

### 2. Gene-Based Preprocessing (Alternative Approach)

This approach groups sequences based on V(D)J gene families without considering specific alleles:

- **Grouping Strategy**: Sequences are grouped by V, D, and J gene families
- **Advantages**:
  - More lenient grouping
  - Captures broader clonal relationships
  - Less sensitive to allele calling errors
- **Use Case**: Suitable for general repertoire analysis or when studying broader clonal relationships

**Run with:**
```bash
python preprocess.py --config config.yaml
```

Note: Each approach requires its own configuration file (`config_allele.yaml` or `config_gene.yaml`) with appropriate settings for the grouping strategy.

## Configuration Parameters

The `config.yaml` file contains several important parameters that control the pipeline's behavior. For a complete example, see the provided [`config.yaml`](config.yaml) file.

### Dataset Parameters
- `base_path`: Base directory containing input files
- `all_genes_file`: CSV file containing gene information
- `metadata_file`: CSV file with metadata
- `clonotype_file`: Tab-separated file with clonotype information
- `annotation_file`: Tab-separated file with annotation data
- `barcodes_file`: Tab-separated file with cell barcode information

### Barcode Configuration
- `multiplet_column`: Column name identifying multiplet status
- `barcode_column`: Column name containing cell barcodes
- `full_length_column`: Column indicating full-length chain status
- `cdr3_columns`: List of columns containing CDR3 amino acid sequences
- `chain_columns`:
  - `heavy_full_length`: Column indicating full-length heavy chain
  - `kappa_full_length`: Column for full-length kappa chain
  - `lambda_full_length`: Column for full-length lambda chain
  - `heavy_isotype`: Column containing heavy chain isotype information

### Clonotype Configuration
- `clonotype_column`: Column containing clonotype information
- `clonotype_id_column`: Column containing clonotype IDs
- `barcode_column`: Column containing cell barcodes
- `clonotype_mapping_columns`:
  - `light_chain`: Column mapping light chain information
  - `heavy_chain`: Column mapping heavy chain information

### Annotation Configuration
- `cell_barcode_column`: Column name for cell barcodes
- `locus_column`: Column containing chain locus information
- `sequence_column`: Column containing sequence data
- `v_call_column`: Column for V gene calls
- `d_call_column`: Column for D gene calls
- `j_call_column`: Column for J gene calls
- `germline_alignment_column`: Column with germline alignment information
- `chain_types`:
  - `heavy`: Identifier for heavy chain
  - `kappa`: Identifier for kappa chain
  - `lambda`: Identifier for lambda chain

### Column Naming Configuration
- `records`: List of column names for sequence-level data output
  - `cellid`: Cell identifier
  - `clonotype`: Clonotype identifier
  - `heavy_chain_isotype`: Heavy chain isotype
  - `heavy_chain_seq`: Heavy chain sequence
  - `heavy_chain_v_allele`: Heavy chain V allele
  - `light_chain_seq`: Light chain sequence
  - `light_chain_v_allele`: Light chain V allele
- `clono_records`: List of column names for clonotype-level data output
  - `clonotype`: Clonotype identifier
  - `heavy_chain_root`: Consensus heavy chain sequence
  - `light_chain_root`: Consensus light chain sequence

### Output Configuration
- `seq_data_file_script1`: Output path for sequence data (CDR3-based)
- `root_data_file_script1`: Output path for consensus sequences (CDR3-based)
- `clonotype_map_file_script1`: Output path for clonotype mapping (CDR3-based)
- `seq_data_file_script2`: Output path for sequence data (allele-based)
- `root_data_file_script2`: Output path for consensus sequences (allele-based)
- `clonotype_map_file_script2`: Output path for clonotype mapping (allele-based)

### Processing Parameters
- `required_loci`: Minimum number of loci required per cell
- `min_consensus_sequences`: Minimum number of sequences required for consensus generation (default: 1)
- `min_cells_per_clonotype`: Minimum number of cells required to form a clonotype (default: 2)

### Transcript Configuration
- `transcript_count_column`: Column name containing transcript count information

### Consensus Configuration
- `nucleotides`: List of valid nucleotides for consensus generation ['A', 'T', 'C', 'G', 'N']

### Logging Configuration
- `level`: Logging level (e.g., "INFO", "DEBUG", "WARNING")
- `format`: Format string for log messages
- `failed_cells_file`: Output file for recording failed cell processing

## Output Files

The pipeline generates three main output files:
1. `seq_data.csv`: Contains sequence-level information for each cell
2. `root_data.csv`: Contains consensus sequences for each clonotype
3. `clonotype_map.json`: Maps between internal clonotype IDs and their V(D)J definitions

## Logging

The pipeline includes comprehensive logging that tracks:
- Processing progress
- Error messages
- Processing statistics
- Success/failure of file operations

## Error Handling

The pipeline includes robust error handling for:
- Missing or malformed input files
- Data processing errors
- Invalid configurations
- File I/O operations

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[Add your chosen license here]

## Notes

- The pipeline assumes that the input file is a CSV file with the specified columns.
- The pipeline assumes that the input file is sorted by `cell_barcode`.  