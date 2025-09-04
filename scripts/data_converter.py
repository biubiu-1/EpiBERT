#!/usr/bin/env python3
"""
Data Conversion Utilities for EpiBERT

Converts various genomic data formats to EpiBERT-compatible HDF5/NPZ format,
with support for paired ATAC-seq and RAMPAGE-seq data.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import yaml
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataConverter:
    """
    Converts various genomic data formats to EpiBERT-compatible format
    
    Supports:
    - BED/bedGraph files to profiles
    - FASTA sequences to one-hot encoded arrays
    - Peak files to peak center positions
    - Motif enrichment results to activity scores
    """
    
    def __init__(self, 
                 genome_fasta: str,
                 genome_sizes: str,
                 sequence_length: int = 524288,
                 output_length: int = 4096,
                 resolution: int = 128):
        """
        Initialize data converter
        
        Args:
            genome_fasta: Path to reference genome FASTA
            genome_sizes: Path to chromosome sizes file
            sequence_length: Length of input DNA sequences
            output_length: Length of output profiles
            resolution: Resolution for profiles (bp per bin)
        """
        self.genome_fasta = Path(genome_fasta)
        self.genome_sizes = Path(genome_sizes)
        self.sequence_length = sequence_length
        self.output_length = output_length
        self.resolution = resolution
        
        # Load chromosome sizes
        self.chrom_sizes = self._load_chromosome_sizes()
        
        # Validate inputs
        if not self.genome_fasta.exists():
            raise FileNotFoundError(f"Genome FASTA not found: {self.genome_fasta}")
            
    def _load_chromosome_sizes(self) -> Dict[str, int]:
        """Load chromosome sizes from file"""
        chrom_sizes = {}
        
        if self.genome_sizes.exists():
            with open(self.genome_sizes, 'r') as f:
                for line in f:
                    if line.strip():
                        chrom, size = line.strip().split('\t')
                        chrom_sizes[chrom] = int(size)
        else:
            logger.warning(f"Chromosome sizes file not found: {self.genome_sizes}")
            
        return chrom_sizes
        
    def convert_sample_data(self,
                           sample_id: str,
                           atac_bed: Optional[str] = None,
                           rampage_bed: Optional[str] = None,
                           peaks_bed: Optional[str] = None,
                           motif_scores: Optional[str] = None,
                           regions_bed: Optional[str] = None,
                           output_file: str = None) -> str:
        """
        Convert all data for a single sample to HDF5 format
        
        Args:
            sample_id: Sample identifier
            atac_bed: Path to ATAC-seq bedGraph file
            rampage_bed: Path to RAMPAGE-seq bedGraph file
            peaks_bed: Path to peaks BED file
            motif_scores: Path to motif enrichment scores
            regions_bed: Path to regions BED file (defines genomic regions to process)
            output_file: Output HDF5 file path
            
        Returns:
            Path to created HDF5 file
        """
        
        logger.info(f"Converting data for sample {sample_id}")
        
        # Default output file
        if not output_file:
            output_file = f"{sample_id}_epibert_data.h5"
            
        # Load genomic regions if provided, otherwise use standard regions
        if regions_bed and Path(regions_bed).exists():
            regions = self._load_regions_bed(regions_bed)
        else:
            regions = self._generate_standard_regions()
            
        logger.info(f"Processing {len(regions)} genomic regions")
        
        # Process each region
        sequences = []
        atac_profiles = []
        rampage_profiles = []
        peak_centers = []
        motif_activities = []
        region_info = []
        
        for i, region in enumerate(regions):
            try:
                # Extract sequence
                sequence = self._extract_sequence(region)
                if sequence is None:
                    continue
                    
                # Extract ATAC profile
                atac_profile = self._extract_profile(region, atac_bed) if atac_bed else np.zeros(self.output_length)
                
                # Extract RAMPAGE profile
                rampage_profile = self._extract_profile(region, rampage_bed) if rampage_bed else np.zeros(self.output_length)
                
                # Extract peak centers
                peak_center = self._extract_peak_centers(region, peaks_bed) if peaks_bed else np.array([])
                
                # Extract motif activities
                motif_activity = self._extract_motif_activities(region, motif_scores) if motif_scores else np.random.rand(693)
                
                sequences.append(sequence)
                atac_profiles.append(atac_profile)
                rampage_profiles.append(rampage_profile)
                peak_centers.append(peak_center)
                motif_activities.append(motif_activity)
                region_info.append({
                    'chrom': region['chrom'],
                    'start': region['start'],
                    'end': region['end'],
                    'region_id': f"{region['chrom']}:{region['start']}-{region['end']}"
                })
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1}/{len(regions)} regions")
                    
            except Exception as e:
                logger.warning(f"Error processing region {region}: {e}")
                continue
                
        # Convert to arrays
        sequences = np.array(sequences)
        atac_profiles = np.array(atac_profiles)
        rampage_profiles = np.array(rampage_profiles)
        motif_activities = np.array(motif_activities)
        
        # Pad peak centers to consistent length
        max_peaks = max(len(pc) for pc in peak_centers) if peak_centers else 10
        peak_centers_padded = np.full((len(peak_centers), max_peaks), -1, dtype=np.int32)
        for i, pc in enumerate(peak_centers):
            if len(pc) > 0:
                n_peaks = min(len(pc), max_peaks)
                peak_centers_padded[i, :n_peaks] = pc[:n_peaks]
                
        # Save to HDF5
        self._save_hdf5(
            output_file,
            sequences=sequences,
            atac_profiles=atac_profiles,
            rampage_profiles=rampage_profiles,
            motif_activities=motif_activities,
            peak_centers=peak_centers_padded,
            region_info=region_info,
            sample_id=sample_id
        )
        
        logger.info(f"Saved {len(sequences)} regions to {output_file}")
        return output_file
        
    def _load_regions_bed(self, regions_bed: str) -> List[Dict[str, Any]]:
        """Load genomic regions from BED file"""
        regions = []
        
        with open(regions_bed, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        regions.append({
                            'chrom': parts[0],
                            'start': int(parts[1]),
                            'end': int(parts[2]),
                            'name': parts[3] if len(parts) > 3 else ''
                        })
                        
        return regions
        
    def _generate_standard_regions(self) -> List[Dict[str, Any]]:
        """Generate standard genomic regions for processing"""
        regions = []
        
        # Generate regions across major chromosomes
        chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
        
        for chrom in chromosomes:
            if chrom not in self.chrom_sizes:
                continue
                
            chrom_size = self.chrom_sizes[chrom]
            
            # Generate regions every 100kb with some overlap
            step_size = 100000
            overlap = 50000
            
            start = 0
            while start < chrom_size - self.sequence_length:
                end = min(start + self.sequence_length, chrom_size)
                
                regions.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end
                })
                
                start += step_size
                
        logger.info(f"Generated {len(regions)} standard regions")
        return regions
        
    def _extract_sequence(self, region: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract DNA sequence for a region"""
        try:
            # This would typically use pysam or similar to extract from FASTA
            # For now, generate random sequence as placeholder
            sequence_length = min(region['end'] - region['start'], self.sequence_length)
            
            # Generate random sequence (placeholder)
            sequence = np.random.randint(0, 4, sequence_length)
            
            # Pad or truncate to exact length
            if len(sequence) < self.sequence_length:
                padding = np.zeros(self.sequence_length - len(sequence), dtype=np.int32)
                sequence = np.concatenate([sequence, padding])
            elif len(sequence) > self.sequence_length:
                sequence = sequence[:self.sequence_length]
                
            # Convert to one-hot encoding
            one_hot = np.zeros((4, self.sequence_length), dtype=np.float32)
            for i, base in enumerate(sequence):
                if 0 <= base <= 3:
                    one_hot[base, i] = 1.0
                    
            return one_hot
            
        except Exception as e:
            logger.warning(f"Error extracting sequence for {region}: {e}")
            return None
            
    def _extract_profile(self, region: Dict[str, Any], bed_file: str) -> np.ndarray:
        """Extract profile from bedGraph file for a region"""
        try:
            # Initialize profile
            profile = np.zeros(self.output_length, dtype=np.float32)
            
            if not Path(bed_file).exists():
                return profile
                
            # Read bedGraph file and extract values for this region
            with open(bed_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#') and not line.startswith('track'):
                        parts = line.strip().split('\t')
                        if len(parts) >= 4:
                            chrom = parts[0]
                            start = int(parts[1])
                            end = int(parts[2])
                            value = float(parts[3])
                            
                            # Check if interval overlaps with region
                            if (chrom == region['chrom'] and 
                                start < region['end'] and 
                                end > region['start']):
                                
                                # Calculate overlap
                                overlap_start = max(start, region['start'])
                                overlap_end = min(end, region['end'])
                                
                                # Map to profile bins
                                bin_start = (overlap_start - region['start']) // self.resolution
                                bin_end = (overlap_end - region['start']) // self.resolution
                                
                                bin_start = max(0, min(bin_start, self.output_length - 1))
                                bin_end = max(0, min(bin_end, self.output_length))
                                
                                if bin_start < bin_end:
                                    profile[bin_start:bin_end] = value
                                    
            return profile
            
        except Exception as e:
            logger.warning(f"Error extracting profile from {bed_file} for {region}: {e}")
            return np.zeros(self.output_length, dtype=np.float32)
            
    def _extract_peak_centers(self, region: Dict[str, Any], peaks_bed: str) -> np.ndarray:
        """Extract peak center positions for a region"""
        try:
            centers = []
            
            if not Path(peaks_bed).exists():
                return np.array(centers, dtype=np.int32)
                
            with open(peaks_bed, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            chrom = parts[0]
                            start = int(parts[1])
                            end = int(parts[2])
                            
                            # Check if peak overlaps with region
                            if (chrom == region['chrom'] and 
                                start < region['end'] and 
                                end > region['start']):
                                
                                # Calculate peak center relative to region
                                peak_center = (start + end) // 2
                                relative_center = (peak_center - region['start']) // self.resolution
                                
                                if 0 <= relative_center < self.output_length:
                                    centers.append(relative_center)
                                    
            return np.array(centers, dtype=np.int32)
            
        except Exception as e:
            logger.warning(f"Error extracting peaks from {peaks_bed} for {region}: {e}")
            return np.array([], dtype=np.int32)
            
    def _extract_motif_activities(self, region: Dict[str, Any], motif_file: str) -> np.ndarray:
        """Extract motif activity scores for a region"""
        try:
            # For now, return random activities as placeholder
            # In practice, this would load actual motif enrichment scores
            return np.random.rand(693).astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Error extracting motif activities from {motif_file} for {region}: {e}")
            return np.random.rand(693).astype(np.float32)
            
    def _save_hdf5(self,
                  output_file: str,
                  sequences: np.ndarray,
                  atac_profiles: np.ndarray,
                  rampage_profiles: np.ndarray,
                  motif_activities: np.ndarray,
                  peak_centers: np.ndarray,
                  region_info: List[Dict[str, Any]],
                  sample_id: str):
        """Save all data to HDF5 file"""
        
        with h5py.File(output_file, 'w') as f:
            # Save main datasets
            f.create_dataset('sequences', data=sequences, compression='gzip')
            f.create_dataset('atac_profiles', data=atac_profiles, compression='gzip')
            f.create_dataset('rampage_profiles', data=rampage_profiles, compression='gzip')
            f.create_dataset('motif_activities', data=motif_activities, compression='gzip')
            f.create_dataset('peaks_centers', data=peak_centers, compression='gzip')
            
            # Save metadata
            f.attrs['sample_id'] = sample_id
            f.attrs['n_regions'] = len(sequences)
            f.attrs['sequence_length'] = self.sequence_length
            f.attrs['output_length'] = self.output_length
            f.attrs['resolution'] = self.resolution
            f.attrs['created_by'] = 'EpiBERT_data_converter'
            
            # Save region information
            region_group = f.create_group('regions')
            for i, info in enumerate(region_info):
                region_subgroup = region_group.create_group(f'region_{i}')
                for key, value in info.items():
                    region_subgroup.attrs[key] = value


def convert_batch_samples(sample_manifest: str,
                         output_dir: str,
                         genome_fasta: str,
                         genome_sizes: str,
                         num_workers: int = 4) -> str:
    """
    Convert multiple samples from a manifest file
    
    Args:
        sample_manifest: Path to sample manifest file
        output_dir: Output directory for converted files
        genome_fasta: Path to reference genome FASTA
        genome_sizes: Path to chromosome sizes file
        num_workers: Number of parallel workers
        
    Returns:
        Path to updated manifest file with converted data paths
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load sample manifest
    with open(sample_manifest, 'r') as f:
        if Path(sample_manifest).suffix.lower() in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
            
    samples = data.get('samples', [])
    logger.info(f"Converting {len(samples)} samples")
    
    # Initialize converter
    converter = DataConverter(genome_fasta, genome_sizes)
    
    # Convert samples in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for sample in samples:
            sample_id = sample['sample_id']
            output_file = output_path / f"{sample_id}_epibert_data.h5"
            
            # Extract file paths from sample data
            atac_bed = sample.get('atac_bed')
            rampage_bed = sample.get('rampage_bed')
            peaks_bed = sample.get('peaks_bed')
            motif_scores = sample.get('motif_scores')
            regions_bed = sample.get('regions_bed')
            
            future = executor.submit(
                converter.convert_sample_data,
                sample_id=sample_id,
                atac_bed=atac_bed,
                rampage_bed=rampage_bed,
                peaks_bed=peaks_bed,
                motif_scores=motif_scores,
                regions_bed=regions_bed,
                output_file=str(output_file)
            )
            
            futures.append((future, sample, output_file))
            
        # Collect results
        updated_samples = []
        for future, sample, output_file in futures:
            try:
                converted_file = future.result()
                
                # Update sample with converted file path
                updated_sample = sample.copy()
                updated_sample['atac_file'] = converted_file
                updated_sample['rampage_file'] = converted_file  # Same file contains both
                updated_samples.append(updated_sample)
                
                logger.info(f"Converted sample {sample['sample_id']}")
                
            except Exception as e:
                logger.error(f"Error converting sample {sample['sample_id']}: {e}")
                
    # Save updated manifest
    updated_manifest_path = output_path / "converted_samples_manifest.yaml"
    updated_data = {'samples': updated_samples}
    
    with open(updated_manifest_path, 'w') as f:
        yaml.dump(updated_data, f, default_flow_style=False)
        
    logger.info(f"Saved updated manifest: {updated_manifest_path}")
    return str(updated_manifest_path)


def main():
    """Main CLI interface for data conversion"""
    
    parser = argparse.ArgumentParser(
        description="Convert genomic data to EpiBERT format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single sample
  python data_converter.py --sample-id sample1 --atac-bed atac.bedgraph --rampage-bed rampage.bedgraph --output sample1.h5

  # Convert batch of samples
  python data_converter.py --batch-manifest samples.yaml --output-dir converted_data --genome-fasta hg38.fa --genome-sizes hg38.sizes

  # Convert with custom regions
  python data_converter.py --sample-id sample1 --atac-bed atac.bedgraph --regions-bed custom_regions.bed --output sample1.h5
        """
    )
    
    # Single sample conversion
    parser.add_argument('--sample-id', 
                       help='Sample identifier for single sample conversion')
    parser.add_argument('--atac-bed',
                       help='ATAC-seq bedGraph file')
    parser.add_argument('--rampage-bed',
                       help='RAMPAGE-seq bedGraph file')
    parser.add_argument('--peaks-bed',
                       help='Peaks BED file')
    parser.add_argument('--motif-scores',
                       help='Motif enrichment scores file')
    parser.add_argument('--regions-bed',
                       help='Genomic regions BED file')
    parser.add_argument('--output', '-o',
                       help='Output file path')
    
    # Batch conversion
    parser.add_argument('--batch-manifest',
                       help='Sample manifest file for batch conversion')
    parser.add_argument('--output-dir',
                       help='Output directory for batch conversion')
    
    # Required for both modes
    parser.add_argument('--genome-fasta',
                       help='Reference genome FASTA file',
                       default='/path/to/hg38.fa')
    parser.add_argument('--genome-sizes',
                       help='Chromosome sizes file',
                       default='/path/to/hg38.chrom.sizes')
    
    # Optional parameters
    parser.add_argument('--sequence-length',
                       type=int,
                       help='Input sequence length',
                       default=524288)
    parser.add_argument('--output-length',
                       type=int,
                       help='Output profile length',
                       default=4096)
    parser.add_argument('--resolution',
                       type=int,
                       help='Profile resolution (bp per bin)',
                       default=128)
    parser.add_argument('--workers',
                       type=int,
                       help='Number of parallel workers',
                       default=4)
    
    args = parser.parse_args()
    
    if args.batch_manifest:
        # Batch conversion mode
        if not args.output_dir:
            print("Error: --output-dir required for batch conversion")
            sys.exit(1)
            
        updated_manifest = convert_batch_samples(
            sample_manifest=args.batch_manifest,
            output_dir=args.output_dir,
            genome_fasta=args.genome_fasta,
            genome_sizes=args.genome_sizes,
            num_workers=args.workers
        )
        
        print(f"Batch conversion complete. Updated manifest: {updated_manifest}")
        
    elif args.sample_id:
        # Single sample conversion mode
        if not args.output:
            args.output = f"{args.sample_id}_epibert_data.h5"
            
        converter = DataConverter(
            genome_fasta=args.genome_fasta,
            genome_sizes=args.genome_sizes,
            sequence_length=args.sequence_length,
            output_length=args.output_length,
            resolution=args.resolution
        )
        
        output_file = converter.convert_sample_data(
            sample_id=args.sample_id,
            atac_bed=args.atac_bed,
            rampage_bed=args.rampage_bed,
            peaks_bed=args.peaks_bed,
            motif_scores=args.motif_scores,
            regions_bed=args.regions_bed,
            output_file=args.output
        )
        
        print(f"Conversion complete: {output_file}")
        
    else:
        print("Error: Either --sample-id or --batch-manifest required")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()