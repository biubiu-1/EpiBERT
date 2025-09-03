#!/usr/bin/env python3
"""
Batch Data Processing Utilities for EpiBERT

Handles batch processing of multiple ATAC-seq and RAMPAGE-seq samples,
including parallel processing, sample validation, and data conversion.
"""

import os
import sys
import argparse
import yaml
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from dataclasses import dataclass
import shutil
import subprocess
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProcessingJob:
    """Represents a single sample processing job"""
    sample_id: str
    condition: str
    cell_type: str
    atac_bam: Optional[str] = None
    rampage_bam: Optional[str] = None
    output_dir: str = ""
    config: Optional[Dict[str, Any]] = None


class BatchDataProcessor:
    """
    Batch processor for multiple EpiBERT samples
    
    Features:
    - Parallel processing of multiple samples
    - Automatic sample pairing and validation
    - Progress tracking and error handling
    - Configurable processing pipelines
    """
    
    def __init__(self, 
                 base_config: str,
                 output_dir: str,
                 num_workers: int = 4,
                 overwrite: bool = False):
        """
        Initialize batch processor
        
        Args:
            base_config: Path to base configuration file
            output_dir: Base output directory for processed data
            num_workers: Number of parallel workers
            overwrite: Whether to overwrite existing outputs
        """
        self.base_config = Path(base_config)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.overwrite = overwrite
        
        # Load base configuration
        self.config = self._load_config()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging for batch processing
        self._setup_batch_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load base configuration file"""
        if not self.base_config.exists():
            raise FileNotFoundError(f"Base config file not found: {self.base_config}")
            
        with open(self.base_config, 'r') as f:
            if self.base_config.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif self.base_config.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {self.base_config.suffix}")
                
    def _setup_batch_logging(self):
        """Setup logging for batch processing"""
        log_file = self.output_dir / "batch_processing.log"
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
    def process_sample_manifest(self, 
                               manifest_file: str,
                               pipeline_steps: List[str] = None) -> Dict[str, Any]:
        """
        Process all samples defined in a manifest file
        
        Args:
            manifest_file: Path to sample manifest
            pipeline_steps: List of processing steps to run
            
        Returns:
            Dictionary with processing results and statistics
        """
        
        manifest_path = Path(manifest_file)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            
        # Load sample manifest
        samples = self._load_sample_manifest(manifest_path)
        logger.info(f"Loaded {len(samples)} samples from manifest")
        
        # Create processing jobs
        jobs = self._create_processing_jobs(samples, pipeline_steps or ['all'])
        logger.info(f"Created {len(jobs)} processing jobs")
        
        # Execute jobs in parallel
        results = self._execute_jobs_parallel(jobs)
        
        # Create summary
        summary = self._create_processing_summary(results)
        
        # Save results
        self._save_processing_results(results, summary)
        
        return summary
        
    def process_sample_list(self,
                           sample_list: List[Dict[str, Any]],
                           pipeline_steps: List[str] = None) -> Dict[str, Any]:
        """
        Process a list of sample dictionaries
        
        Args:
            sample_list: List of sample dictionaries
            pipeline_steps: List of processing steps to run
            
        Returns:
            Dictionary with processing results and statistics
        """
        
        # Create temporary manifest
        temp_manifest = self.output_dir / "temp_manifest.yaml"
        manifest_data = {'samples': sample_list}
        
        with open(temp_manifest, 'w') as f:
            yaml.dump(manifest_data, f)
            
        try:
            return self.process_sample_manifest(str(temp_manifest), pipeline_steps)
        finally:
            if temp_manifest.exists():
                temp_manifest.unlink()
                
    def _load_sample_manifest(self, manifest_path: Path) -> List[Dict[str, Any]]:
        """Load samples from manifest file"""
        
        with open(manifest_path, 'r') as f:
            if manifest_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
                if 'samples' in data:
                    return data['samples']
                else:
                    return [data]  # Single sample
            elif manifest_path.suffix.lower() == '.csv':
                df = pd.read_csv(manifest_path)
                return df.to_dict('records')
            elif manifest_path.suffix.lower() == '.json':
                data = json.load(f)
                if 'samples' in data:
                    return data['samples']
                else:
                    return [data]
        
        return []
        
    def _create_processing_jobs(self, 
                               samples: List[Dict[str, Any]], 
                               pipeline_steps: List[str]) -> List[ProcessingJob]:
        """Create processing jobs from sample list"""
        
        jobs = []
        
        for sample_data in samples:
            # Create sample-specific output directory
            sample_output_dir = self.output_dir / sample_data.get('sample_id', f"sample_{len(jobs)}")
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create sample-specific config
            sample_config = self.config.copy()
            sample_config.update(sample_data)
            sample_config['pipeline_steps'] = pipeline_steps
            sample_config['output_dir'] = str(sample_output_dir)
            
            job = ProcessingJob(
                sample_id=sample_data.get('sample_id', f"sample_{len(jobs)}"),
                condition=sample_data.get('condition', 'unknown'),
                cell_type=sample_data.get('cell_type', 'unknown'),
                atac_bam=sample_data.get('atac_bam'),
                rampage_bam=sample_data.get('rampage_bam'),
                output_dir=str(sample_output_dir),
                config=sample_config
            )
            
            jobs.append(job)
            
        return jobs
        
    def _execute_jobs_parallel(self, jobs: List[ProcessingJob]) -> List[Dict[str, Any]]:
        """Execute processing jobs in parallel"""
        
        logger.info(f"Starting parallel processing with {self.num_workers} workers")
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all jobs
            future_to_job = {executor.submit(self._process_single_sample, job): job for job in jobs}
            
            # Collect results as they complete
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed processing for sample {job.sample_id}")
                except Exception as e:
                    error_result = {
                        'sample_id': job.sample_id,
                        'success': False,
                        'error': str(e),
                        'output_dir': job.output_dir
                    }
                    results.append(error_result)
                    logger.error(f"Error processing sample {job.sample_id}: {e}")
                    
        return results
        
    def _process_single_sample(self, job: ProcessingJob) -> Dict[str, Any]:
        """Process a single sample"""
        
        try:
            # Create sample-specific config file
            config_file = Path(job.output_dir) / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(job.config, f)
                
            # Run data processing pipeline
            pipeline_script = Path(__file__).parent.parent / "data_processing" / "run_pipeline.sh"
            
            if not pipeline_script.exists():
                raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")
                
            # Prepare command
            cmd = [
                str(pipeline_script),
                "-c", str(config_file),
                "-o", job.output_dir,
                "-j", "1"  # Single thread per job since we're running multiple jobs
            ]
            
            # Add specific steps if requested
            if 'pipeline_steps' in job.config and job.config['pipeline_steps'] != ['all']:
                cmd.extend(["-s", ",".join(job.config['pipeline_steps'])])
                
            # Execute pipeline
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per sample
            )
            
            # Process results
            success = result.returncode == 0
            
            # Save logs
            log_file = Path(job.output_dir) / "processing.log"
            with open(log_file, 'w') as f:
                f.write(f"STDOUT:\n{result.stdout}\n\n")
                f.write(f"STDERR:\n{result.stderr}\n")
                
            # Collect output files
            output_files = self._collect_output_files(job.output_dir)
            
            return {
                'sample_id': job.sample_id,
                'condition': job.condition,
                'cell_type': job.cell_type,
                'success': success,
                'output_dir': job.output_dir,
                'output_files': output_files,
                'log_file': str(log_file),
                'config_file': str(config_file),
                'return_code': result.returncode,
                'error': result.stderr if not success else None
            }
            
        except Exception as e:
            return {
                'sample_id': job.sample_id,
                'condition': job.condition,
                'cell_type': job.cell_type,
                'success': False,
                'error': str(e),
                'output_dir': job.output_dir
            }
            
    def _collect_output_files(self, output_dir: str) -> Dict[str, List[str]]:
        """Collect output files from processing"""
        
        output_path = Path(output_dir)
        output_files = {
            'atac': [],
            'rampage': [],
            'peaks': [],
            'motifs': [],
            'processed_data': []
        }
        
        # Look for ATAC files
        atac_dir = output_path / "atac"
        if atac_dir.exists():
            output_files['atac'] = [str(f) for f in atac_dir.glob("*.bed*")]
            
        # Look for RAMPAGE files
        rampage_dir = output_path / "rampage"
        if rampage_dir.exists():
            output_files['rampage'] = [str(f) for f in rampage_dir.glob("*.bed*")]
            
        # Look for peak files
        peaks_dir = output_path / "peaks"
        if peaks_dir.exists():
            output_files['peaks'] = [str(f) for f in peaks_dir.glob("*.bed")]
            
        # Look for motif files
        motifs_dir = output_path / "motifs"
        if motifs_dir.exists():
            output_files['motifs'] = [str(f) for f in motifs_dir.glob("*.txt")]
            
        # Look for processed data files
        for ext in ['*.h5', '*.npz', '*.pkl']:
            output_files['processed_data'].extend([str(f) for f in output_path.glob(ext)])
            
        return output_files
        
    def _create_processing_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of processing results"""
        
        total_samples = len(results)
        successful_samples = sum(1 for r in results if r.get('success', False))
        failed_samples = total_samples - successful_samples
        
        # Group by condition and cell type
        conditions = {}
        cell_types = {}
        
        for result in results:
            condition = result.get('condition', 'unknown')
            cell_type = result.get('cell_type', 'unknown')
            
            if condition not in conditions:
                conditions[condition] = {'total': 0, 'success': 0, 'failed': 0}
            if cell_type not in cell_types:
                cell_types[cell_type] = {'total': 0, 'success': 0, 'failed': 0}
                
            conditions[condition]['total'] += 1
            cell_types[cell_type]['total'] += 1
            
            if result.get('success', False):
                conditions[condition]['success'] += 1
                cell_types[cell_type]['success'] += 1
            else:
                conditions[condition]['failed'] += 1
                cell_types[cell_type]['failed'] += 1
                
        # Collect errors
        errors = [r.get('error') for r in results if r.get('error')]
        
        summary = {
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'failed_samples': failed_samples,
            'success_rate': successful_samples / total_samples if total_samples > 0 else 0,
            'conditions': conditions,
            'cell_types': cell_types,
            'errors': errors,
            'output_dir': str(self.output_dir)
        }
        
        return summary
        
    def _save_processing_results(self, 
                                results: List[Dict[str, Any]], 
                                summary: Dict[str, Any]):
        """Save processing results and summary"""
        
        # Save detailed results
        results_file = self.output_dir / "processing_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save summary
        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Save as CSV for easy viewing
        results_df = pd.DataFrame(results)
        csv_file = self.output_dir / "processing_results.csv"
        results_df.to_csv(csv_file, index=False)
        
        logger.info(f"Saved processing results to {results_file}")
        logger.info(f"Saved processing summary to {summary_file}")
        
    def create_training_manifest(self, 
                                processing_results: List[Dict[str, Any]],
                                output_file: str) -> str:
        """
        Create a training manifest from processing results
        
        Args:
            processing_results: Results from batch processing
            output_file: Path for output manifest file
            
        Returns:
            Path to created manifest file
        """
        
        training_samples = []
        
        for result in processing_results:
            if not result.get('success', False):
                continue
                
            sample_data = {
                'sample_id': result['sample_id'],
                'condition': result['condition'],
                'cell_type': result['cell_type'],
                'output_dir': result['output_dir']
            }
            
            # Find ATAC and RAMPAGE files
            output_files = result.get('output_files', {})
            
            # Look for processed data files first
            processed_files = output_files.get('processed_data', [])
            atac_files = [f for f in processed_files if 'atac' in f.lower()]
            rampage_files = [f for f in processed_files if 'rampage' in f.lower()]
            
            if atac_files:
                sample_data['atac_file'] = atac_files[0]
            if rampage_files:
                sample_data['rampage_file'] = rampage_files[0]
                
            # Add to training samples
            training_samples.append(sample_data)
            
        # Create manifest
        manifest_data = {'samples': training_samples}
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            yaml.dump(manifest_data, f, default_flow_style=False)
            
        logger.info(f"Created training manifest with {len(training_samples)} samples: {output_path}")
        return str(output_path)


def create_example_batch_config() -> str:
    """Create an example batch processing configuration"""
    
    config = {
        'genome_fasta': '/path/to/hg38.fa',
        'genome_sizes': '/path/to/hg38.chrom.sizes',
        'blacklist': '/path/to/hg38.blacklist.bed',
        'motif_database': '/path/to/motifs.meme',
        'background_peaks': '/path/to/background.bed',
        'threads': 4,
        'rampage_scale_factor': 2.0
    }
    
    config_file = "/tmp/example_batch_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    return config_file


def create_example_sample_manifest() -> str:
    """Create an example sample manifest for batch processing"""
    
    samples = [
        {
            'sample_id': 'sample1_condition1',
            'condition': 'condition1',
            'cell_type': 'K562',
            'atac_bam': '/path/to/sample1_atac.bam',
            'rampage_bam': '/path/to/sample1_rampage.bam',
            'replicate': 1
        },
        {
            'sample_id': 'sample2_condition1',
            'condition': 'condition1',
            'cell_type': 'K562',
            'atac_bam': '/path/to/sample2_atac.bam',
            'rampage_bam': '/path/to/sample2_rampage.bam',
            'replicate': 2
        },
        {
            'sample_id': 'sample1_condition2',
            'condition': 'condition2',
            'cell_type': 'GM12878',
            'atac_bam': '/path/to/sample3_atac.bam',
            'rampage_bam': '/path/to/sample3_rampage.bam',
            'replicate': 1
        }
    ]
    
    manifest_data = {'samples': samples}
    manifest_file = "/tmp/example_sample_manifest.yaml"
    
    with open(manifest_file, 'w') as f:
        yaml.dump(manifest_data, f, default_flow_style=False)
        
    return manifest_file


def main():
    """Main CLI interface for batch data processing"""
    
    parser = argparse.ArgumentParser(
        description="Batch Data Processing for EpiBERT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process samples from manifest
  python batch_data_processor.py --config base_config.yaml --manifest samples.yaml --output /path/to/output

  # Process with specific pipeline steps
  python batch_data_processor.py --config base_config.yaml --manifest samples.yaml --output /path/to/output --steps atac,rampage

  # Create example files
  python batch_data_processor.py --create-examples

  # Generate training manifest from processing results
  python batch_data_processor.py --create-training-manifest /path/to/processing_results.json --output training_manifest.yaml
        """
    )
    
    parser.add_argument('--config', '-c', 
                       help='Base configuration file')
    parser.add_argument('--manifest', '-m',
                       help='Sample manifest file')
    parser.add_argument('--output', '-o',
                       help='Output directory',
                       default='./batch_processed_data')
    parser.add_argument('--workers', '-w',
                       type=int,
                       help='Number of parallel workers',
                       default=4)
    parser.add_argument('--steps', '-s',
                       help='Comma-separated list of pipeline steps',
                       default='all')
    parser.add_argument('--overwrite',
                       action='store_true',
                       help='Overwrite existing outputs')
    parser.add_argument('--create-examples',
                       action='store_true',
                       help='Create example configuration and manifest files')
    parser.add_argument('--create-training-manifest',
                       help='Create training manifest from processing results JSON file')
    
    args = parser.parse_args()
    
    if args.create_examples:
        print("Creating example files...")
        config_file = create_example_batch_config()
        manifest_file = create_example_sample_manifest()
        print(f"Example config: {config_file}")
        print(f"Example manifest: {manifest_file}")
        return
        
    if args.create_training_manifest:
        if not args.output:
            print("Error: --output required for training manifest creation")
            sys.exit(1)
            
        with open(args.create_training_manifest, 'r') as f:
            results = json.load(f)
            
        # Create temporary processor for manifest creation
        temp_config = create_example_batch_config()
        processor = BatchDataProcessor(temp_config, "/tmp")
        manifest_path = processor.create_training_manifest(results, args.output)
        print(f"Created training manifest: {manifest_path}")
        return
        
    if not args.config or not args.manifest:
        print("Error: Both --config and --manifest are required")
        parser.print_help()
        sys.exit(1)
        
    # Parse pipeline steps
    pipeline_steps = [step.strip() for step in args.steps.split(',')]
    
    # Initialize processor
    processor = BatchDataProcessor(
        base_config=args.config,
        output_dir=args.output,
        num_workers=args.workers,
        overwrite=args.overwrite
    )
    
    # Process samples
    print(f"Starting batch processing with {args.workers} workers...")
    summary = processor.process_sample_manifest(args.manifest, pipeline_steps)
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Successful: {summary['successful_samples']}")
    print(f"Failed: {summary['failed_samples']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Output directory: {summary['output_dir']}")
    
    if summary['failed_samples'] > 0:
        print(f"\nErrors encountered:")
        for error in summary['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(summary['errors']) > 5:
            print(f"  ... and {len(summary['errors']) - 5} more")
            
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()