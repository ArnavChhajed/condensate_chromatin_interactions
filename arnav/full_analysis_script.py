# Analysis script for genome organization simulation results
# Implements all metrics from mentor's notes: COM distances, contact fractions, clustering scores

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import os
import json
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenomeAnalyzer:
    """Analyzer for genome organization simulation results"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.config = self._load_config()
        self.monomer_types = self._load_monomer_types()
        self.block_indices = self._get_block_indices()
        self.positions = None
        self.n_frames = 0
        
    def _load_config(self) -> Dict:
        """Load simulation configuration"""
        try:
            config_path = os.path.join(self.data_dir, "config.json")
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("No config.json found, using defaults")
            return {"N_CHROMATIN": 120, "mode": "chromatin"}
    
    def _load_monomer_types(self) -> np.ndarray:
        """Load monomer types from saved file"""
        try:
            # Try condensate file first, then chromatin
            for filename in ["monomer_types_cond.npy", "monomer_types_chr.npy"]:
                filepath = os.path.join(self.data_dir, filename)
                if os.path.exists(filepath):
                    types = np.load(filepath)
                    logger.info(f"Loaded monomer types from {filename}")
                    return types
            
            # Fallback: generate pattern
            logger.warning("No monomer types file found, generating pattern")
            return self._generate_pattern_blocks()
            
        except Exception as e:
            logger.error(f"Error loading monomer types: {e}")
            return self._generate_pattern_blocks()
    
    def _generate_pattern_blocks(self) -> np.ndarray:
        """Generate the pattern blocks as fallback"""
        monomer_types = []
        for _ in range(4):
            monomer_types.extend([0]*10)  # A block
            monomer_types.extend([1]*10)  # B block
            monomer_types.extend([2]*10)  # C block
        return np.array(monomer_types, dtype=int)
    
    def _get_block_indices(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Get block indices as specified in mentor's notes:
        - A blocks: [0:10], [30:40], [60:70], [90:100]
        - B blocks: [10:20], [40:50], [70:80], [100:110] 
        - C blocks: [20:30], [50:60], [80:90], [110:120]
        """
        block_indices = {
            'A': [(0, 10), (30, 40), (60, 70), (90, 100)],
            'B': [(10, 20), (40, 50), (70, 80), (100, 110)],
            'C': [(20, 30), (50, 60), (80, 90), (110, 120)]
        }
        return block_indices
    
    def load_trajectory(self) -> bool:
        """Load trajectory data from HDF5 files"""
        try:
            trajectory_files = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]
            if not trajectory_files:
                logger.error("No HDF5 trajectory files found")
                return False
            
            logger.info(f"Found {len(trajectory_files)} HDF5 files")
            
            # Sort files by block number to ensure correct order
            def get_block_number(filename):
                # Extract block number from filename like "blocks_97-97.h5"
                try:
                    return int(filename.split('_')[1].split('-')[0])
                except:
                    return 0
            
            trajectory_files.sort(key=get_block_number)
            logger.info(f"Loading trajectory from {len(trajectory_files)} files")
            
            # Load all frames from all files
            all_positions = []
            
            for i, filename in enumerate(trajectory_files):
                filepath = os.path.join(self.data_dir, filename)
                
                try:
                    with h5py.File(filepath, 'r') as f:
                        # Get the block number (group name)
                        group_names = list(f.keys())
                        if not group_names:
                            logger.warning(f"No groups found in {filename}")
                            continue
                        
                        block_group = group_names[0]  # Should be the block number
                        
                        if 'pos' in f[block_group]:
                            positions = f[block_group]['pos'][:]
                            all_positions.append(positions)
                            
                            if i < 5:  # Log first few for debugging
                                logger.info(f"Loaded frame {i} from {filename}: shape {positions.shape}")
                        else:
                            logger.warning(f"No 'pos' dataset found in group {block_group} of {filename}")
                            
                except Exception as e:
                    logger.warning(f"Error loading {filename}: {e}")
                    continue
            
            if not all_positions:
                logger.error("No valid trajectory data found")
                return False
            
            # Convert to numpy array: (n_frames, n_particles, 3)
            self.positions = np.array(all_positions)
            self.n_frames = self.positions.shape[0]
            
            logger.info(f"Successfully loaded {self.n_frames} frames")
            logger.info(f"Trajectory shape: {self.positions.shape}")
            
            # Verify we have the expected number of particles
            n_particles = self.positions.shape[1]
            expected_chromatin = self.config.get('N_CHROMATIN', 120)
            
            if n_particles != expected_chromatin:
                logger.warning(f"Expected {expected_chromatin} chromatin particles, got {n_particles}")
            
            return True
                    
        except Exception as e:
            logger.error(f"Error loading trajectory: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def compute_block_coms(self) -> Dict[str, np.ndarray]:
        """
        Compute center of mass for each block per frame.
        Returns dict with keys 'A', 'B', 'C' and values as (n_frames, n_blocks_per_type, 3) arrays
        """
        try:
            if self.positions is None:
                raise ValueError("No trajectory data loaded")
            
            block_coms = {}
            n_chromatin = self.config.get('N_CHROMATIN', 120)
            
            for block_type, indices_list in self.block_indices.items():
                # Each type has 4 blocks
                type_coms = np.zeros((self.n_frames, len(indices_list), 3))
                
                for frame in range(self.n_frames):
                    chromatin_positions = self.positions[frame, :n_chromatin, :]
                    
                    for block_idx, (start, end) in enumerate(indices_list):
                        block_positions = chromatin_positions[start:end, :]
                        type_coms[frame, block_idx, :] = np.mean(block_positions, axis=0)
                
                block_coms[block_type] = type_coms
                logger.info(f"Computed COMs for {len(indices_list)} blocks of type {block_type}")
            
            return block_coms
            
        except Exception as e:
            logger.error(f"Error computing block COMs: {e}")
            return {}
    
    def compute_type_to_type_distances(self, block_coms: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Compute COM distances between all pairs of blocks, aggregated by type.
        Follows mentor's specification: "For each pair of blocks (p,q) compute d_pq(frame) = ||COM_p - COM_q||"
        Then aggregate by type labels and average across frames.
        """
        try:
            type_pairs = [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'B'), ('B', 'C'), ('C', 'C')]
            type_distances = {}
            
            for type1, type2 in type_pairs:
                distances_per_frame = []
                
                for frame in range(self.n_frames):
                    frame_distances = []
                    
                    # Get all pairwise distances between blocks of these types
                    coms1 = block_coms[type1][frame]  # (n_blocks, 3)
                    coms2 = block_coms[type2][frame]  # (n_blocks, 3)
                    
                    for i, com1 in enumerate(coms1):
                        for j, com2 in enumerate(coms2):
                            if type1 == type2 and i >= j:
                                continue  # Avoid double counting and self-distances for same type
                            
                            distance = np.linalg.norm(com1 - com2)
                            frame_distances.append(distance)
                    
                    if frame_distances:
                        # Average all pairwise distances for this type pair in this frame
                        distances_per_frame.append(np.mean(frame_distances))
                    else:
                        distances_per_frame.append(0.0)
                
                type_distances[(type1, type2)] = np.array(distances_per_frame)
                mean_dist = np.mean(distances_per_frame)
                std_dist = np.std(distances_per_frame)
                logger.info(f"Type-to-type distances {type1}-{type2}: {mean_dist:.3f} ± {std_dist:.3f}")
            
            return type_distances
            
        except Exception as e:
            logger.error(f"Error computing type-to-type distances: {e}")
            return {}
    
    def compute_contact_fractions(self, cutoff: float = 2.0) -> Dict[Tuple[str, str], float]:
        """
        Compute contact fractions between chromatin types using cutoff distance.
        """
        try:
            if self.positions is None:
                raise ValueError("No trajectory data loaded")
            
            n_chromatin = self.config.get('N_CHROMATIN', 120)
            type_pairs = [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'B'), ('B', 'C'), ('C', 'C')]
            type_to_indices = {
                'A': np.where(self.monomer_types == 0)[0],
                'B': np.where(self.monomer_types == 1)[0], 
                'C': np.where(self.monomer_types == 2)[0]
            }
            
            contact_fractions = {}
            
            for type1, type2 in type_pairs:
                indices1 = type_to_indices[type1]
                indices2 = type_to_indices[type2]
                
                total_contacts = 0
                total_pairs = 0
                
                for frame in range(self.n_frames):
                    chromatin_positions = self.positions[frame, :n_chromatin, :]
                    
                    for i in indices1:
                        for j in indices2:
                            if type1 == type2 and i >= j:
                                continue  # Avoid double counting
                            
                            distance = np.linalg.norm(chromatin_positions[i] - chromatin_positions[j])
                            if distance < cutoff:
                                total_contacts += 1
                            total_pairs += 1
                
                contact_fraction = total_contacts / total_pairs if total_pairs > 0 else 0.0
                contact_fractions[(type1, type2)] = contact_fraction
                logger.info(f"Contact fraction {type1}-{type2}: {contact_fraction:.4f}")
            
            return contact_fractions
            
        except Exception as e:
            logger.error(f"Error computing contact fractions: {e}")
            return {}
    
    def compute_clustering_scores(self, k: int = 10) -> Dict[str, float]:
        """
        Compute clustering score for each type: fraction of monomers whose k nearest neighbors 
        are also the same type.
        """
        try:
            if self.positions is None:
                raise ValueError("No trajectory data loaded")
            
            n_chromatin = self.config.get('N_CHROMATIN', 120)
            type_to_indices = {
                'A': np.where(self.monomer_types == 0)[0],
                'B': np.where(self.monomer_types == 1)[0], 
                'C': np.where(self.monomer_types == 2)[0]
            }
            
            clustering_scores = {}
            
            for type_name, type_indices in type_to_indices.items():
                type_scores = []
                
                for frame in range(self.n_frames):
                    chromatin_positions = self.positions[frame, :n_chromatin, :]
                    frame_scores = []
                    
                    for monomer_idx in type_indices:
                        # Compute distances to all other monomers
                        distances = np.linalg.norm(
                            chromatin_positions - chromatin_positions[monomer_idx], axis=1
                        )
                        
                        # Get k nearest neighbors (excluding self)
                        neighbor_indices = np.argsort(distances)[1:k+1]
                        neighbor_types = self.monomer_types[neighbor_indices]
                        
                        # Count how many neighbors are the same type
                        same_type_count = np.sum(neighbor_types == self.monomer_types[monomer_idx])
                        score = same_type_count / k
                        frame_scores.append(score)
                    
                    type_scores.append(np.mean(frame_scores))
                
                clustering_scores[type_name] = np.mean(type_scores)
                logger.info(f"Clustering score for type {type_name}: {clustering_scores[type_name]:.4f}")
            
            return clustering_scores
            
        except Exception as e:
            logger.error(f"Error computing clustering scores: {e}")
            return {}
    
    def compute_condensate_recruitment(self) -> Optional[Dict[str, float]]:
        """
        Compute average nearest-condensate distance per monomer type.
        Only applicable for condensate simulations.
        """
        try:
            if self.config.get('mode') != 'condensate':
                logger.info("Skipping condensate recruitment - not a condensate simulation")
                return None
            
            if self.positions is None:
                raise ValueError("No trajectory data loaded")
            
            n_chromatin = self.config.get('N_CHROMATIN', 120)
            n_condensates = self.config.get('N_CONDENSATES', 10)
            
            if self.positions.shape[1] < n_chromatin + n_condensates:
                logger.warning("Not enough particles for condensate analysis")
                return None
            
            type_to_indices = {
                'A': np.where(self.monomer_types == 0)[0],
                'B': np.where(self.monomer_types == 1)[0], 
                'C': np.where(self.monomer_types == 2)[0]
            }
            
            recruitment_distances = {}
            
            for type_name, type_indices in type_to_indices.items():
                type_distances = []
                
                for frame in range(self.n_frames):
                    chromatin_positions = self.positions[frame, :n_chromatin, :]
                    condensate_positions = self.positions[frame, n_chromatin:n_chromatin+n_condensates, :]
                    
                    frame_distances = []
                    for monomer_idx in type_indices:
                        # Find nearest condensate
                        distances_to_condensates = np.linalg.norm(
                            condensate_positions - chromatin_positions[monomer_idx], axis=1
                        )
                        nearest_distance = np.min(distances_to_condensates)
                        frame_distances.append(nearest_distance)
                    
                    type_distances.append(np.mean(frame_distances))
                
                recruitment_distances[type_name] = np.mean(type_distances)
                logger.info(f"Average nearest-condensate distance for type {type_name}: {recruitment_distances[type_name]:.4f}")
            
            return recruitment_distances
            
        except Exception as e:
            logger.error(f"Error computing condensate recruitment: {e}")
            return None
    
    def plot_distance_matrix(self, type_distances: Dict[Tuple[str, str], np.ndarray], 
                           save_path: Optional[str] = None) -> None:
        """Plot 3x3 matrix of mean COM distances with error bars"""
        try:
            # Create 3x3 matrix
            types = ['A', 'B', 'C']
            distance_matrix = np.zeros((3, 3))
            error_matrix = np.zeros((3, 3))
            
            for i, type1 in enumerate(types):
                for j, type2 in enumerate(types):
                    if i <= j:  # Upper triangle and diagonal
                        key = (type1, type2)
                    else:  # Lower triangle - use symmetry
                        key = (type2, type1)
                    
                    if key in type_distances:
                        distances = type_distances[key]
                        distance_matrix[i, j] = np.mean(distances)
                        distance_matrix[j, i] = np.mean(distances)  # Symmetry
                        error_matrix[i, j] = np.std(distances)
                        error_matrix[j, i] = np.std(distances)
            
            # Create plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Heatmap
            im = ax.imshow(distance_matrix, cmap='viridis', aspect='equal')
            
            # Add text annotations with error bars
            for i in range(3):
                for j in range(3):
                    mean_val = distance_matrix[i, j]
                    error_val = error_matrix[i, j]
                    text = f'{mean_val:.2f}\n±{error_val:.2f}'
                    ax.text(j, i, text, ha='center', va='center', 
                           color='white' if mean_val > distance_matrix.max()/2 else 'black')
            
            # Labels and formatting
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(types)
            ax.set_yticklabels(types)
            ax.set_xlabel('Monomer Type')
            ax.set_ylabel('Monomer Type')
            ax.set_title('Mean COM Distances Between Types')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Distance')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Distance matrix plot saved to {save_path}")
            else:
                plt.savefig(os.path.join(self.data_dir, 'distance_matrix.png'), 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting distance matrix: {e}")
    
    def generate_full_report(self, save_plots: bool = True) -> Dict:
        """Generate complete analysis report with all metrics"""
        try:
            logger.info("Starting full analysis report generation")
            
            # Load trajectory
            if not self.load_trajectory():
                logger.error("Failed to load trajectory data")
                return {}
            
            # Compute all metrics
            results = {
                'config': self.config,
                'n_frames': self.n_frames,
                'monomer_counts': {
                    'A': np.sum(self.monomer_types == 0),
                    'B': np.sum(self.monomer_types == 1), 
                    'C': np.sum(self.monomer_types == 2)
                }
            }
            
            # Block COMs
            logger.info("Computing block center of mass...")
            block_coms = self.compute_block_coms()
            
            # Type-to-type distances
            logger.info("Computing type-to-type distances...")
            type_distances = self.compute_type_to_type_distances(block_coms)
            results['com_distances'] = {
                f"{k[0]}-{k[1]}": {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                for k, v in type_distances.items()
            }
            
            # Contact fractions
            logger.info("Computing contact fractions...")
            contact_fractions = self.compute_contact_fractions()
            results['contact_fractions'] = {
                f"{k[0]}-{k[1]}": float(v) for k, v in contact_fractions.items()
            }
            
            # Clustering scores
            logger.info("Computing clustering scores...")
            clustering_scores = self.compute_clustering_scores()
            results['clustering_scores'] = {k: float(v) for k, v in clustering_scores.items()}
            
            # Condensate recruitment (if applicable)
            recruitment_distances = self.compute_condensate_recruitment()
            if recruitment_distances:
                results['condensate_recruitment'] = {k: float(v) for k, v in recruitment_distances.items()}
            
            # Generate plots
            if save_plots and type_distances:
                self.plot_distance_matrix(type_distances)
            
            # Save results (convert numpy types to native Python types)
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            results_clean = convert_numpy_types(results)
            
            results_file = os.path.join(self.data_dir, 'analysis_results.json')
            with open(results_file, 'w') as f:
                json.dump(results_clean, f, indent=2)
                
            # Also save a human-readable summary
            summary_file = os.path.join(self.data_dir, 'analysis_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("GENOME ORGANIZATION ANALYSIS SUMMARY\n")
                f.write("="*50 + "\n\n")
                f.write(f"Configuration: {results['config']}\n")
                f.write(f"Frames analyzed: {results['n_frames']}\n")
                f.write(f"Monomer counts: {results['monomer_counts']}\n\n")
                
                f.write("COM DISTANCES (mean ± std):\n")
                f.write("-" * 30 + "\n")
                for pair, stats in results['com_distances'].items():
                    f.write(f"{pair:8s}: {stats['mean']:.3f} ± {stats['std']:.3f}\n")
                
                f.write("\nCONTACT FRACTIONS:\n")
                f.write("-" * 30 + "\n")
                for pair, fraction in results['contact_fractions'].items():
                    f.write(f"{pair:8s}: {fraction:.4f}\n")
                
                f.write("\nCLUSTERING SCORES:\n")
                f.write("-" * 30 + "\n")
                for type_name, score in results['clustering_scores'].items():
                    f.write(f"Type {type_name}: {score:.4f}\n")
                
                if 'condensate_recruitment' in results:
                    f.write("\nCONDENSATE RECRUITMENT:\n")
                    f.write("-" * 30 + "\n")
                    for type_name, distance in results['condensate_recruitment'].items():
                        f.write(f"Type {type_name}: {distance:.4f}\n")
            
            logger.info(f"Human-readable summary saved to {summary_file}")
            
            logger.info(f"Analysis complete! Results saved to {results_file}")
            return results
            
        except Exception as e:
            logger.error(f"Error generating full report: {e}")
            return {}

    def debug_hdf5_files(self) -> None:
        """Debug function to inspect HDF5 file contents"""
        try:
            trajectory_files = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]
            logger.info(f"Found {len(trajectory_files)} HDF5 files: {trajectory_files}")
            
            for filename in trajectory_files:
                filepath = os.path.join(self.data_dir, filename)
                logger.info(f"\nInspecting {filename}:")
                
                with h5py.File(filepath, 'r') as f:
                    def print_structure(name, obj):
                        indent = "  " * name.count('/')
                        if isinstance(obj, h5py.Dataset):
                            logger.info(f"{indent}{name}: Dataset, shape={obj.shape}, dtype={obj.dtype}")
                        elif isinstance(obj, h5py.Group):
                            logger.info(f"{indent}{name}: Group")
                    
                    f.visititems(print_structure)
                    
        except Exception as e:
            logger.error(f"Error debugging HDF5 files: {e}")

def main():
    """Command line interface for analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze genome organization simulation results")
    parser.add_argument("data_dir", help="Directory containing simulation results")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--debug", action="store_true", help="Debug HDF5 file structure")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1
    
    try:
        analyzer = GenomeAnalyzer(args.data_dir)
        
        if args.debug:
            analyzer.debug_hdf5_files()
            return 0
        
        results = analyzer.generate_full_report(save_plots=not args.no_plots)
        
        if results:
            print("\n=== ANALYSIS SUMMARY ===")
            print(f"Configuration: {results['config']}")
            print(f"Frames analyzed: {results['n_frames']}")
            print(f"Monomer counts: {results['monomer_counts']}")
            
            print("\nCOM Distances:")
            for pair, stats in results['com_distances'].items():
                print(f"  {pair}: {stats['mean']:.3f} ± {stats['std']:.3f}")
            
            print("\nContact Fractions:")
            for pair, fraction in results['contact_fractions'].items():
                print(f"  {pair}: {fraction:.4f}")
            
            print("\nClustering Scores:")
            for type_name, score in results['clustering_scores'].items():
                print(f"  Type {type_name}: {score:.4f}")
            
            if 'condensate_recruitment' in results:
                print("\nCondensate Recruitment (nearest distance):")
                for type_name, distance in results['condensate_recruitment'].items():
                    print(f"  Type {type_name}: {distance:.4f}")
            
            return 0
        else:
            logger.error("Analysis failed")
            return 1
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())