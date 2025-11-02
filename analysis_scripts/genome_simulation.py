# Genome Organization Simulation with Comprehensive Error Handling
# Based on mentor's notes for testing chromosome sequence organization

import argparse
import numpy as np
import os
import json
import logging
from typing import Tuple, List, Optional, Dict, Any
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from chunkchromatin.simulation import Simulation
    from chunkchromatin.chromosome import Chromosome
    from chunkchromatin.lamina import Lamina
    from chunkchromatin.hdf5_format import HDF5Reporter
    from chunkchromatin.simulation import EKExceedsError
    from chunkchromatin.condensate import Condensate
    import openmm as mm
    from initial_conformations import create_random_walk
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please ensure all chunkchromatin modules are properly installed")
    sys.exit(1)

class SimulationConfig:
    """Configuration class with validation and error checking"""
    
    def __init__(self):
        # Fixed simulation parameters (as per mentor's notes)
        self.N_CHROMATIN = 120
        self.N_CONDENSATES = 10
        self.DENSITY = 0.33
        self.TEMPERATURE = 300.0
        self.GAMMA = 0.05
        self.TIMESTEP = 5
        self.PLATFORM = 'CPU'
        self.RANDOM_SEED = 42
        
        # Pattern validation
        self.BLOCKS_PER_TYPE = 4
        self.MONOMERS_PER_BLOCK = 10
        self.N_TYPES = 3
        
        # Simulation control
        self.SIMULATION_BLOCKS = 200
        self.STEPS_PER_BLOCK = 1000
        self.MAX_DATA_LENGTH = 500
        
    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            assert self.N_CHROMATIN == self.BLOCKS_PER_TYPE * self.N_TYPES * self.MONOMERS_PER_BLOCK, \
                f"N_CHROMATIN ({self.N_CHROMATIN}) != {self.BLOCKS_PER_TYPE}*{self.N_TYPES}*{self.MONOMERS_PER_BLOCK}"
            
            assert self.DENSITY > 0, "Density must be positive"
            assert self.TEMPERATURE > 0, "Temperature must be positive"
            assert self.GAMMA > 0, "Gamma must be positive"
            assert self.TIMESTEP > 0, "Timestep must be positive"
            
            logger.info("Configuration validation passed")
            return True
            
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

def generate_pattern_blocks() -> np.ndarray:
    """
    Generate monomer_types array for 120 monomers with fixed block pattern:
    4 repetitions of {10*A, 10*B, 10*C} where A=0, B=1, C=2.
    
    Returns block indices as specified in mentor's notes:
    - A blocks: [0:10], [30:40], [60:70], [90:100]
    - B blocks: [10:20], [40:50], [70:80], [100:110] 
    - C blocks: [20:30], [50:60], [80:90], [110:120]
    """
    try:
        monomer_types = []
        for rep in range(4):
            monomer_types.extend([0]*10)  # A block
            monomer_types.extend([1]*10)  # B block  
            monomer_types.extend([2]*10)  # C block
        
        monomer_types = np.array(monomer_types, dtype=int)
        
        # Validation as per mentor's notes
        counts = [np.sum(monomer_types == i) for i in range(3)]
        if not all(count == 40 for count in counts):
            raise ValueError(f"Expected 40 monomers per type, got: {counts}")
        
        logger.info(f"Generated pattern blocks - A: {counts[0]}, B: {counts[1]}, C: {counts[2]}")
        return monomer_types
        
    except Exception as e:
        logger.error(f"Error generating pattern blocks: {e}")
        raise

def get_condition_parameters(condition: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return interaction matrix and condensate_eps for specified condition.
    Implements all conditions from mentor's notes (C0-C5).
    """
    conditions = {
        'C0': {  # All-moderate control
            'interaction_matrix': np.array([
                [0.20, 0.20, 0.20],
                [0.20, 0.20, 0.20], 
                [0.20, 0.20, 0.20]
            ]),
            'condensate_eps': np.array([0.20, 0.20, 0.20])
        },
        'C1': {  # Condensate-driven Type-0 clustering
            'interaction_matrix': np.array([
                [0.20, 0.20, 0.20],
                [0.20, 0.20, 0.20],
                [0.20, 0.20, 0.20]
            ]),
            'condensate_eps': np.array([0.50, 0.20, 0.20])
        },
        'C2': {  # Chromatin-driven Type-0 clustering
            'interaction_matrix': np.array([
                [0.50, 0.20, 0.20],
                [0.20, 0.20, 0.20],
                [0.20, 0.20, 0.20]
            ]),
            'condensate_eps': np.array([0.20, 0.20, 0.20])
        },
        'C3': {  # Competition: chromatin favors Type-0, condensates favor Type-1
            'interaction_matrix': np.array([
                [0.50, 0.20, 0.20],
                [0.20, 0.20, 0.20],
                [0.20, 0.20, 0.20]
            ]),
            'condensate_eps': np.array([0.20, 0.50, 0.20])
        },
        'C4': {  # Condensate "excludes" Type-0 by preferring 1 and 2
            'interaction_matrix': np.array([
                [0.20, 0.20, 0.20],
                [0.20, 0.20, 0.20],
                [0.20, 0.20, 0.20]
            ]),
            'condensate_eps': np.array([0.10, 0.45, 0.45])
        },
        'C5': {  # Strong like-likes-like + condensate neutral
            'interaction_matrix': np.array([
                [0.45, 0.15, 0.15],
                [0.15, 0.45, 0.15],
                [0.15, 0.15, 0.45]
            ]),
            'condensate_eps': np.array([0.20, 0.20, 0.20])
        }
    }
    
    if condition not in conditions:
        raise ValueError(f"Unknown condition: {condition}. Available: {list(conditions.keys())}")
    
    params = conditions[condition]
    logger.info(f"Using condition {condition}")
    return params['interaction_matrix'], params['condensate_eps']

def generate_condensate_positions(n_condensates: int, chromatin_positions: np.ndarray, 
                                box_length: float, min_dist: float = 1.0, seed: int = 42) -> np.ndarray:
    """Generate condensate positions with overlap checking"""
    try:
        np.random.seed(seed)
        condensate_positions = []
        max_attempts = 1000
        
        for i in range(n_condensates):
            attempts = 0
            while attempts < max_attempts:
                trial_pos = np.random.uniform(0, box_length, size=3)
                
                # Check distance from chromatin
                dists_chromatin = np.linalg.norm(chromatin_positions - trial_pos, axis=1)
                
                # Check distance from existing condensates
                if len(condensate_positions) > 0:
                    dists_condensates = np.linalg.norm(
                        np.array(condensate_positions) - trial_pos, axis=1
                    )
                    if not np.all(dists_condensates > min_dist):
                        attempts += 1
                        continue
                
                if np.all(dists_chromatin > min_dist):
                    condensate_positions.append(trial_pos)
                    break
                
                attempts += 1
            
            if attempts >= max_attempts:
                logger.warning(f"Could not place condensate {i} after {max_attempts} attempts")
                # Use a fallback position
                fallback_pos = np.array([i * box_length / n_condensates] * 3)
                condensate_positions.append(fallback_pos)
        
        result = np.array(condensate_positions)
        logger.info(f"Generated {len(result)} condensate positions")
        return result
        
    except Exception as e:
        logger.error(f"Error generating condensate positions: {e}")
        raise

def setup_chromatin_only_simulation(config: SimulationConfig, condition: str, 
                                   output_dir: str) -> Tuple[Simulation, Any]:
    """Setup simulation for chromatin-only case"""
    try:
        # Generate pattern
        monomer_types = generate_pattern_blocks()
        np.save(os.path.join(output_dir, "monomer_types_chr.npy"), monomer_types)
        
        # Get condition parameters
        interaction_matrix, _ = get_condition_parameters(condition)
        
        # Setup simulation
        box_length = (config.N_CHROMATIN / config.DENSITY) ** (1/3)
        chains = [(0, 60, False), (60, 120, False)]  # Two chains as in original
        
        # Create reporter
        reporter = HDF5Reporter(folder=output_dir, max_data_length=config.MAX_DATA_LENGTH, overwrite=True)
        
        # Create simulation
        sim = Simulation(
            integrator_type="variableLangevin",
            temperature=config.TEMPERATURE,
            gamma=config.GAMMA,
            timestep=config.TIMESTEP,
            platform=config.PLATFORM,
            N=config.N_CHROMATIN,
            reporter=reporter
        )
        
        # Create chromosome and lamina
        chromosome = Chromosome(config.N_CHROMATIN, chains, sim)
        lamina = Lamina(config.N_CHROMATIN, chains, sim)
        
        # Initial positions
        monomer_positions = create_random_walk(step_size=1, N=config.N_CHROMATIN)
        sim.set_positions(monomer_positions)
        
        # Add forces
        harmonic_bond_force = chromosome.add_harmonic_bond()
        angle_force = chromosome.add_angle_force()
        nonbonded_pair_potential_force = chromosome.add_nonbonded_pair_potential(
            sim, interaction_matrix, monomer_types
        )
        spherical_confinement_force = lamina.add_spherical_confinement(sim)
        
        sim.add_force(harmonic_bond_force)
        sim.add_force(angle_force)
        sim.add_force(nonbonded_pair_potential_force)
        sim.add_force(spherical_confinement_force)
        
        # Finalize
        sim.create_context()
        sim.set_velocities()
        
        logger.info("Chromatin-only simulation setup complete")
        return sim, reporter
        
    except Exception as e:
        logger.error(f"Error setting up chromatin-only simulation: {e}")
        raise

def setup_condensate_simulation(config: SimulationConfig, condition: str, 
                              output_dir: str) -> Tuple[Simulation, Any]:
    """Setup simulation with condensates"""
    try:
        # Generate pattern  
        monomer_types = generate_pattern_blocks()
        np.save(os.path.join(output_dir, "monomer_types_cond.npy"), monomer_types)
        
        # Get condition parameters
        interaction_matrix, condensate_eps = get_condition_parameters(condition)
        
        # Setup parameters
        N = config.N_CHROMATIN + config.N_CONDENSATES
        box_length = (N / config.DENSITY) ** (1/3)
        chains = [(0, config.N_CHROMATIN, False)]  # Single chain
        
        # Condensate setup
        np.random.seed(config.RANDOM_SEED)
        condensate_types = np.random.choice([0, 1], size=config.N_CONDENSATES, replace=True)
        chromatin_types_full = np.concatenate([monomer_types, [-1] * config.N_CONDENSATES])
        
        # Interaction matrices for condensates
        epsilon_cc = np.array([[1.0, 0.8], [0.8, 1.2]])
        
        # Convert condensate_eps to proper format
        epsilon_cchr = np.zeros((2, 3))  # 2 condensate types, 3 chromatin types
        for i in range(2):
            epsilon_cchr[i, :] = condensate_eps
        
        # Create reporter
        reporter = HDF5Reporter(folder=output_dir, max_data_length=config.MAX_DATA_LENGTH, overwrite=True)
        
        # Create simulation
        sim = Simulation(
            integrator_type="variableLangevin",
            temperature=config.TEMPERATURE,
            gamma=config.GAMMA,
            timestep=config.TIMESTEP,
            platform=config.PLATFORM,
            N=N,
            reporter=reporter
        )
        
        # Create objects
        chromosome = Chromosome(N, chains, sim)
        lamina = Lamina(N, chains, sim)
        condensate = Condensate(
            N=N,
            chains=chains,
            simulation=sim,
            condensate_types=condensate_types,
            chromatin_types=chromatin_types_full,
            epsilon_cc=epsilon_cc,
            epsilon_cchr=epsilon_cchr,
            cutoff=3.0,
            alpha=6
        )
        
        # Initial positions
        chromatin_positions = create_random_walk(step_size=1, N=config.N_CHROMATIN)
        condensate_positions = generate_condensate_positions(
            config.N_CONDENSATES, chromatin_positions, box_length, min_dist=1.0, seed=config.RANDOM_SEED
        )
        
        positions = np.vstack([chromatin_positions, condensate_positions])
        sim.set_positions(positions)
        
        # Add forces
        harmonic_bond_force = chromosome.add_harmonic_bond()
        angle_force = chromosome.add_angle_force()
        nonbonded_pair_potential_force = chromosome.add_nonbonded_pair_potential(
            sim, interaction_matrix, chromatin_types_full
        )
        spherical_confinement_force = lamina.add_spherical_confinement(sim)
        
        sim.add_force(harmonic_bond_force)
        sim.add_force(angle_force)
        sim.add_force(nonbonded_pair_potential_force)
        sim.add_force(spherical_confinement_force)
        
        # Finalize
        sim.create_context()
        sim.set_velocities()
        
        logger.info("Condensate simulation setup complete")
        return sim, reporter
        
    except Exception as e:
        logger.error(f"Error setting up condensate simulation: {e}")
        raise

def run_simulation_with_error_handling(sim: Simulation, reporter: Any, config: SimulationConfig, 
                                     output_dir: str) -> bool:
    """Run simulation with comprehensive error handling"""
    try:
        stats_file = os.path.join(output_dir, "simulation_stats.txt")
        error_file = os.path.join(output_dir, "simulation_errors.txt")
        
        # Initialize stats file
        with open(stats_file, "w") as f:
            f.write(f"Simulation started\n")
            f.write(f"Blocks: {config.SIMULATION_BLOCKS}, Steps per block: {config.STEPS_PER_BLOCK}\n\n")
        
        successful_blocks = 0
        
        for block in range(config.SIMULATION_BLOCKS):
            try:
                logger.info(f"Running block {block+1}/{config.SIMULATION_BLOCKS}")
                
                # Run simulation block
                sim.run_simulation_block(config.STEPS_PER_BLOCK)
                
                # Write stats
                with open(stats_file, "a") as f:
                    stats = str(sim.print_stats())
                    f.write(f"Block {block+1}: {stats}\n")
                
                # Dump data
                reporter.dump_data()
                successful_blocks += 1
                
            except EKExceedsError as e:
                error_msg = f"Block {block+1}: EKExceeds error - {e}"
                logger.warning(error_msg)
                with open(error_file, "a") as f:
                    f.write(f"{error_msg}\n")
                
                # Try to continue with adjusted parameters
                try:
                    sim.set_velocities()  # Reset velocities
                    logger.info(f"Reset velocities, continuing...")
                except Exception as reset_e:
                    logger.error(f"Failed to reset velocities: {reset_e}")
                    break
                    
            except Exception as e:
                error_msg = f"Block {block+1}: Unexpected error - {e}"
                logger.error(error_msg)
                with open(error_file, "a") as f:
                    f.write(f"{error_msg}\n")
                break
        
        # Final summary
        success_rate = successful_blocks / config.SIMULATION_BLOCKS * 100
        logger.info(f"Simulation completed: {successful_blocks}/{config.SIMULATION_BLOCKS} blocks successful ({success_rate:.1f}%)")
        
        with open(stats_file, "a") as f:
            f.write(f"\nSimulation Summary:\n")
            f.write(f"Successful blocks: {successful_blocks}/{config.SIMULATION_BLOCKS} ({success_rate:.1f}%)\n")
        
        return successful_blocks > 0
        
    except Exception as e:
        logger.error(f"Critical error during simulation: {e}")
        return False

def main():
    """Main execution function with command line interface"""
    parser = argparse.ArgumentParser(description="Genome Organization Simulation")
    parser.add_argument("--condition", "-c", default="C0", 
                       choices=["C0", "C1", "C2", "C3", "C4", "C5"],
                       help="Simulation condition (default: C0)")
    parser.add_argument("--mode", "-m", default="chromatin", 
                       choices=["chromatin", "condensate"],
                       help="Simulation mode (default: chromatin)")
    parser.add_argument("--output", "-o", default=None,
                       help="Output directory (default: auto-generated)")
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        config = SimulationConfig()
        if not config.validate():
            logger.error("Configuration validation failed")
            return 1
        
        # Setup output directory
        if args.output is None:
            output_dir = f"genome_sim_{args.mode}_{args.condition}"
        else:
            output_dir = args.output
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration
        config_dict = {
            'condition': args.condition,
            'mode': args.mode,
            'N_CHROMATIN': config.N_CHROMATIN,
            'N_CONDENSATES': config.N_CONDENSATES if args.mode == 'condensate' else 0,
            'DENSITY': config.DENSITY,
            'TEMPERATURE': config.TEMPERATURE,
            'RANDOM_SEED': config.RANDOM_SEED
        }
        
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Setup and run simulation
        logger.info(f"Starting {args.mode} simulation with condition {args.condition}")
        
        if args.mode == "chromatin":
            sim, reporter = setup_chromatin_only_simulation(config, args.condition, output_dir)
        else:
            sim, reporter = setup_condensate_simulation(config, args.condition, output_dir)
        
        success = run_simulation_with_error_handling(sim, reporter, config, output_dir)
        
        if success:
            logger.info(f"Simulation completed successfully. Results in: {output_dir}")
            return 0
        else:
            logger.error("Simulation failed")
            return 1
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())