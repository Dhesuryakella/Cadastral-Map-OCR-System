import numpy as np
from pathlib import Path
import json
from model_evaluation import ModelEvaluator
from comprehensive_detector import ComprehensiveMapDetector
from tqdm import tqdm
import optuna
import logging

class ParameterTuner:
    def __init__(self):
        self.evaluator = ModelEvaluator()
        self.test_images = list(Path('datasets/images').glob('*.jpg'))[:5]  # Use subset for tuning
        self.ground_truth_file = 'datasets/symbols/annotations.json'
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('parameter_tuning.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def objective(self, trial):
        """Optuna objective function for parameter optimization"""
        # Define parameter search space
        params = {
            'text_confidence_threshold': trial.suggest_float('text_confidence_threshold', 0.3, 0.8),
            'symbol_match_threshold': trial.suggest_float('symbol_match_threshold', 0.6, 0.9),
            'line_detection_params': {
                'rho': trial.suggest_int('rho', 1, 3),
                'theta': np.pi/trial.suggest_int('theta_div', 120, 240),
                'threshold': trial.suggest_int('threshold', 30, 70),
                'minLineLength': trial.suggest_int('minLineLength', 50, 150),
                'maxLineGap': trial.suggest_int('maxLineGap', 5, 20)
            }
        }
        
        # Update detector parameters
        self.evaluator.detector.text_confidence_threshold = params['text_confidence_threshold']
        self.evaluator.detector.symbol_match_threshold = params['symbol_match_threshold']
        self.evaluator.detector.line_detection_params = params['line_detection_params']
        
        # Evaluate with current parameters
        results = self.evaluator.evaluate_model(self.test_images, self.ground_truth_file)
        
        # Calculate average F1 scores
        avg_f1_scores = {
            'text': np.mean(self.evaluator.metrics['text_detection']['f1_score']),
            'symbol': np.mean(self.evaluator.metrics['symbol_detection']['f1_score']),
            'boundary': np.mean(self.evaluator.metrics['boundary_detection']['f1_score'])
        }
        
        # Log current trial results
        self.logger.info(f"Trial {trial.number}:")
        self.logger.info(f"Parameters: {params}")
        self.logger.info(f"Average F1 Scores: {avg_f1_scores}")
        
        # Return weighted average of F1 scores
        return (avg_f1_scores['text'] * 0.4 + 
                avg_f1_scores['symbol'] * 0.3 + 
                avg_f1_scores['boundary'] * 0.3)
    
    def tune_parameters(self, n_trials=50):
        """Run parameter tuning"""
        self.logger.info("Starting parameter tuning...")
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Log best results
        self.logger.info("\nTuning Complete!")
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best weighted F1 score: {best_value:.3f}")
        
        # Save best parameters
        self.save_best_parameters(best_params, best_value)
        
        return best_params, best_value
    
    def save_best_parameters(self, params, score):
        """Save best parameters to file"""
        output_file = Path('output/best_parameters.json')
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Add derived parameters
        full_params = {
            'parameters': params,
            'performance': {
                'weighted_f1_score': score
            },
            'parameter_ranges': {
                'text_confidence_threshold': [0.3, 0.8],
                'symbol_match_threshold': [0.6, 0.9],
                'line_detection': {
                    'rho': [1, 3],
                    'theta_div': [120, 240],
                    'threshold': [30, 70],
                    'minLineLength': [50, 150],
                    'maxLineGap': [5, 20]
                }
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(full_params, f, indent=2)
        
        self.logger.info(f"Saved best parameters to {output_file}")

def main():
    # Initialize tuner
    tuner = ParameterTuner()
    
    # Run parameter tuning
    best_params, best_score = tuner.tune_parameters(n_trials=50)
    
    print("\nParameter Tuning Complete!")
    print(f"\nBest Parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"\nBest Weighted F1 Score: {best_score:.3f}")

if __name__ == "__main__":
    main()