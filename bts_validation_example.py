"""
Example integration script for BTS zero-shot validation with HT-HGNN.

This script demonstrates how to use the BTS data loader with your actual
trained HT-HGNN model for journal submission validation.

Usage:
1. Download BTS data from https://www.transtats.bts.gov/DL_SelectFields.aspx
2. Train your HT-HGNN model on IndiGo data and save checkpoint
3. Run this script for zero-shot validation on US aviation data

The results provide external validation for journal submission.
"""

from bts_data_loader import run_zero_shot_validation
import torch
import json
from datetime import datetime


def run_bts_validation_for_journal():
    """
    Run BTS validation for journal submission.

    This function loads your trained HT-HGNN model and evaluates it
    on completely unseen US aviation data from BTS.
    """

    print("="*80)
    print("BTS ZERO-SHOT VALIDATION FOR JOURNAL SUBMISSION")
    print("="*80)

    # Configuration
    config = {
        'model_checkpoint': 'outputs/models/ht_hgnn_best_model.pth',  # Your trained model
        'bts_csv_path': 'data/bts_flight_data.csv',                  # Downloaded BTS data
        'carrier': 'AA',                                             # American Airlines
        'start_date': '2023-12-20',                                  # Validation period
        'end_date': '2023-12-26',                                    # Holiday travel week
        'output_dir': 'outputs/bts_validation/'
    }

    # Alternative carriers and time periods for comprehensive evaluation
    test_scenarios = [
        {'carrier': 'AA', 'start': '2023-12-20', 'end': '2023-12-26', 'description': 'American Airlines - Holiday Week'},
        {'carrier': 'UA', 'start': '2023-12-20', 'end': '2023-12-26', 'description': 'United Airlines - Holiday Week'},
        {'carrier': 'DL', 'start': '2023-11-15', 'end': '2023-11-21', 'description': 'Delta Airlines - Normal Operations'},
        {'carrier': 'WN', 'start': '2024-01-08', 'end': '2024-01-14', 'description': 'Southwest Airlines - Post-Holiday'},
    ]

    all_results = []

    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario['description']}")
        print(f"{'='*60}")

        try:
            # Run zero-shot validation
            results = run_zero_shot_validation(
                model_checkpoint_path=config['model_checkpoint'],
                bts_csv_path=config['bts_csv_path'],
                carrier=scenario['carrier'],
                start=scenario['start'],
                end=scenario['end']
            )

            # Add scenario metadata
            results.update({
                'scenario_description': scenario['description'],
                'validation_timestamp': datetime.now().isoformat()
            })

            all_results.append(results)

            print(f"✓ {scenario['description']} completed successfully")
            print(f"  Accuracy: {results.get('crit_acc', 0):.4f}")
            print(f"  Macro F1: {results.get('macro_f1', 0):.4f}")

        except Exception as e:
            print(f"✗ {scenario['description']} failed: {e}")
            error_result = {
                'scenario_description': scenario['description'],
                'error': str(e),
                'validation_timestamp': datetime.now().isoformat()
            }
            all_results.append(error_result)

    # Aggregate results
    print(f"\n{'='*80}")
    print("COMPREHENSIVE BTS VALIDATION SUMMARY")
    print(f"{'='*80}")

    successful_results = [r for r in all_results if 'error' not in r]

    if successful_results:
        # Compute aggregate metrics
        avg_accuracy = sum(r['crit_acc'] for r in successful_results) / len(successful_results)
        avg_macro_f1 = sum(r['macro_f1'] for r in successful_results) / len(successful_results)
        total_flights = sum(r['num_flights'] for r in successful_results)

        print(f"Successful validations: {len(successful_results)}/{len(test_scenarios)}")
        print(f"Total flights evaluated: {total_flights:,}")
        print(f"Average accuracy: {avg_accuracy:.4f}")
        print(f"Average macro F1: {avg_macro_f1:.4f}")

        # Save comprehensive results
        output_file = 'bts_validation_comprehensive_results.json'
        summary = {
            'validation_date': datetime.now().isoformat(),
            'model_checkpoint': config['model_checkpoint'],
            'aggregate_metrics': {
                'avg_accuracy': avg_accuracy,
                'avg_macro_f1': avg_macro_f1,
                'total_flights': total_flights,
                'successful_scenarios': len(successful_results)
            },
            'individual_results': all_results
        }

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nComprehensive results saved to: {output_file}")

        # Journal submission summary
        print(f"\n{'='*80}")
        print("JOURNAL SUBMISSION SUMMARY")
        print(f"{'='*80}")
        print("External Validation Results:")
        print(f"• Dataset: US Bureau of Transportation Statistics (BTS)")
        print(f"• Period: Holiday 2023 and normal operations")
        print(f"• Airlines: American, United, Delta, Southwest")
        print(f"• Total flights: {total_flights:,}")
        print(f"• Zero-shot accuracy: {avg_accuracy:.4f}")
        print(f"• Zero-shot macro F1: {avg_macro_f1:.4f}")
        print("\nThis demonstrates generalization of HT-HGNN from IndiGo")
        print("aviation data to broader US commercial aviation operations.")

        return summary

    else:
        print("No successful validations completed.")
        return None


def validate_real_bts_data_single():
    """
    Simple wrapper for single BTS validation.

    Use this function when you have:
    1. A trained HT-HGNN model saved as checkpoint
    2. Downloaded BTS CSV data
    """

    # EXAMPLE USAGE:
    # Replace these paths with your actual files
    model_path = "outputs/checkpoints/ht_hgnn_final.pth"
    bts_csv = "data/bts_december_2023.csv"

    print("Running single BTS validation...")

    results = run_zero_shot_validation(
        model_checkpoint_path=model_path,
        bts_csv_path=bts_csv,
        carrier='AA',
        start='2023-12-20',
        end='2023-12-26'
    )

    print(f"Zero-shot validation results:")
    print(f"  Accuracy: {results.get('crit_acc', 0):.4f}")
    print(f"  Macro F1: {results.get('macro_f1', 0):.4f}")
    print(f"  Flights processed: {results.get('num_flights', 0):,}")

    return results


def main():
    """Main function for BTS validation scenarios."""
    print("BTS Validation Integration Examples")
    print("="*50)

    print("\nAvailable validation options:")
    print("1. Single BTS validation (basic)")
    print("2. Comprehensive multi-carrier validation (journal)")

    # For demonstration, show the comprehensive validation structure
    print("\nDemonstrating comprehensive validation structure...")
    print("(Replace model_checkpoint and bts_csv_path with your actual files)")

    # Note: Uncomment the line below when you have real model and data
    # results = run_bts_validation_for_journal()

    print("\nTo run real validation:")
    print("1. Train your HT-HGNN model and save checkpoint")
    print("2. Download BTS data from https://www.transtats.bts.gov/DL_SelectFields.aspx")
    print("3. Update file paths in validate_real_bts_data_single()")
    print("4. Run validation for journal submission")


if __name__ == "__main__":
    main()