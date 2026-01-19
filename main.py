import os
import json
import argparse
# from Ocr import GeminiBanknoteVerifier
from gemini_verifier import GeminiBanknoteVerifier

from resnet_placeholder import ResNetSegmentor
from utils.image_processing import create_directory_structure
from config import Config

def main():
    # Check for configuration errors first
    config_errors = Config.validate_config()
    if config_errors:
        print("❌ Configuration errors detected. Please fix your .env file:")
        for error in config_errors:
            print(f"   - {error}")
        return
    
    parser = argparse.ArgumentParser(description='Indian Banknote Verification System')
    parser.add_argument('--image_path', type=str, required=True, help='Path to banknote image')
    parser.add_argument('--denomination', type=int, required=True, 
                       choices=[10, 20, 50, 100, 200, 500],
                       help='Banknote denomination (10,20,50,100,200,500)')
    parser.add_argument('--api_key', type=str, help='Gemini API key (optional if set in .env)')
    
    args = parser.parse_args()
    
    # Use API key from args or config
    api_key = args.api_key or Config.GEMINI_API_KEY
    if not api_key:
        print("❌ No Gemini API key provided. Use --api_key or set GEMINI_API_KEY in .env file")
        return
    
    # Create directory structure
    create_directory_structure()
    print(ResNetSegmentor)
    print("🪙 Indian Banknote Verification System")
    print("=" * 50)
    print(f"Image: {args.image_path}")
    print(f"Denomination: ₹{args.denomination}")
    print("=" * 50)
    
    try:
        # Initialize verifier
        verifier = GeminiBanknoteVerifier(api_key=api_key)
        
        # Perform verification
        result = verifier.verify_banknote(
            image_path=args.image_path,
            denomination=args.denomination
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("📊 VERIFICATION RESULTS")
        print("=" * 50)
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        if result['verdict'] == 'REAL':
            print("✅ Banknote appears to be GENUINE")
        elif result['verdict'] == 'FAKE':
            print("❌ Banknote appears to be COUNTERFEIT")
        else:
            print("⚠️  Banknote requires MANUAL INSPECTION")
        
        # Print feature analysis
        print("\n🔍 FEATURE ANALYSIS:")
        for feature, details in result['feature_details'].items():
            if details['matching_score'] >= Config.STRONG_MATCH_THRESHOLD:
                status = "✅"
            elif details['matching_score'] >= Config.WEAK_MATCH_THRESHOLD:
                status = "⚠️"
            else:
                status = "❌"
            print(f"  {status} {feature}: {details['matching_score']:.2f} - {details['explanation']}")
        
        # Serial validation
        serial_status = "✅" if result['serial_validation']['pass'] else "❌"
        print(f"\n🔢 SERIAL VALIDATION: {serial_status}")
        print(f"   Left: {result['serial_validation'].get('left_serial', 'N/A')}")
        print(f"   Right: {result['serial_validation'].get('right_serial', 'N/A')}")
        
        # Failed features
        if result['failed_features']:
            print(f"\n❌ FAILED FEATURES: {', '.join(result['failed_features'])}")
        
        # Manual inspection suggestions
        if result['manual_inspection_suggestions']:
            print(f"\n🔎 MANUAL INSPECTION SUGGESTIONS:")
            for suggestion in result['manual_inspection_suggestions']:
                print(f"   • {suggestion}")
        
        # Save detailed report
        output_file = f"verification_report_{os.path.basename(args.image_path)}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during verification: {str(e)}")
        if Config.DEBUG:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()