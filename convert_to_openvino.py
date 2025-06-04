import os
import argparse
import openvino as ov


def convert_models(onnx_path, openvino_path, compress_fp16=True):
    """Convert ONNX models to OpenVINO format with directory structure"""
    # Create output directories
    diffusion_ov_path = os.path.join(openvino_path, 'diffusion_head_openvino')
    evo_ov_path = os.path.join(openvino_path, 'evo_vino')
    confidence_ov_path = os.path.join(openvino_path, 'confidence_vino')

    os.makedirs(diffusion_ov_path, exist_ok=True)
    os.makedirs(evo_ov_path, exist_ok=True)
    os.makedirs(confidence_ov_path, exist_ok=True)

    # Construct input paths
    diffusion_onnx = os.path.join(onnx_path, 'diffusion_head_onnx', 'diffusion_head.onnx')
    evo_onnx = os.path.join(onnx_path, 'evo_onnx', 'evoformer.onnx')
    confidence_onnx = os.path.join(onnx_path, 'confidence_onnx', 'confidence_head.onnx')

    # Conversion workflow
    try:
        print(f"[1/3] Converting Evoformer: {evo_onnx}")
        evo_vino = ov.convert_model(evo_onnx)
        ov.save_model(evo_vino, os.path.join(evo_ov_path, "model.xml"), compress_to_fp16=compress_fp16)
        print(f"  ✓ Saved to: {evo_ov_path}")

        print(f"[2/3] Converting Diffusion: {diffusion_onnx}")
        diffusion_vino = ov.convert_model(diffusion_onnx)
        ov.save_model(diffusion_vino, os.path.join(diffusion_ov_path, "model.xml"), compress_to_fp16=compress_fp16)
        print(f"  ✓ Saved to: {diffusion_ov_path}")

        print(f"[3/3] Converting Confidence: {confidence_onnx}")
        confidence_vino = ov.convert_model(confidence_onnx)
        ov.save_model(confidence_vino, os.path.join(confidence_ov_path, "model.xml"), compress_to_fp16=compress_fp16)
        print(f"  ✓ Saved to: {confidence_ov_path}")

        print("\n✅ All models converted successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Conversion failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Setup command-line arguments
    parser = argparse.ArgumentParser(
        description='AlphaFold3 ONNX to OpenVINO Converter',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--onnx_path', type=str, default='/root/asc25',
                        help='Base directory containing ONNX models')
    parser.add_argument('--output_path', type=str, default='/root/asc25',
                        help='Output directory for OpenVINO models')
    parser.add_argument('--no_fp16', action='store_false', dest='compress_fp16',
                        help='Disable FP16 compression (use full precision)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose conversion logging')

    args = parser.parse_args()

    # Configure OpenVINO logging
    ov_log_level = ov.log.Level.DEBUG if args.verbose else ov.log.Level.ERROR
    ov.set_log_level(ov_log_level)

    print("\n" + "=" * 50)
    print(f"AlphaFold3 ONNX → OpenVINO Conversion")
    print("=" * 50)
    print(f"• ONNX source: {args.onnx_path}")
    print(f"• OpenVINO target: {args.output_path}")
    print(f"• FP16 compression: {'Enabled' if args.compress_fp16 else 'Disabled'}")
    print("=" * 50 + "\n")

    # Run conversion
    success = convert_models(
        onnx_path=args.onnx_path,
        openvino_path=args.output_path,
        compress_fp16=args.compress_fp16
    )

    exit(0 if success else 1)
