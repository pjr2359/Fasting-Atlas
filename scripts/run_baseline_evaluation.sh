#!/bin/bash

# Baseline evaluation: Single-pass extraction (1 sample)
# Compares against 3-sample conservative configuration

set -e

PAPERS_DIR="papers"
OUTPUT_DIR="results/evaluation/run_baseline_single_sample"
mkdir -p "$OUTPUT_DIR"

echo "=== Baseline Evaluation: Single-Sample Extraction ==="
echo "Output directory: $OUTPUT_DIR"
echo ""

# List of papers
papers=(
  "1-s2.0-0026049582900245-main.pdf"
  "1-s2.0-0026049589900474-main.pdf"
  "jcem0698.pdf"
  "jcem3646.pdf"
  "jcem4776.pdf"
  "maccario-et-al-2000-short-term-fasting-abolishes-the-sex-related-difference-in-gh-and-leptin-secretion-in-humans.pdf"
  "mittendorfer-et-al-2001-gender-differences-in-lipid-and-glucose-kinetics-during-short-term-fasting.pdf"
  "pone.0200817.pdf"
  "young-women-partition-fatty-acids-towards-ketone-body-production-rather-than-vldl-tag-synthesis-compared-with-young-men.pdf"
)

for paper in "${papers[@]}"; do
  paper_path="$PAPERS_DIR/$paper"
  output_file="$OUTPUT_DIR/${paper%.*}.json"
  
  echo "Processing: $paper"
  start_time=$(date +%s)
  
  python3 -m fasting_atlas parse "$paper_path" \
    --consensus-samples 1 \
    --consensus-temperature 0.6 \
    --output "$output_file" \
    || echo "Warning: Failed on $paper, continuing..."
  
  end_time=$(date +%s)
  elapsed=$((end_time - start_time))
  echo "  ✓ Completed in ${elapsed}s"
  echo ""
done

echo "=== Baseline Evaluation Complete ==="
echo "Results saved to: $OUTPUT_DIR"
