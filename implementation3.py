import os
import json
import numpy as np
from datetime import datetime
from planetary_computer import sign
from pystac_client import Client
import rasterio
from rasterio.enums import Resampling
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import cv2
from collections import defaultdict
import google.generativeai as genai
import base64
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

class SatelliteChangeAnalyzer:
    def __init__(self, location_name, bbox, collection="naip", years=[2010, 2023], output_dir="satellite_analysis", gemini_api_key=None):
        self.location_name = location_name
        self.bbox = bbox
        self.collection = collection
        self.years = years
        self.output_dir = output_dir
        self.images = {}
        self.metadata = {}
        self.gemini_api_key = gemini_api_key
        
        # Configure Gemini if API key provided
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def download_imagery(self):
        """Download satellite imagery for specified years"""
        print("🔗 Connecting to Microsoft Planetary Computer STAC...")
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        
        for year in self.years:
            start_date = f"{year}-01-01T00:00:00Z"
            end_date = f"{year}-12-31T23:59:59Z"
            
            search = catalog.search(
                collections=[self.collection],
                bbox=self.bbox,
                datetime=f"{start_date}/{end_date}",
                max_items=1
            )
            
            items = list(search.items())
            if not items:
                print(f"❌ No {self.collection.upper()} image found for {year}")
                continue
                
            item = items[0]
            
            # Handle different asset keys for different collections
            asset_key = None
            if "image" in item.assets:
                asset_key = "image"
            elif "cog_default" in item.assets:
                asset_key = "cog_default"
            elif "data" in item.assets:
                asset_key = "data"
            else:
                # Find the first asset that looks like an image
                for key, asset in item.assets.items():
                    if asset.media_type and ("image" in asset.media_type.lower() or "tiff" in asset.media_type.lower()):
                        asset_key = key
                        break
            
            if not asset_key:
                print(f"❌ No suitable image asset found for {year}. Available assets: {list(item.assets.keys())}")
                continue
                
            print(f"📡 Using asset key: {asset_key}")
            cog_url = sign(item.assets[asset_key].href)
            self.metadata[year] = item.to_dict()
            
            print(f"📡 Downloading imagery for {year}...")
            
            # Save metadata
            metadata_path = os.path.join(self.output_dir, f"{year}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata[year], f, indent=2)
            
            # Download and process image
            with rasterio.open(cog_url) as src:
                print(f"📊 Original image size: {src.width} x {src.height} pixels")
                
                # Use more aggressive scaling for faster processing
                scale_factor = 0.1  # Reduced from 0.5 for much faster processing
                new_height = int(src.height * scale_factor)
                new_width = int(src.width * scale_factor)
                
                print(f"📊 Resampling to: {new_width} x {new_height} pixels ({scale_factor:.1%} of original)")
                print("⏳ Reading and resampling image data... (this may take a moment)")
                
                # Read with windowed approach for better memory efficiency
                try:
                    # Try reading the entire image with resampling
                    data = src.read(
                        out_shape=(src.count, new_height, new_width),
                        resampling=Resampling.bilinear
                    )
                except Exception as e:
                    print(f"⚠️  Large image detected, using windowed reading approach...")
                    # Fallback: read a central window of the image
                    window_size = min(src.width, src.height, 4000)  # Max 4000px window
                    
                    # Calculate central window
                    col_start = max(0, (src.width - window_size) // 2)
                    row_start = max(0, (src.height - window_size) // 2)
                    
                    from rasterio.windows import Window
                    window = Window(col_start, row_start, 
                                  min(window_size, src.width - col_start),
                                  min(window_size, src.height - row_start))
                    
                    data = src.read(
                        window=window,
                        out_shape=(src.count, new_height, new_width),
                        resampling=Resampling.bilinear
                    )
                    print(f"✅ Used central window: {window}")
                
                print("🔄 Converting to image format...")
                img_array = np.transpose(data, (1, 2, 0))
                
                # Handle different band configurations
                if img_array.shape[2] >= 3:
                    img = Image.fromarray(img_array[:, :, :3].astype(np.uint8))  # RGB only
                else:
                    # Handle grayscale or single band
                    img = Image.fromarray(img_array[:, :, 0].astype(np.uint8)).convert('RGB')
                
                # Light enhancement for better visibility
                img = ImageEnhance.Contrast(img).enhance(1.1)
                img = ImageEnhance.Sharpness(img).enhance(1.1)
                
                # Save high-quality image
                output_path = os.path.join(self.output_dir, f"{year}_image.jpg")
                img.save(output_path, "JPEG", quality=90)  # Reduced quality for smaller files
                self.images[year] = img
                
            print(f"✅ Saved {year} image and metadata")
    
    def analyze_changes(self):
        """Perform comprehensive change analysis"""
        if len(self.images) < 2:
            print("❌ Need at least 2 images for change analysis")
            return None
            
        print("🔍 Performing comprehensive change detection...")
        
        # Standardize image sizes for analysis
        standard_size = (512, 512)  # Reduced from 1024x1024 for faster processing
        print(f"🔄 Standardizing images to {standard_size[0]}x{standard_size[1]} for analysis...")
        img1 = self.images[self.years[0]].convert("RGB").resize(standard_size, Image.Resampling.LANCZOS)
        img2 = self.images[self.years[1]].convert("RGB").resize(standard_size, Image.Resampling.LANCZOS)
        
        # Save standardized images for LLM
        img1.save(os.path.join(self.output_dir, f"{self.years[0]}_standardized.jpg"), "JPEG", quality=95)
        img2.save(os.path.join(self.output_dir, f"{self.years[1]}_standardized.jpg"), "JPEG", quality=95)
        
        # Multiple change detection methods
        change_analysis = self._perform_change_detection(img1, img2)
        
        # Generate comprehensive analysis report
        analysis_report = self._generate_analysis_summary(change_analysis)
        
        # Save analysis results
        with open(os.path.join(self.output_dir, "change_analysis.json"), "w") as f:
            json.dump(analysis_report, f, indent=2)
            
        print("✅ Change analysis complete")
        return analysis_report
    
    def _perform_change_detection(self, img1, img2):
        """Multiple change detection algorithms"""
        results = {}
        
        # Convert to numpy arrays for OpenCV processing
        img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
        img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
        
        # 1. Simple pixel difference
        diff_simple = ImageChops.difference(img1, img2)
        diff_gray = diff_simple.convert("L")
        
        # Multiple threshold levels
        thresholds = [20, 40, 60]
        for threshold in thresholds:
            diff_thresh = diff_gray.point(lambda x: 255 if x > threshold else 0)
            change_pixels = np.sum(np.array(diff_thresh) > 128)
            total_pixels = diff_thresh.size[0] * diff_thresh.size[1]
            change_percentage = (change_pixels / total_pixels) * 100
            
            results[f'threshold_{threshold}'] = {
                'changed_pixels': int(change_pixels),
                'total_pixels': int(total_pixels),
                'change_percentage': float(change_percentage)
            }
            
            # Save thresholded difference image
            diff_thresh.save(os.path.join(self.output_dir, f"diff_threshold_{threshold}.jpg"))
        
        # 2. Color-based change detection
        color_changes = self._detect_color_changes(img1, img2)
        results['color_analysis'] = color_changes
        
        # 3. Create enhanced change visualization
        change_overlay = self._create_change_overlay(img1, img2, diff_gray)
        change_overlay.save(os.path.join(self.output_dir, "change_overlay.jpg"), "JPEG", quality=95)
        
        # 4. Statistical analysis
        stats = self._calculate_image_statistics(img1, img2)
        results['statistics'] = stats
        
        return results
    
    def _detect_color_changes(self, img1, img2):
        """Analyze changes in different color channels"""
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        
        color_changes = {}
        color_names = ['red', 'green', 'blue']
        
        for i, color in enumerate(color_names):
            channel1 = img1_array[:, :, i]
            channel2 = img2_array[:, :, i]
            
            diff = np.abs(channel1.astype(float) - channel2.astype(float))
            
            color_changes[color] = {
                'mean_change': float(np.mean(diff)),
                'max_change': float(np.max(diff)),
                'std_change': float(np.std(diff)),
                'significant_change_pixels': int(np.sum(diff > 30))
            }
        
        return color_changes
    
    def _create_change_overlay(self, img1, img2, diff_gray):
        """Create an enhanced change overlay visualization"""
        # Create color-coded change map
        diff_array = np.array(diff_gray)
        
        # Create RGB overlay
        overlay = np.zeros((*diff_array.shape, 3), dtype=np.uint8)
        
        # Low changes - yellow
        low_change_mask = (diff_array > 15) & (diff_array <= 40)
        overlay[low_change_mask] = [255, 255, 0]
        
        # Medium changes - orange  
        med_change_mask = (diff_array > 40) & (diff_array <= 80)
        overlay[med_change_mask] = [255, 165, 0]
        
        # High changes - red
        high_change_mask = diff_array > 80
        overlay[high_change_mask] = [255, 0, 0]
        
        # Blend with base image
        overlay_img = Image.fromarray(overlay)
        alpha_mask = Image.fromarray(((diff_array > 15) * 128).astype(np.uint8))
        
        result = Image.composite(overlay_img, img2, alpha_mask)
        return result
    
    def _calculate_image_statistics(self, img1, img2):
        """Calculate statistical differences between images"""
        img1_array = np.array(img1).astype(float)
        img2_array = np.array(img2).astype(float)
        
        stats = {}
        
        # Overall statistics
        stats['overall'] = {
            'mean_difference': float(np.mean(np.abs(img1_array - img2_array))),
            'std_difference': float(np.std(img1_array - img2_array)),
            'correlation': float(np.corrcoef(img1_array.flatten(), img2_array.flatten())[0, 1])
        }
        
        # Per-channel statistics
        for i, channel in enumerate(['red', 'green', 'blue']):
            ch1 = img1_array[:, :, i]
            ch2 = img2_array[:, :, i]
            
            stats[channel] = {
                'mean_img1': float(np.mean(ch1)),
                'mean_img2': float(np.mean(ch2)),
                'std_img1': float(np.std(ch1)),
                'std_img2': float(np.std(ch2)),
                'correlation': float(np.corrcoef(ch1.flatten(), ch2.flatten())[0, 1])
            }
        
        return stats
    
    def _generate_analysis_summary(self, change_analysis):
        """Generate a comprehensive summary for LLM processing"""
        summary = {
            'analysis_metadata': {
                'location': self.location_name,
                'bbox': self.bbox,
                'years_compared': self.years,
                'collection': self.collection,
                'analysis_timestamp': datetime.now().isoformat(),
                'image_resolution': '512x512',
                'processing_optimizations': 'Applied aggressive scaling (10%) and windowed reading for faster processing'
            },
            'change_detection_results': change_analysis,
            'key_findings': self._extract_key_findings(change_analysis),
            'files_generated': {
                'original_images': [f"{year}_image.jpg" for year in self.years],
                'standardized_images': [f"{year}_standardized.jpg" for year in self.years],
                'change_visualizations': [
                    'change_overlay.jpg',
                    'diff_threshold_20.jpg',
                    'diff_threshold_40.jpg', 
                    'diff_threshold_60.jpg'
                ],
                'metadata': [f"{year}_metadata.json" for year in self.years],
                'analysis_report': 'change_analysis.json'
            },
            'suggested_analysis_areas': [
                'Urban development patterns',
                'Infrastructure changes',
                'Land use modifications',
                'Environmental changes',
                'Transportation network evolution'
            ]
        }
        
        return summary
    
    def _extract_key_findings(self, change_analysis):
        """Extract key findings for LLM attention"""
        findings = {}
        
        # Change magnitude assessment
        moderate_threshold = change_analysis['threshold_40']['change_percentage']
        if moderate_threshold > 10:
            findings['change_magnitude'] = 'HIGH'
        elif moderate_threshold > 5:
            findings['change_magnitude'] = 'MODERATE'
        else:
            findings['change_magnitude'] = 'LOW'
        
        # Dominant change type based on color analysis
        color_changes = change_analysis['color_analysis']
        max_change_color = max(color_changes.items(), key=lambda x: x[1]['mean_change'])
        findings['dominant_change_channel'] = max_change_color[0]
        
        # Statistical significance
        overall_correlation = change_analysis['statistics']['overall']['correlation']
        if overall_correlation < 0.7:
            findings['change_significance'] = 'HIGHLY_SIGNIFICANT'
        elif overall_correlation < 0.85:
            findings['change_significance'] = 'MODERATELY_SIGNIFICANT'
        else:
            findings['change_significance'] = 'LOW_SIGNIFICANCE'
            
        return findings
    
    def _prepare_image_for_gemini(self, image_path):
        """Convert image to format suitable for Gemini API"""
        with open(image_path, 'rb') as img_file:
            return {
                'mime_type': 'image/jpeg',
                'data': img_file.read()
            }
    
    def generate_gemini_report(self, analysis_results):
        """Generate comprehensive report using Gemini API"""
        if not self.gemini_api_key:
            print("❌ Gemini API key not provided. Skipping LLM report generation.")
            return None
            
        print("🤖 Generating comprehensive report using Gemini...")
        
        try:
            # Prepare images for Gemini
            img1_path = os.path.join(self.output_dir, f"{self.years[0]}_standardized.jpg")
            img2_path = os.path.join(self.output_dir, f"{self.years[1]}_standardized.jpg")
            change_overlay_path = os.path.join(self.output_dir, "change_overlay.jpg")
            
            img1_data = self._prepare_image_for_gemini(img1_path)
            img2_data = self._prepare_image_for_gemini(img2_path)
            change_overlay_data = self._prepare_image_for_gemini(change_overlay_path)
            
            # Prepare analysis context
            context = self._prepare_analysis_context(analysis_results)
            
            # Create comprehensive prompt


            prompt = f"""
You are a professional satellite imagery analyst specializing in urban development, land use change detection, and environmental impact assessment.

Your task is to analyze three inputs for {self.location_name}:
1. **Image 1** – Satellite imagery from {self.years[0]}
2. **Image 2** – Satellite imagery from {self.years[1]}
3. **Change Overlay** – Color-coded visualization of detected changes:
   - Yellow = Minor changes
   - Orange = Moderate changes
   - Red = Major changes

**Context for Analysis:**
{context}

---

## OUTPUT REQUIREMENTS

Produce a **professional, technical, and data-driven report** in the following structure. Use clear headings, concise language, and avoid speculation not supported by imagery or provided data.

# Satellite Imagery Change Analysis Report
## {self.location_name} ({self.years[0]} – {self.years[1]})

### 1. Executive Summary
- Provide a clear 2–3 paragraph synthesis of the most significant changes.
- Highlight major geographic areas and change types.

### 2. Methodology
- Briefly outline the analytical process, including image comparison and change detection principles.

### 3. Quantitative Analysis
Include:
- Overall change percentages by threshold (minor, moderate, major).
- Color channel change metrics.
- Statistical correlations or significant patterns from the data.

### 4. Visual Analysis & Spatial Patterns
Describe observed changes in detail:
- Infrastructure (roads, bridges, buildings, airports).
- Land use conversions (forest → urban, agricultural → industrial).
- Spatial clustering and geographic distribution.
- Specific high-change zones from the overlay.

### 5. Development Assessment
Focus on development near {self.location_name.lower()}:
- Airport infrastructure changes.
- Expansion of urban boundaries.
- Road network growth.
- Commercial/industrial zone development.

### 6. Temporal Context
Within the {self.years[1] - self.years[0]}-year period:
- Growth rate and pace of change.
- How patterns compare to typical regional urbanization.
- Economic or policy influences if directly visible.

### 7. Environmental Impact Assessment
- Loss of vegetation or green cover.
- Encroachment into natural areas.
- Urban sprawl dynamics.
- Possible impacts on waterways, wetlands, or terrain.

### 8. Conclusions & Implications
Summarize findings with implications for:
- Urban and regional planning.
- Infrastructure investment.
- Environmental conservation.
- Forecast of future change patterns.

---

**Analysis Rules:**
- Base all claims on visible evidence and/or provided statistics.
- Cross-reference overlay colors with actual spatial locations.
- Be precise about change magnitude and location.
- Maintain professional tone suitable for submission to planning authorities.
- Integrate quantitative and qualitative findings seamlessly.
"""

           

            # Generate report using Gemini
            response = self.model.generate_content([
                prompt,
                img1_data,
                img2_data, 
                change_overlay_data
            ])
            
            # Save the generated report
            report_path = os.path.join(self.output_dir, "gemini_analysis_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            
            # Also save as JSON for programmatic access
            report_data = {
                "report_text": response.text,
                "generation_timestamp": datetime.now().isoformat(),
                "model_used": "gemini-1.5-pro",
                "location": self.location_name,
                "years": self.years,
                "analysis_context": context
            }
            
            json_report_path = os.path.join(self.output_dir, "gemini_analysis_report.json")
            with open(json_report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            
            print(f"✅ Gemini report generated successfully!")
            print(f"📄 Markdown report: {report_path}")
            print(f"📄 JSON report: {json_report_path}")
            
            return response.text
            
        except Exception as e:
            print(f"❌ Error generating Gemini report: {str(e)}")
            return None
    
    def _prepare_analysis_context(self, analysis_results):
        """Prepare structured context for Gemini analysis"""
        context = f"""
LOCATION: {self.location_name}
BOUNDING BOX: {self.bbox}
YEARS COMPARED: {self.years[0]} to {self.years[1]}
SATELLITE COLLECTION: {self.collection.upper()}

QUANTITATIVE CHANGE DETECTION RESULTS:

Change Detection at Different Thresholds:
"""
        
        # Add threshold results
        for threshold in [20, 40, 60]:
            key = f'threshold_{threshold}'
            if key in analysis_results:
                data = analysis_results[key]
                context += f"- Threshold {threshold}: {data['change_percentage']:.2f}% of pixels changed ({data['changed_pixels']:,} pixels)\n"
        
        # Add color analysis
        if 'color_analysis' in analysis_results:
            context += "\nColor Channel Analysis:\n"
            for color, data in analysis_results['color_analysis'].items():
                context += f"- {color.upper()} channel: Mean change {data['mean_change']:.1f}, Max change {data['max_change']:.1f}\n"
        
        # Add statistical analysis
        if 'statistics' in analysis_results:
            stats = analysis_results['statistics']
            if 'overall' in stats:
                context += f"\nOverall Statistics:\n"
                context += f"- Mean pixel difference: {stats['overall']['mean_difference']:.2f}\n"
                context += f"- Image correlation: {stats['overall']['correlation']:.3f}\n"
        
        # Add key findings
        if 'key_findings' in analysis_results:
            findings = analysis_results['key_findings']
            context += f"\nKey Automated Findings:\n"
            context += f"- Change Magnitude: {findings.get('change_magnitude', 'N/A')}\n"
            context += f"- Dominant Change Channel: {findings.get('dominant_change_channel', 'N/A')}\n"
            context += f"- Statistical Significance: {findings.get('change_significance', 'N/A')}\n"
        
        return context

# Usage
if __name__ == "__main__":
    # Configuration
    LOCATION_NAME = "Orlando Airport Development Imagery"
    BBOX = [-81.40, 28.41, -81.27, 28.47]
    COLLECTION = "naip"
    YEARS = [2010, 2023]
    OUTPUT_DIR = "satellite_analysis"
    
    # Get Gemini API key from environment variable or prompt
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("⚠️  GEMINI_API_KEY not found in environment variables.")
        print("Please set your Gemini API key as an environment variable:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        print("Or uncomment and set it directly in the code below:")
        GEMINI_API_KEY = "your_api_key_here"  # Uncomment and add your key
    
    # Initialize analyzer
    analyzer = SatelliteChangeAnalyzer(
        location_name=LOCATION_NAME,
        bbox=BBOX,
        collection=COLLECTION,
        years=YEARS,
        output_dir=OUTPUT_DIR,
        gemini_api_key=GEMINI_API_KEY
    )
    
    # Run complete analysis pipeline
    print("🚀 Starting satellite imagery analysis pipeline...")
    
    # Step 1: Download imagery
    analyzer.download_imagery()
    
    # Step 2: Perform change detection analysis
    analysis_results = analyzer.analyze_changes()
    
    # Step 3: Generate AI report with Gemini
    if GEMINI_API_KEY and analysis_results:
        gemini_report = analyzer.generate_gemini_report(analysis_results)
        
        if gemini_report:
            print("\n🎉 Complete analysis pipeline finished successfully!")
            print(f"📁 All outputs saved to: {OUTPUT_DIR}")
            print("📋 Generated files:")
            print("   • Satellite imagery (original and standardized)")
            print("   • Change detection visualizations")
            print("   • Quantitative analysis (JSON)")
            print("   • Comprehensive AI-generated report (Markdown & JSON)")
        else:
            print("⚠️  AI report generation failed, but analysis data is available")
    else:
        print("\n📦 Analysis complete! Quantitative analysis saved.")
        print(f"📁 Output directory: {OUTPUT_DIR}")
        print("💡 Set GEMINI_API_KEY to enable AI report generation")
    
    print(f"\n📊 Analysis Summary:")
    if analysis_results and 'key_findings' in analysis_results:
        findings = analysis_results['key_findings']
        print(f"   • Change Magnitude: {findings.get('change_magnitude', 'N/A')}")
        print(f"   • Dominant Change: {findings.get('dominant_change_channel', 'N/A')} channel")
        print(f"   • Significance: {findings.get('change_significance', 'N/A')}")
    
    if analysis_results and 'threshold_40' in analysis_results:
        change_pct = analysis_results['threshold_40']['change_percentage']
        print(f"   • Moderate Changes: {change_pct:.2f}% of image area")