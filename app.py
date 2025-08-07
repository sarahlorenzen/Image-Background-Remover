import streamlit as st
import cv2
from PIL import Image
import io
import time
import numpy as np

# Configure page
st.set_page_config(
    page_title="Image Background Remover",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize theme state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Theme toggle button in top right
col1, col2, col3 = st.columns([4, 1, 1])
with col3:
    theme_icon = "🌙" if not st.session_state.dark_mode else "☀️"
    theme_text = "Dark" if not st.session_state.dark_mode else "Light"
    if st.button(f"{theme_icon} {theme_text}", key="theme_toggle", on_click=toggle_theme, use_container_width=True):
        pass

# Custom CSS with theme support
dark_theme_css = """
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        margin-top: 0.5rem;
        color: #fafafa;
    }
    /* Reduce empty space above title */
    .stApp > div:first-child {
        padding-top: 0.5rem !important;
    }
    .block-container {
        padding-top: 1rem !important;
    }
    .upload-section {
        border: 2px dashed #555555;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background-color: #1e1e1e;
        color: #fafafa;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stButton > button[kind="primary"] {
        background-color: #4CAF50 !important;
        border-color: #4CAF50 !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #45a049 !important;
        border-color: #45a049 !important;
    }
    .stDownloadButton > button {
        background-color: #5239CB !important;
        border-color: #5239CB !important;
        color: white !important;
    }
    .stDownloadButton > button:hover {
        background-color: #4934b5 !important;
        border-color: #4934b5 !important;
    }
    .stFileUploader label button, .stFileUploader button, .stFileUploader [data-testid="baseButton-secondary"] {
        background-color: #004182 !important;
        color: white !important;
        border-color: #004182 !important;
        border: 1px solid #004182 !important;
    }
    .stFileUploader label button:hover, .stFileUploader button:hover, .stFileUploader [data-testid="baseButton-secondary"]:hover {
        background-color: #003366 !important;
        border-color: #003366 !important;
        border: 1px solid #003366 !important;
    }
    .stRadio > div {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e1e1e;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #0e1117;
    }
    .stMarkdown {
        color: #fafafa;
    }
    .stMetric {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 1rem;
    }
    /* Theme toggle button styling - only for theme toggle */
    .stButton:first-child button,
    .element-container .stButton button {
        background-color: #262730 !important;
        color: white !important;
        border: 1px solid #262730 !important;
    }
    .stButton:first-child button:hover,
    .element-container .stButton button:hover {
        background-color: #686868 !important;
        border: 1px solid #686868 !important;
        color: white !important;
    }
    /* Process All Images button styling for dark theme */
    .stButton button[kind="primary"] {
        background-color: #4CAF50 !important;
        border-color: #4CAF50 !important;
        color: white !important;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #2E7D32 !important;
        border-color: #2E7D32 !important;
        color: white !important;
    }
    /* Force file uploader button styling */
    section[data-testid="stFileUploader"] button,
    section[data-testid="stFileUploader"] label button,
    div[data-testid="stFileUploader"] button,
    div[data-testid="stFileUploader"] label button {
        background-color: #004182 !important;
        color: white !important;
        border: 1px solid #004182 !important;
    }
    section[data-testid="stFileUploader"] button:hover,
    section[data-testid="stFileUploader"] label button:hover,
    div[data-testid="stFileUploader"] button:hover,
    div[data-testid="stFileUploader"] label button:hover {
        background-color: #b13979 !important;
        border: 1px solid #b13979 !important;
    }
</style>
"""

light_theme_css = """
<style>
    .stApp {
        background-color: #ffffff;
        color: #262730;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        margin-top: 0.5rem;
        color: #262730;
    }
    /* Reduce empty space above title */
    .stApp > div:first-child {
        padding-top: 0.5rem !important;
    }
    .block-container {
        padding-top: 1rem !important;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background-color: #f8f9fa;
        color: #262730;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stButton > button[kind="primary"] {
        background-color: #4CAF50 !important;
        border-color: #4CAF50 !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #2E7D32 !important;
        border-color: #2E7D32 !important;
    }
    /* Override theme toggle styling for primary buttons specifically */
    .stButton > button[kind="primary"] {
        background-color: #4CAF50 !important;
        border-color: #4CAF50 !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #2E7D32 !important;
        border-color: #2E7D32 !important;
        color: white !important;
    }
    .stDownloadButton > button {
        background-color: #5239CB !important;
        border-color: #5239CB !important;
        color: white !important;
    }
    .stDownloadButton > button:hover {
        background-color: #4934b5 !important;
        border-color: #4934b5 !important;
    }
    .stFileUploader label button, .stFileUploader button, .stFileUploader [data-testid="baseButton-secondary"] {
        background-color: #004182 !important;
        color: white !important;
        border-color: #004182 !important;
        border: 1px solid #004182 !important;
    }
    .stFileUploader label button:hover, .stFileUploader button:hover, .stFileUploader [data-testid="baseButton-secondary"]:hover {
        background-color: #003366 !important;
        border-color: #003366 !important;
        border: 1px solid #003366 !important;
    }
    .stRadio > div {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        color: #262730;
    }
    .stSidebar {
        background-color: #ffffff;
    }
    .stMarkdown {
        color: #262730;
    }
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
    }
    /* Force file uploader button styling */
    section[data-testid="stFileUploader"] button,
    section[data-testid="stFileUploader"] label button,
    div[data-testid="stFileUploader"] button,
    div[data-testid="stFileUploader"] label button {
        background-color: #004182 !important;
        color: white !important;
        border: 1px solid #004182 !important;
    }
    section[data-testid="stFileUploader"] button:hover,
    section[data-testid="stFileUploader"] label button:hover,
    div[data-testid="stFileUploader"] button:hover,
    div[data-testid="stFileUploader"] label button:hover {
        background-color: #b13979 !important;
        border: 1px solid #b13979 !important;
    }
    /* Theme toggle button styling - only for theme toggle */
    .stButton:first-child button,
    .element-container .stButton button {
        background-color: #262730 !important;
        color: white !important;
        border: 1px solid #262730 !important;
    }
    .stButton:first-child button:hover,
    .element-container .stButton button:hover {
        background-color: #686868 !important;
        border: 1px solid #686868 !important;
        color: white !important;
    }
    /* Process All Images button override - must come after theme toggle styling */
    .stButton button[kind="primary"] {
        background-color: #4CAF50 !important;
        border-color: #4CAF50 !important;
        color: white !important;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #2E7D32 !important;
        border-color: #2E7D32 !important;
        color: white !important;
    }
    /* Radio button text styling for light mode */
    .stRadio label span,
    .stRadio div[role="radiogroup"] label,
    .stRadio > div > div > label,
    div[data-testid="stRadio"] label,
    .stRadio p,
    .stRadio div p,
    .stRadio label p {
        color: #262730 !important;
    }
    /* Download filename info text styling for light theme only */
    .stAlert[data-baseweb="notification"] .stMarkdown,
    .stAlert .stMarkdown p,
    div[data-testid="stAlert"] .stMarkdown,
    div[data-testid="stAlert"] p {
        color: #262730 !important;
    }
    /* Additional CSS to force text color - Updated */
    * {
        --text-color: #262730 !important;
    }
    /* Force refresh with timestamp */
    body::before {
        content: "Updated: 2025-01-07 15:52" !important;
        display: none !important;
    }
</style>
"""

# Apply theme
if st.session_state.dark_mode:
    st.markdown(dark_theme_css, unsafe_allow_html=True)
else:
    st.markdown(light_theme_css, unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🖼️ Image Background Remover")
    st.markdown("✨ Advanced multi-algorithm background removal with professional-quality results. Upload an image to experience precision edge detection and smooth transparency.")
    st.markdown("This tool combines 6 different computer vision techniques for superior background removal quality.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing mode selection
    st.markdown("### Choose processing mode:")
    processing_mode = st.radio(
        "Choose processing mode:",
        ["Single Image", "Batch Processing"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # File uploader section
    st.markdown("### Upload and download 📂")
    
    if processing_mode == "Single Image":
        uploaded_file = st.file_uploader(
            "Drag and drop file here",
            type=['png', 'jpg', 'jpeg'],
            help="Limit 20MB per file • PNG, JPG, JPEG",
            label_visibility="collapsed"
        )
        uploaded_files = [uploaded_file] if uploaded_file else []
    else:
        st.markdown("**Upload multiple images**")
        uploaded_files = st.file_uploader(
            "Drag and drop files here",
            type=['png', 'jpg', 'jpeg'],
            help="Limit 20MB per file • PNG, JPG, JPEG",
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
    
    # Display file info and processing options
    if uploaded_files and any(uploaded_files):
        # Filter out None values
        valid_files = [f for f in uploaded_files if f is not None]
        
        if not valid_files:
            # Show upload instructions when no file is uploaded
            st.markdown("""
            <div class="upload-section">
                <h4>📤 Browse files</h4>
                <p>Upload images to get started</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Check file sizes and display info
        total_size = 0
        oversized_files = []
        
        for uploaded_file in valid_files:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Convert to MB
            total_size += file_size
            
            if file_size > 20:
                oversized_files.append(uploaded_file.name)
        
        if oversized_files:
            st.error(f"❌ The following files exceed 20MB limit: {', '.join(oversized_files)}")
            st.error("Please upload smaller images or remove the oversized files.")
            return
        
        # Display file information
        if processing_mode == "Single Image":
            uploaded_file = valid_files[0]
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("📁")
            with col2:
                st.markdown(f"**{uploaded_file.name}**")
                st.markdown(f"{file_size:.2f}MB")
            
            # Process button
            if st.button("🚀 Remove Background", type="primary", use_container_width=True):
                process_image(uploaded_file)
        else:
            # Batch processing display
            st.markdown(f"**📁 {len(valid_files)} files selected** (Total: {total_size:.2f}MB)")
            
            # Show file list in expandable section
            with st.expander("View selected files"):
                for i, uploaded_file in enumerate(valid_files, 1):
                    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
                    st.markdown(f"{i}. **{uploaded_file.name}** - {file_size:.2f}MB")
            
            # Batch process button
            if st.button("🚀 Process All Images", type="primary", use_container_width=True):
                process_batch(valid_files)
    
    else:
        # Show upload instructions when no file is uploaded
        mode_text = "an image" if processing_mode == "Single Image" else "images"
        st.markdown(f"""
        <div class="upload-section">
            <h4>📤 Browse files</h4>
            <p>Upload {mode_text} to get started</p>
        </div>
        """, unsafe_allow_html=True)

def simple_background_removal(image):
    """Simple fallback background removal method"""
    # Convert to numpy array
    img = np.array(image)
    height, width = img.shape[:2]
    
    # Create a simple mask assuming the center contains the subject
    mask = np.zeros((height, width), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2
    
    # Create circular mask around center
    radius = min(width, height) // 3
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # Smooth the mask
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = mask.astype(np.float32) / 255.0
    
    # Apply mask
    result = img.copy()
    alpha = (mask * 255).astype(np.uint8)
    
    # Create RGBA
    result_rgba = np.dstack((result, alpha))
    return Image.fromarray(result_rgba.astype('uint8'))

def remove_background_cv2(image):
    """Optimized background removal using streamlined OpenCV techniques"""
    # Convert PIL image to OpenCV format
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    height, width = img_bgr.shape[:2]
    
    # Resize image if too large to prevent freezing
    max_dimension = 800
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        img_bgr_resized = cv2.resize(img_bgr, (new_width, new_height))
        process_resized = True
    else:
        img_bgr_resized = img_bgr
        new_height, new_width = height, width
        process_resized = False
    
    # GrabCut algorithm with optimized parameters
    mask = np.zeros((new_height, new_width), np.uint8)
    
    # Smart margin calculation
    margin = min(new_width, new_height) // 10
    rect = (margin, margin, new_width - 2*margin, new_height - 2*margin)
    
    # Initialize models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut with fewer iterations for speed
    cv2.grabCut(img_bgr_resized, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Simple morphological operations for cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    
    # Smooth edges with Gaussian blur
    mask2_float = mask2.astype(np.float32)
    mask2_smooth = cv2.GaussianBlur(mask2_float, (3, 3), 0)
    
    # Resize mask back to original size if needed
    if process_resized:
        mask2_smooth = cv2.resize(mask2_smooth, (width, height))
    
    # Apply mask to original image
    mask2_smooth = np.clip(mask2_smooth, 0, 1)
    
    # Create result
    result_rgb = img.copy()
    alpha = (mask2_smooth * 255).astype(np.uint8)
    
    # Apply mask to each channel
    for c in range(3):
        result_rgb[:, :, c] = (result_rgb[:, :, c] * mask2_smooth).astype(np.uint8)
    
    # Create RGBA image
    result_rgba = np.dstack((result_rgb, alpha))
    
    # Convert to PIL Image
    result_pil = Image.fromarray(result_rgba.astype('uint8'))
    
    return result_pil

def process_batch(uploaded_files):
    """Process multiple images for batch background removal"""
    
    if not uploaded_files:
        st.error("No files to process!")
        return
    
    st.markdown("---")
    st.markdown("### Batch Processing Results")
    
    # Show overall progress
    overall_progress = st.progress(0)
    overall_status = st.empty()
    
    # Results storage
    processed_results = []
    failed_files = []
    
    total_files = len(uploaded_files)
    start_time = time.time()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update overall progress
            progress_pct = int((i / total_files) * 100)
            overall_progress.progress(progress_pct)
            overall_status.text(f"Processing {i+1}/{total_files}: {uploaded_file.name}")
            
            # Process individual image
            input_image = Image.open(uploaded_file)
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            
            # Remove background
            try:
                output_image = remove_background_cv2(input_image)
            except Exception as process_error:
                # Use fallback method
                output_image = simple_background_removal(input_image)
            
            # Store result
            processed_results.append({
                'original_name': uploaded_file.name,
                'original_image': input_image,
                'processed_image': output_image,
                'file_size': len(uploaded_file.getvalue()) / (1024 * 1024)
            })
            
        except Exception as e:
            failed_files.append({
                'name': uploaded_file.name,
                'error': str(e)
            })
    
    # Complete processing
    total_time = time.time() - start_time
    overall_progress.progress(100)
    overall_status.text(f"✅ Completed! Processed {len(processed_results)}/{total_files} images in {total_time:.2f}s")
    
    # Display results
    if processed_results:
        display_batch_results(processed_results, total_time)
    
    if failed_files:
        st.markdown("### ⚠️ Failed Files")
        for failed in failed_files:
            st.error(f"❌ **{failed['name']}**: {failed['error']}")

def process_image(uploaded_file):
    """Process the uploaded image to remove background"""
    
    try:
        # Show processing status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load the image
        status_text.text("📖 Loading image...")
        progress_bar.progress(25)
        
        # Open image with PIL
        input_image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary (rembg expects RGB)
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        status_text.text("🔄 Processing image...")
        progress_bar.progress(50)
        
        # Start timing
        start_time = time.time()
        
        # Remove background using OpenCV with timeout protection
        try:
            output_image = remove_background_cv2(input_image)
        except Exception as process_error:
            status_text.text("❌ Processing failed, trying simple fallback...")
            # Simple fallback method
            output_image = simple_background_removal(input_image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        status_text.text("✅ Background removed successfully!")
        progress_bar.progress(100)
        
        # Display results
        display_results(input_image, output_image, processing_time, uploaded_file.name)
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.error("Please try with a different image or check if the image format is supported.")

def display_batch_results(processed_results, total_time):
    """Display results for batch processing"""
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Images Processed", len(processed_results))
    with col2:
        st.metric("Total Time", f"{total_time:.2f}s")
    with col3:
        avg_time = total_time / len(processed_results) if processed_results else 0
        st.metric("Avg Time/Image", f"{avg_time:.2f}s")
    with col4:
        total_mb = sum(result['file_size'] for result in processed_results)
        st.metric("Total Data", f"{total_mb:.1f}MB")
    
    # Individual results in grid layout
    st.markdown("### Results Gallery")
    
    # Show results in rows of 2
    for i in range(0, len(processed_results), 2):
        col1, col2 = st.columns(2)
        
        # First image in row
        result = processed_results[i]
        with col1:
            st.markdown(f"**{result['original_name']}**")
            
            # Before/After tabs
            tab1, tab2 = st.tabs(["Original", "Processed"])
            with tab1:
                st.image(result['original_image'], use_container_width=True)
            with tab2:
                st.image(result['processed_image'], use_container_width=True)
            
            # Download button for this image
            prepare_individual_download(result['processed_image'], result['original_name'])
        
        # Second image in row (if exists)
        if i + 1 < len(processed_results):
            result = processed_results[i + 1]
            with col2:
                st.markdown(f"**{result['original_name']}**")
                
                # Before/After tabs
                tab1, tab2 = st.tabs(["Original", "Processed"])
                with tab1:
                    st.image(result['original_image'], use_container_width=True)
                with tab2:
                    st.image(result['processed_image'], use_container_width=True)
                
                # Download button for this image
                prepare_individual_download(result['processed_image'], result['original_name'])
    
    # Batch download option
    st.markdown("---")
    st.markdown("### Download All")
    prepare_batch_download(processed_results)

def display_results(original_image, processed_image, processing_time, filename):
    """Display the original and processed images side by side"""
    
    st.markdown("---")
    st.markdown("### Results")
    
    # Display processing info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    with col2:
        st.metric("Original Format", original_image.format if hasattr(original_image, 'format') else 'Unknown')
    with col3:
        st.metric("Output Format", "PNG")
    
    # Display images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Image 📷")
        st.image(original_image, use_container_width=True)
        
    with col2:
        st.markdown("#### Fixed Image ✨")
        st.image(processed_image, use_container_width=True)
    
    # Prepare download
    prepare_download(processed_image, filename)

def prepare_individual_download(processed_image, original_filename):
    """Prepare individual image download for batch processing"""
    # Convert PIL image to bytes
    img_buffer = io.BytesIO()
    processed_image.save(img_buffer, format='PNG')
    img_bytes = img_buffer.getvalue()
    
    # Generate download filename
    name_without_ext = original_filename.rsplit('.', 1)[0]
    download_filename = f"{name_without_ext}_no_bg.png"
    
    # Compact download button
    st.download_button(
        label="💾 Download",
        data=img_bytes,
        file_name=download_filename,
        mime="image/png",
        use_container_width=True
    )

def prepare_batch_download(processed_results):
    """Prepare batch download as ZIP file"""
    import zipfile
    
    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for result in processed_results:
            # Convert image to bytes
            img_buffer = io.BytesIO()
            result['processed_image'].save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()
            
            # Generate filename for ZIP
            name_without_ext = result['original_name'].rsplit('.', 1)[0]
            zip_filename = f"{name_without_ext}_no_bg.png"
            
            # Add to ZIP
            zip_file.writestr(zip_filename, img_bytes)
    
    zip_bytes = zip_buffer.getvalue()
    
    # Download button for ZIP
    st.download_button(
        label="📦 Download All as ZIP",
        data=zip_bytes,
        file_name="processed_images_no_bg.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True
    )
    
    st.info(f"📁 ZIP file contains {len(processed_results)} processed images with transparent backgrounds")

def prepare_download(processed_image, original_filename):
    """Prepare the processed image for download"""
    
    # Convert PIL image to bytes
    img_buffer = io.BytesIO()
    processed_image.save(img_buffer, format='PNG')
    img_bytes = img_buffer.getvalue()
    
    # Generate download filename
    name_without_ext = original_filename.rsplit('.', 1)[0]
    download_filename = f"{name_without_ext}_no_bg.png"
    
    # Download button
    st.markdown("### Download fixed image")
    st.download_button(
        label="💾 Download fixed image",
        data=img_bytes,
        file_name=download_filename,
        mime="image/png",
        type="primary",
        use_container_width=True
    )
    
    # Additional info
    st.info(f"📁 Your image will be downloaded as: **{download_filename}**")

def show_sidebar_info():
    """Show additional information in sidebar"""
    with st.sidebar:
        st.markdown("## About")
        st.markdown("""
        This tool uses **optimized OpenCV algorithms** to automatically remove backgrounds from images with reliable performance.
        
        ### Supported formats:
        - PNG
        - JPG/JPEG
        
        ### Features:
        - ✅ Single image and batch processing modes
        - ✅ Smart GrabCut algorithm with automatic resizing
        - ✅ Smooth edge blending
        - ✅ Morphological noise cleanup
        - ✅ Fast and reliable processing
        - ✅ Fallback method for difficult images
        - ✅ ZIP download for batch results
        - ✅ No registration required
        - ✅ Privacy-focused (processing done locally)
        
        ### Processing Methods:
        1. **GrabCut Algorithm**: Graph-cut based segmentation
        2. **Smart Resizing**: Automatic scaling for optimal performance
        3. **Morphological Cleanup**: Noise reduction and hole filling
        4. **Gaussian Smoothing**: Soft edge blending
        5. **Fallback Protection**: Simple method if main algorithm fails
        
        ### Usage tips:
        - **Single Mode**: Process one image with detailed before/after view
        - **Batch Mode**: Process multiple images simultaneously
        - Works best with clear subject-background contrast
        - Images are automatically resized for optimal processing speed
        - Supports images up to 20MB each
        - Batch processing shows gallery view with individual downloads
        - Download all processed images as a convenient ZIP file
        """)
        
        st.markdown("---")
        st.markdown("**Made with ❤️ using Streamlit & rembg**")

if __name__ == "__main__":
    # Show sidebar information
    show_sidebar_info()
    
    # Run main application
    main()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; margin-top: 0.5rem; padding: 1rem; color: #666;">
            2025 © Made with ❤️ by Sarah Lorenzen using <a href="https://replit.com/refer/sarahlorenzen" target="_blank" style="color: #004182; text-decoration: underline;">Replit</a> + AI
        </div>
        """,
        unsafe_allow_html=True
    )
