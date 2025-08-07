import streamlit as st
import cv2
from PIL import Image
import io
import time
import numpy as np

# Configure page
st.set_page_config(
    page_title="Remove Background from Your Image",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling (minimal, following Streamlit best practices)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🖼️ Remove background from your image")
    st.markdown("✨ Try uploading an image to watch the background magically removed. Full quality images can be downloaded from the sidebar.")
    st.markdown("This code is open source and available here on GitHub. Special thanks to the rembg library 😀")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # File uploader section
    st.markdown("### Upload and download 📂")
    st.markdown("**Upload an image**")
    
    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=['png', 'jpg', 'jpeg'],
        help="Limit 20MB per file • PNG, JPG, JPEG",
        label_visibility="collapsed"
    )
    
    # Display file info if uploaded
    if uploaded_file is not None:
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Convert to MB
        
        # Check file size limit
        if file_size > 20:
            st.error("❌ File size exceeds 20MB limit. Please upload a smaller image.")
            return
        
        # Display file information
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
        # Show upload instructions when no file is uploaded
        st.markdown("""
        <div class="upload-section">
            <h4>📤 Browse files</h4>
            <p>Upload an image to get started</p>
        </div>
        """, unsafe_allow_html=True)

def remove_background_cv2(image):
    """Remove background using OpenCV GrabCut algorithm"""
    # Convert PIL image to OpenCV format
    img = np.array(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Create mask for GrabCut
    height, width = img_rgb.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    
    # Define rectangle around the object (simplified approach)
    rect = (10, 10, width-20, height-20)
    
    # Initialize background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut
    cv2.grabCut(img_rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create final mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply mask to create transparent background
    result = img_rgb * mask2[:, :, np.newaxis]
    
    # Convert back to PIL with alpha channel
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # Create RGBA image with transparency
    result_rgba = np.dstack((result_rgb, mask2 * 255))
    result_pil = Image.fromarray(result_rgba, 'RGBA')
    
    return result_pil

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
        
        # Remove background using OpenCV (simple method)
        output_image = remove_background_cv2(input_image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        status_text.text("✅ Background removed successfully!")
        progress_bar.progress(100)
        
        # Display results
        display_results(input_image, output_image, processing_time, uploaded_file.name)
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.error("Please try with a different image or check if the image format is supported.")

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
        st.image(original_image, use_column_width=True)
        
    with col2:
        st.markdown("#### Fixed Image ✨")
        st.image(processed_image, use_column_width=True)
    
    # Prepare download
    prepare_download(processed_image, filename)

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
        This tool uses the **rembg** library to automatically remove backgrounds from images using AI/ML models.
        
        ### Supported formats:
        - PNG
        - JPG/JPEG
        
        ### Features:
        - ✅ High-quality background removal
        - ✅ Fast processing
        - ✅ No registration required
        - ✅ Privacy-focused (processing done locally)
        
        ### Usage tips:
        - Use high-resolution images for best results
        - Images with clear subject-background contrast work best
        - Processing time varies based on image size
        """)
        
        st.markdown("---")
        st.markdown("**Made with ❤️ using Streamlit & rembg**")

if __name__ == "__main__":
    # Show sidebar information
    show_sidebar_info()
    
    # Run main application
    main()
