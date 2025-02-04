import time
import math
import base64
import streamlit as st
from PyPDF2 import PdfReader
from fpdf import FPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai

# --- Configuration ---
API_KEY = "AIzaSyBpeqtqemeM0YGxwxE3DTzc3Cny-w2Q_GA"  # Replace with your actual API key
genai.configure(api_key=API_KEY)

# --- Helper to Embed Local Image as Base64 ---
def get_base64_of_image(image_path):
    """
    Reads an image file and returns a base64 encoded string.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return ""

# --- Custom PDF Class without Header Text ---
class PDF(FPDF):
    def header(self):
        """
        Override header with no content.
        """
        pass

# --- Functions ---

def extract_text_from_pdf(uploaded_file):
    """Extract text from each page of the PDF file."""
    try:
        pdf_reader = PdfReader(uploaded_file)
        all_text = ""
        for i, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                all_text += f"\n\n--- Page {i} ---\n\n" + page_text
        return all_text.strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def divide_text_into_chunks(text, chunk_size=3000):
    """Divide the text into manageable chunks."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def call_gemini_api(prompt, max_retries=3):
    """Call the Gemini API with retries for content generation."""
    retries = 0
    model = genai.GenerativeModel("gemini-1.5-pro")
    while retries < max_retries:
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            retries += 1
            wait_time = 2 ** retries  # Exponential backoff
            st.warning(f"Error with Gemini API: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    st.error("Max retries exceeded. Please try again later.")
    return None

def review_ctd_document(document_text, template_content, workers=4, chunk_size=3000):
    """Generate AI review outputs concurrently for the CTD document."""
    chunks = divide_text_into_chunks(document_text, chunk_size=chunk_size)
    reviews = []
    prompts = []
    for chunk in chunks:
        prompt = (
            "You are an expert in reviewing Common Technical Dossiers (CTDs) for regulatory compliance and quality.\n"
            "Analyze the content below in detail and provide a structured review with comprehensive comments and suggestions for improvement.\n"
            "Follow the exact structure provided in the review format template. The output should include:\n"
            "Section:{Name}\nSubsection:{Name}\nReview comments:{Your detailed feedback}\n"
            "If you encounter tables or images that need special review, include a placeholder note such as:\n"
            "[Placeholder: Detailed table review required] or [Placeholder: Detailed image review required].\n"
            "Separate each section review with a line of dashes (e.g. '---------------------------').\n\n"
            "Review Template:\n"
            f"{template_content}\n\n"
            "Content:\n"
            f"{chunk}"
        )
        prompts.append(prompt)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_prompt = {executor.submit(call_gemini_api, prompt): prompt for prompt in prompts}
        for future in as_completed(future_to_prompt):
            review = future.result()
            if review:
                reviews.append(review)
    return "\n\n".join(reviews)

def parse_review_entries(review_output):
    """Parse review output blocks into structured entries."""
    blocks = [block.strip() for block in review_output.split('---------------------------') if block.strip()]
    review_entries = []
    for block in blocks:
        sec_name = "N/A"
        subsec_name = "N/A"
        review_comments = ""
        lines = block.splitlines()
        for line in lines:
            if line.startswith("Section:"):
                sec_name = line[len("Section:"):].strip()
            elif line.startswith("Subsection:"):
                subsec_name = line[len("Subsection:"):].strip()
            elif line.startswith("Review comments:"):
                review_comments = line[len("Review comments:"):].strip()
            else:
                if review_comments:
                    review_comments += " " + line.strip()
                else:
                    review_comments = line.strip()
        if "table" in review_comments.lower():
            review_comments += "\n[Placeholder: Detailed table review required]"
        if any(kw in review_comments.lower() for kw in ["image", "figure"]):
            review_comments += "\n[Placeholder: Detailed image review required]"
        review_entries.append({
            "section": sec_name,
            "subsection": subsec_name,
            "review_comments": review_comments
        })
    return review_entries

def parse_template_entries(template_content):
    """Parse the template file and return a list of (section, subsection) tuples."""
    template_entries = []
    lines = template_content.splitlines()
    current_section = None
    for line in lines:
        if line.startswith("Section:"):
            current_section = line[len("Section:"):].strip()
        elif line.startswith("Subsection:") and current_section:
            subsec = line[len("Subsection:"):].strip()
            template_entries.append((current_section, subsec))
    return template_entries

def generate_pdf_report(review_output, template_content):
    """Generate a PDF report from the AI review output with company header on the first page."""
    review_entries = parse_review_entries(review_output)
    template_entries = parse_template_entries(template_content)
    reviewed_sections = [(entry["section"], entry["subsection"]) for entry in review_entries]
    missing_entries = [t for t in template_entries if t not in reviewed_sections]
    
    # Create PDF using the custom PDF class.
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    # Register fonts (update paths as needed)
    pdf.add_font("ArialUnicode", "", "Arial.ttf", uni=True)
    pdf.add_font("ArialUnicode", "B", "ArialCEBoldItalic.ttf", uni=True)
    
    # First Page: Add company logo and title at center.
    pdf.add_page()
    # Add company logo at center.
    logo_path = "123.png"  # Update the image path if needed.
    logo_width = 50  # Adjust logo size as required.
    x_logo = (pdf.w - logo_width) / 2
    pdf.image(logo_path, x=x_logo, y=20, w=logo_width)
    
    # Add the review report title.
    pdf.set_font("ArialUnicode", "B", 14)
    pdf.cell(0, 10, "CTD Module 3 & Module 2 Review Report", ln=True, align='C')
    pdf.ln(5)
    
    # Add each review entry.
    for idx, entry in enumerate(review_entries, start=1):
        pdf.set_font("ArialUnicode", "B", 12)
        pdf.cell(0, 10, f"{idx}. Section: {entry['section']}", ln=True)
        pdf.set_font("ArialUnicode", "", 12)
        pdf.cell(0, 10, f"    Subsection: {entry['subsection']}", ln=True)
        pdf.multi_cell(0, 10, f"    Review Comments: {entry['review_comments']}")
        pdf.ln(3)
        pdf.cell(0, 0, "--------------------------------------------------", ln=True)
        pdf.ln(5)
    
    # Second Page: Checklist for Missing Entries.
    pdf.add_page()
    pdf.set_font("ArialUnicode", "B", 14)
    pdf.cell(0, 10, "Checklist: Missing Sections/Subsections", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("ArialUnicode", "", 12)
    if missing_entries:
        for idx, (sec, subsec) in enumerate(missing_entries, start=1):
            pdf.cell(0, 10, f"{idx}. Section: {sec} | Subsection: {subsec} [Missing]", ln=True)
    else:
        pdf.cell(0, 10, "All sections and subsections from the template are covered.", ln=True)
    
    pdf_path = "CTD_Module3_Module2_Review_Report.pdf"
    pdf.output(pdf_path)
    return pdf_path

# --- Streamlit App Layout and Branding ---

# Page configuration with custom page title and icon.
st.set_page_config(page_title="Pharma docket Document Review Generation Tool", page_icon="📄", layout="wide")

# Custom CSS for better styling.
st.markdown(
    """
    <style>
    .header {
        text-align: center;
        padding: 20px;
    }
    .logo {
        max-width: 50px;
    }
    .social-links {
        text-align: center;
        margin-top: 10px;
    }
    .social-links a {
        margin: 0 10px;
        font-size: 20px;
        text-decoration: none;
        color: #0a66c2;
    }
    </style>
    """, unsafe_allow_html=True
)

# Embed the logo image as base64 for the web interface.
logo_base64 = get_base64_of_image("123.png")
logo_html = f'<img class="logo" src="data:image/png;base64,{logo_base64}" alt="Pharma docket Logo">'

# Header with company logo and name.
st.markdown(
    f"""
    <div class="header">
        {logo_html}
        <h1>Pharma docket Document Review Generation Tool</h1>
        <p>Automation for modules of the Common Technical Document</p>
    </div>
    """, unsafe_allow_html=True
)

# Sidebar for social media links.
st.sidebar.markdown("## Connect with Pharma docket")
st.sidebar.markdown(
    """
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Pharma%20docket-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/company/pharmadocket/)
    
    [![Twitter](https://img.shields.io/badge/Twitter-Pharma%20docket-blue?style=for-the-badge&logo=twitter)](https://x.com/pharmadocket)
    """
)

st.markdown(
    """
    Upload the Module 3 and Module 2 CTD documents (PDF format) along with a text file containing your desired review format template.
    
    The template **must** include the following structure:
    
    ```
    Section:{Name}
    Subsection:{Name}
    Review comments:{Your detailed feedback}
    ---------------------------
    ```
    
    The AI agent will analyze the combined content of both documents and generate a detailed, structured review following your template.
    Additionally, the generated PDF will include:
      - Proper numbering for each review section.
      - Placeholders for tables or images when encountered.
      - A final checklist page highlighting any sections/subsections from the template that are missing in the review.
    """, unsafe_allow_html=True
)

# File uploader for Module 3 PDF.
uploaded_pdf_module3 = st.file_uploader("Upload Module 3 (PDF format only)", type=["pdf"], key="module3_pdf")

# File uploader for Module 2 PDF.
uploaded_pdf_module2 = st.file_uploader("Upload Module 2 (PDF format only)", type=["pdf"], key="module2_pdf")

# File uploader for the review format template (text file).
uploaded_template = st.file_uploader("Upload the Review Format Template (Text file)", type=["txt"], key="template_txt")

if uploaded_pdf_module3 and uploaded_pdf_module2 and uploaded_template:
    template_content = uploaded_template.read().decode("utf-8").strip()
    if not template_content:
        st.error("The review template file appears to be empty. Please provide a valid template.")
    else:
        # Extract text from both Module 3 and Module 2 PDFs.
        with st.spinner("Extracting text from the PDFs..."):
            document_text_module3 = extract_text_from_pdf(uploaded_pdf_module3)
            document_text_module2 = extract_text_from_pdf(uploaded_pdf_module2)
        
        if document_text_module3 and document_text_module2:
            # Combine texts with clear separation.
            combined_text = (
                "#### Module 3 Content ####\n" +
                document_text_module3 +
                "\n\n#### Module 2 Content ####\n" +
                document_text_module2
            )
            st.subheader("📑 Extracted Document Text Preview (First 1000 characters)")
            st.text_area("Extracted Text Preview", combined_text[:1000], height=200)

            if st.button("Analyze CTD Documents"):
                with st.spinner("Analyzing the documents using the AI agent..."):
                    review_output = review_ctd_document(combined_text, template_content)
                
                if review_output:
                    st.subheader("🔍 AI-Generated Review")
                    st.text_area("Review Output", review_output, height=500)
                    
                    pdf_path = generate_pdf_report(review_output, template_content)
                    with open(pdf_path, "rb") as f:
                        st.download_button("Download Review Report (PDF)", f, file_name="CTD_Module3_Module2_Review_Report.pdf", mime="application/pdf")
                else:
                    st.error("Failed to generate a review. Please try again.")
        else:
            st.error("Failed to extract text from one or both of the uploaded PDFs. Please check the files.")
else:
    st.info("Please upload the Module 3 PDF, Module 2 PDF, and the review format template to proceed.")
