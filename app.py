import streamlit as st
import fitz  # PyMuPDF
from optik1 import BubbleSheetScanner
import cv2
import numpy as np
import base64
import io
from mistralai import Mistral
from PIL import Image
from langchain_groq import ChatGroq
# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Bubble Sheet Student Feedback", layout="wide")

# Groq LLM
from groq import Groq
from langchain_core.prompts import PromptTemplate
API_KEY = "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"
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant", 
                api_key=API_KEY,
                verbose=True,
                temperature=0.0,
                
                )
# Bubble Sheet Scanner (your class)
scanner = BubbleSheetScanner(bubble_count=5)

# ======================
# Bubble Sheet Logic
# ======================
def process_bubble_sheet(image):
    h = int(600 * image.shape[0] / image.shape[1])
    frame = cv2.resize(image, (600, h))

    canny = scanner.getCannyFrame(frame)
    warped = scanner.getWarpedFrame(canny, frame)

    if warped is None:
        raise ValueError("Bubble sheet not detected")

    adaptive = scanner.getAdaptiveThresh(frame)
    ovals = scanner.getOvalContours(adaptive)

    total_bubbles = len(ovals)
    question_count = total_bubbles // scanner.bubbleCount
    ovals = sorted(ovals, key=scanner.y_cord)

    answers = []
    fill_threshold = 1.0

    for q in range(question_count):
        start = q * scanner.bubbleCount
        bubbles = sorted(
            ovals[start:start + scanner.bubbleCount],
            key=scanner.x_cord
        )

        best_idx = None
        best_ratio = 0

        for j, c in enumerate(bubbles):
            area = cv2.contourArea(c)
            mask = np.zeros(adaptive.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            masked = cv2.bitwise_and(adaptive, adaptive, mask=mask)
            filled = cv2.countNonZero(masked)

            ratio = filled / area if area else 0
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = j

        answers.append(best_idx if best_ratio > fill_threshold else None)

    return answers, warped

# ======================
# OCR
# ======================
# def mistral_ocr(pil_image):
#     buf = io.BytesIO()
#     pil_image.save(buf, format="PNG")
#     img_bytes = buf.getvalue()

#     # Base64 encode (as string)
#     b64 = base64.b64encode(img_bytes).decode("utf-8")

#     client = Mistral(api_key="97ZQlsV45YrDusgZRwjArWGbh3nerFPb")
#     resp = client.ocr.process(
#         model="mistral-ocr-latest",
#         document={
#             "type": "document_bytes",
#             "document_bytes": b64
#         }
#     )

    # text = ""
    # for p in resp.pages:
    #     text += p.markdown + "\n\n"

    # return text


# ======================
# LLM Analysis
# ======================
def analyze_student_performance(questions_text, student_answers):
    """
    Send student answers and questions to Groq LLM to generate
    Arabic feedback about weak points and study suggestions.
    """
    Answer_text = ', '.join([chr(ord('A') + a) if a is not None else 'فارغ' for a in student_answers])
    prompt = f"""
أنت معلم خبير في تدريس الفيزياء والعلوم للمرحلة الثانوية.

لديك البيانات التالية:

الأسئلة:
{questions_text}

إجابات الطالب:
{Answer_text}

قبل التحليل، قم بتحديد نوع الامتحان وفق القواعد التالية:

1) إذا كانت إجابات الطالب فارغة أو عددها أقل من 3 إجابات،
    فاعتبر أن الامتحان "مقالي" ويجب تحليل مستوى الفهم من طبيعة الأسئلة نفسها
    ومن مدى محاولة الطالب الإجابة.

2) إذا كان عدد الأسئلة أكبر بكثير من عدد الإجابات،
    فاعتبر أن الامتحان "مختلط بين مقالي واختياري ".

3) إذا كان لكل سؤال إجابة قصيرة أو اختيار واحد واضح،
    فاعتبر الامتحان "اختياري (MCQ)".

بعد تحديد نوع الامتحان، قم بالمهمة التالية:

1. حلّل مستوى الطالب العلمي وحدد نقاط ضعفه المفاهيمية في الفيزياء أو العلوم.
2. لا تذكر إجابات الطالب إطلاقًا، بل اذكر تحليلك أنت فقط.
3. وضّح المفاهيم التي يظهر أنه لا يفهمها بعمق (مثل: القوانين الفيزيائية، التحليل الرياضي، فهم الظواهر العلمية، الربط بين المفاهيم) و وضح له تحديداً الاجزاء و القوانين التي يحتاج ان يعيد دراستها
4. حدد الجزء أو الوحدة التي يجب أن يركز عليها في المنهج الدراسي تحديدًا.
5. اجعل التحليل عميقًا وتربويًا ومفصلًا.
6. اكتب الإجابة بالكامل باللغة العربية الفصحى الواضحة.
"""

    response = llm.invoke([{"role": "user", "content": prompt}])

    return response.content

# ======================
# Streamlit UI
# ======================
st.title("📝 Bubble Sheet + Student Feedback (Arabic)")

uploaded_file = st.file_uploader("Upload Exam PDF", type=["pdf"])
if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Read the uploaded file bytes once
        uploaded_file_bytes = uploaded_file.read()
        doc = fitz.open(stream=uploaded_file_bytes, filetype="pdf")
        answers = []
        questions_text = ""

        for i in range(len(doc)):
            page = doc[i]
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)
            pil_img = Image.open(io.BytesIO(pix.tobytes("png")))

            if i == 0:
                # Bubble sheet
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                answers, warped = process_bubble_sheet(cv_img)
                # st.subheader("🟢 Detected Bubble Sheet")
                # st.image(warped, channels="BGR")

        # Use the stored bytes for OCR
        pdf_bytes = uploaded_file_bytes
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        clint = Mistral(api_key="97ZQlsV45YrDusgZRwjArWGbh3nerFPb")
        ocr_resp = clint.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{b64_pdf}"
            },
            include_image_base64=True,
        )
        if len(answers) == 0:
            for page in ocr_resp.pages[0:]:
                questions_text += page.markdown + "\n\n" 
        else :           
            for page in ocr_resp.pages[1:]:
                questions_text += page.markdown + "\n\n"
        print(questions_text)
        # ======================
        # Show Answers
        # ======================
        # st.subheader("✅ Detected Answers")
        # cols = st.columns(4)
        # for idx, ans in enumerate(answers):
        #     letter = chr(ord("A") + ans) if ans is not None else "فارغ"
        #     cols[idx % 4].metric(f"Q{idx+1}", letter)

        # ======================
        # LLM Feedback
        # ======================
        st.subheader("🧠 Student Performance Feedback (Arabic)")
        with st.spinner("Generating feedback..."):
            feedback = analyze_student_performance(questions_text, answers)
        st.text_area("📌 تقرير الطالب", feedback, height=400)
