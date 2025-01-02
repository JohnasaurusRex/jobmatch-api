from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import PyPDF2
import os
from google.api_core import retry
from functools import partial
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

api_key = os.getenv('GENAI_API_KEY')
if not api_key:
    raise ValueError("GENAI_API_KEY not found in environment variables")

genai.configure(
    api_key=api_key,
    transport='rest'
)

def generate_analysis(prompt):
    try:
        model = genai.GenerativeModel(
            model_name='gemini-pro',
            generation_config={
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
        )
        response = model.generate_content(prompt)
        
        # Add error handling for the response
        if not response.text:
            raise ValueError("Empty response received from Gemini AI")
            
        return response.text
    except Exception as e:
        print(f"Error in generate_analysis: {str(e)}")
        raise

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error in extract_text_from_pdf: {str(e)}")
        raise

def analyze_resume(resume_text, job_description):
    prompt = f"""
    As the Head of Talent Acquisition, **conduct a highly critical and objective analysis** of this resume against the job description using the following categories. 
    Provide a **concise and precise ATS analysis** with **strict adherence to the JSON format** below. 
    **No extra text or explanations are permitted** before, within, or after the JSON. 

    **Categories:**

    1. **Searchability:**
        - **Meticulously** evaluate resume formatting for ATS compatibility. 
        - **Thoroughly** assess presence of essential details: contact, job titles, relevant sections. 
        - **Precisely** compare resume and job description job titles for accuracy and relevance.

    2. **Hard Skills:**
        - **Carefully** analyze and **match** resume skills with job description requirements.
        - **Determine** the level of technical proficiency, identifying **specific strengths and weaknesses.**
        - **Provide actionable recommendations** for skill gaps and improvements.

    3. **Soft Skills:**
        - **Evaluate** how effectively the resume demonstrates soft skills **crucial** for the job.
        - **Assess** leadership, interpersonal, and communication skills with **rigor.**
        - **Offer concrete recommendations** for soft skill enhancement.

    4. **Recruiter Tips:**
        - **Accurately** assess resume suitability for the **appropriate job level** (entry, mid, senior).
        - **Scrutinize** the resume for **quantifiable achievements** (e.g., sales, leads, metrics).
        - **Evaluate** the resume's **tone and professionalism** in relation to the target role.
        - **Analyze** and **recommend** inclusion/exclusion of online presence or portfolio links.

    5. **Overall:**
        - **Assign a precise score (out of 100)** reflecting resume alignment with the job description.
        - **What is the candidate applying for?** Provide a clear explanation of the job title.
        - **Provide a clear and decisive shortlist recommendation** with **strong justification.**
        - **Identify critical improvements** essential for increasing selection chances.
        - **Highlight the candidate's most compelling strengths** for the role.

    **JSON Format (Strictly Enforced):**

    {{
        "searchability": {{
            "score": <number 0-100>,
            "contact_info": {{
                "present": boolean,
                "missing": ["<string>"]
            }},
            "sections": {{
                "has_summary": boolean,
                "has_proper_headings": boolean,
                "properly_formatted_dates": boolean
            }},
            "job_title_match": {{
                "score": <number 0-100>,
                "explanation": "<string>"
            }},
            "recommendations": ["<string>"]
        }},
        "hard_skills": {{
            "score": <number 0-100>,
            "matched_skills": ["<string>"],
            "missing_skills": ["<string>"],
            "technical_proficiency": {{
                "score": <number 0-100>,
                "strengths": ["<string>"],
                "gaps": ["<string>"]
            }},
            "recommendations": ["<string>"]
        }},
        "soft_skills": {{
            "score": <number 0-100>,
            "matched_skills": ["<string>"],
            "missing_skills": ["<string>"],
            "leadership_indicators": ["<string>"],
            "recommendations": ["<string>"]
        }},
        "recruiter_tips": {{
            "score": <number 0-100>,
            "job_level_match": {{
                "assessment": "<string>",
                "recommendation": "<string>"
            }},
            "measurable_results": {{
                "present": ["<string>"],
                "missing": ["<string>"]
            }},
            "resume_tone": {{
                "assessment": "<string>",
                "improvements": ["<string>"]
            }},
            "web_presence": {{
                "mentioned": ["<string>"],
                "recommended": ["<string>"]
            }}
        }},
        "overall": {{
            "total_score": <number 0-100>,
            "applying_for": {{
                "job_title": "<string>",
                "explanation": "<string>"
            }},
            "shortlist_recommendation": {{
                "decision": "<string>", 
                "explanation": "<string>"
            }},
            "critical_improvements": ["<string>"],
            "key_strengths": ["<string>"]
        }}
    }}

    Resume:
    {resume_text}

    Job Description:
    {job_description}
    """

    try:
        analysis_text = generate_analysis(prompt)

        # Debug logging
        print("Raw response from Gemini:", analysis_text)

        # Try to clean the response if it contains markdown code blocks
        if "```json" in analysis_text:
            analysis_text = analysis_text.split("```json")[1].split("```")[0].strip()
        elif "```" in analysis_text:
            analysis_text = analysis_text.split("```")[1].strip()

        # Parse the response as JSON and validate structure
        analysis_json = json.loads(analysis_text)

        # Validate essential structure
        required_keys = ['searchability', 'hard_skills', 'soft_skills', 'recruiter_tips', 'overall']
        if not all(key in analysis_json for key in required_keys):
            raise ValueError("Missing required keys in response structure")

        return analysis_json
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        print(f"Problematic text: {analysis_text}")
        raise ValueError(f"Failed to parse AI response as JSON: {str(e)}")
    except Exception as e:
        print(f"Error in analyze_resume: {str(e)}")
        raise


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        resume_file = request.files['resume']
        job_description = request.form.get('jobDescription', '')
        
        if not job_description:
            return jsonify({'error': 'No job description provided'}), 400
        
        resume_text = extract_text_from_pdf(resume_file)
        
        # Add input validation
        if not resume_text.strip():
            return jsonify({'error': 'Empty resume text extracted'}), 400
            
        analysis = analyze_resume(resume_text, job_description)
        
        return jsonify(analysis)
    
    except ValueError as e:
        print(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error in /api/analyze: {str(e)}")
        return jsonify({'error': 'Internal server error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)