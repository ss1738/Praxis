"""Clinical documentation templates used in UK General Practice.

Each template defines a system prompt, instruction, and output format
that the LLM must follow. These mirror real NHS/GP documentation standards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NoteTemplate(str, Enum):
    """Available clinical note templates."""
    SOAP = "soap"
    SBAR = "sbar"
    CLINICAL_SUMMARY = "clinical_summary"
    REFERRAL_LETTER = "referral_letter"
    DISCHARGE_SUMMARY = "discharge_summary"
    EMIS_CONSULTATION = "emis_consultation"


TEMPLATE_DISPLAY_NAMES = {
    "soap": "SOAP Note (Standard)",
    "sbar": "SBAR (Handover / Urgent)",
    "clinical_summary": "GP Consultation Summary",
    "referral_letter": "Referral Letter (NHS)",
    "discharge_summary": "Discharge / Follow-up Summary",
    "emis_consultation": "EMIS-Style Consultation Record",
}

TEMPLATE_DESCRIPTIONS = {
    "soap": "Standard Subjective/Objective/Assessment/Plan format. The most common structured clinical note format used in primary care worldwide.",
    "sbar": "Situation/Background/Assessment/Recommendation. Used for clinical handovers, urgent referrals, and inter-professional communication (NHS standard).",
    "clinical_summary": "Narrative GP consultation summary as written in EMIS Web or SystmOne. Includes presenting complaint, history, examination, diagnosis, and management plan in free-text format.",
    "referral_letter": "Formal NHS referral letter from GP to secondary care consultant. Follows NICE referral guidelines format.",
    "discharge_summary": "Post-consultation summary for patient records. Includes diagnosis, treatment given, follow-up plan, and safety-netting.",
    "emis_consultation": "Structured consultation record matching EMIS Web / SystmOne clinical system format with Read/SNOMED codes, problem list, and prescriptions.",
}

# Common SNOMED codes for UK GP consultations - used as few-shot examples in prompts
COMMON_SNOMED_CODES = """Common SNOMED CT codes for UK General Practice:
- 267036007: Dyspnoea (shortness of breath)
- 49727002: Cough
- 25064002: Headache
- 161891005: Backache (low back pain)
- 29857009: Chest pain
- 386661006: Fever
- 21522001: Abdominal pain
- 422587007: Nausea
- 44054006: Type 2 diabetes mellitus
- 38341003: Hypertensive disorder
- 195967001: Asthma
- 13645005: Chronic obstructive pulmonary disease
- 35489007: Depressive disorder
- 197480006: Anxiety disorder
- 65363002: Otitis media
- 36971009: Sinusitis
- 10509002: Acute bronchitis
- 233604007: Pneumonia
- 126851005: Urinary tract infection
- 449868002: Acute tonsillitis
- 387517004: Paracetamol (product)
- 372756006: Amoxicillin (product)
- 387207008: Ibuprofen (product)
- 373254001: Naproxen (product)
- 109081006: Metformin (product)
- 108600003: Atorvastatin (product)
- 372913009: Ramipril (product)
- 372897005: Salbutamol (product)
- 372767007: Omeprazole (product)
- 373225007: Prednisolone (product)
- 1003403007: Sertraline (product)"""


@dataclass
class TemplateConfig:
    """Complete configuration for a clinical note template."""
    template: NoteTemplate
    system_prompt: str
    instruction: str
    example_output: str


def get_template_config(template: NoteTemplate) -> TemplateConfig:
    """Get the full prompt configuration for a given template."""
    return TEMPLATE_CONFIGS[template]


TEMPLATE_CONFIGS = {
    # ─── SOAP NOTE ───────────────────────────────────────────────────
    NoteTemplate.SOAP: TemplateConfig(
        template=NoteTemplate.SOAP,
        system_prompt=(
            "You are a UK General Practice clinical documentation specialist. "
            "You generate structured SOAP notes from GP consultation transcripts. "
            "Rules you MUST follow:\n"
            "1. Use UK English spelling and terminology (paracetamol NOT acetaminophen, "
            "A&E NOT ER, physiotherapy NOT physical therapy, GP NOT PCP).\n"
            "2. ONLY document what was explicitly discussed in the transcript. "
            "Do NOT invent symptoms, findings, or diagnoses.\n"
            "3. Include SNOMED CT codes for all diagnoses, symptoms, and medications. "
            "Format as: [SNOMED: code] after each clinical term.\n"
            "4. Include safety-netting advice in the Plan section.\n"
            "5. Structure your output EXACTLY as shown, with these section headers on their own line:\n"
            "   SUBJECTIVE:\n   OBJECTIVE:\n   ASSESSMENT:\n   PLAN:\n\n"
            f"{COMMON_SNOMED_CODES}"
        ),
        instruction=(
            "Generate a structured SOAP note from this GP consultation transcript. "
            "You MUST use exactly these four section headers: SUBJECTIVE:, OBJECTIVE:, ASSESSMENT:, PLAN: "
            "(each on its own line). Include SNOMED CT codes in square brackets after clinical terms. "
            "Include safety-netting advice in the Plan."
        ),
        example_output=(
            "SUBJECTIVE:\n"
            "Presenting complaint: Patient reports persistent cough for 3 weeks [SNOMED: 49727002]. "
            "History of presenting illness: Initially dry, now productive with yellowish phlegm. "
            "Mild temperature last week. Currently taking ramipril 5mg [SNOMED: 372913009] and "
            "atorvastatin 20mg [SNOMED: 108600003]. No known allergies.\n\n"
            "OBJECTIVE:\n"
            "Examination findings: Crackles at right base on auscultation. "
            "Temperature: 37.8C [SNOMED: 386661006]. BP: 128/82.\n\n"
            "ASSESSMENT:\n"
            "Lower respiratory tract infection [SNOMED: 10509002]. "
            "Likely bacterial given productive cough and focal chest signs.\n\n"
            "PLAN:\n"
            "1. Prescribe amoxicillin 500mg TDS for 5 days [SNOMED: 372756006]\n"
            "2. Continue ramipril and atorvastatin unchanged\n"
            "3. Safety-netting: Return if symptoms worsen, develop chest pain, "
            "haemoptysis, or breathlessness. Call 111 or attend A&E if acutely unwell.\n"
            "4. Follow-up: Review in 1 week if not improving"
        ),
    ),

    # ─── SBAR (Handover / Urgent Communication) ─────────────────────
    NoteTemplate.SBAR: TemplateConfig(
        template=NoteTemplate.SBAR,
        system_prompt=(
            "You are a UK GP writing an SBAR clinical handover communication. "
            "SBAR is the NHS-standard format for urgent clinical communication between "
            "healthcare professionals. It must be concise, factual, and actionable.\n"
            "Rules:\n"
            "1. Use UK English.\n"
            "2. Be concise - SBAR is designed for rapid communication.\n"
            "3. Include SNOMED CT codes for key diagnoses.\n"
            "4. ONLY include information from the transcript.\n"
            "5. Structure EXACTLY as: SITUATION:, BACKGROUND:, ASSESSMENT:, RECOMMENDATION:\n\n"
            f"{COMMON_SNOMED_CODES}"
        ),
        instruction=(
            "Generate an SBAR handover note from this GP consultation transcript. "
            "Use exactly these headers: SITUATION:, BACKGROUND:, ASSESSMENT:, RECOMMENDATION: "
            "Include SNOMED CT codes. Be concise and clinical."
        ),
        example_output=(
            "SITUATION:\n"
            "I am calling about Mr Smith, 58-year-old male, seen today with "
            "persistent cough [SNOMED: 49727002] for 6 weeks, unintentional weight loss, "
            "and 25 pack-year smoking history.\n\n"
            "BACKGROUND:\n"
            "No significant past medical history. No current medications. "
            "Examination: crackles right base, SpO2 96%, temperature 37.1C.\n\n"
            "ASSESSMENT:\n"
            "Concerning for possible lung malignancy given persistent cough, weight loss, "
            "and smoking history. Requires urgent investigation.\n\n"
            "RECOMMENDATION:\n"
            "Requesting urgent chest X-ray and two-week wait referral to respiratory medicine. "
            "Bloods taken: FBC, CRP, LFTs. Patient advised to attend A&E if develops "
            "haemoptysis or acute breathlessness."
        ),
    ),

    # ─── GP CONSULTATION SUMMARY ─────────────────────────────────────
    NoteTemplate.CLINICAL_SUMMARY: TemplateConfig(
        template=NoteTemplate.CLINICAL_SUMMARY,
        system_prompt=(
            "You are a UK GP writing a consultation summary for the patient's clinical record. "
            "This is a narrative-style note as would be entered into EMIS Web or SystmOne. "
            "It should be professional, concise, and clinically complete.\n"
            "Rules:\n"
            "1. Use UK English.\n"
            "2. Write in third person ('Patient reports...' not 'I have...').\n"
            "3. Include SNOMED CT codes for diagnoses and key findings.\n"
            "4. Structure as: Presenting Complaint, History, Examination, Diagnosis, Management, Safety-netting.\n"
            "5. ONLY document what was discussed.\n\n"
            f"{COMMON_SNOMED_CODES}"
        ),
        instruction=(
            "Write a GP consultation summary from this transcript. "
            "Use these sections: Presenting Complaint, History of Presenting Illness, "
            "Examination, Diagnosis, Management Plan, Safety-netting & Follow-up. "
            "Include SNOMED codes. Write in third person."
        ),
        example_output=(
            "Presenting Complaint:\n"
            "Persistent cough for 3 weeks [SNOMED: 49727002]\n\n"
            "History of Presenting Illness:\n"
            "Patient reports initially dry cough, now productive with yellowish phlegm. "
            "Mild pyrexia last week. Some breathlessness on exertion. No haemoptysis, "
            "no chest pain. PMH: hypertension (ramipril 5mg OD), hyperlipidaemia "
            "(atorvastatin 20mg ON). NKDA.\n\n"
            "Examination:\n"
            "T 37.8C. BP 128/82. SpO2 97% RA. Chest: crackles right base. "
            "No wheeze. Heart sounds normal.\n\n"
            "Diagnosis:\n"
            "Lower respiratory tract infection [SNOMED: 10509002]\n\n"
            "Management Plan:\n"
            "- Amoxicillin 500mg TDS x 5 days [SNOMED: 372756006]\n"
            "- Continue existing medications\n\n"
            "Safety-netting & Follow-up:\n"
            "Advised to return if symptoms worsen, haemoptysis, chest pain, "
            "or increasing breathlessness. Review 1 week if not improving."
        ),
    ),

    # ─── NHS REFERRAL LETTER ─────────────────────────────────────────
    NoteTemplate.REFERRAL_LETTER: TemplateConfig(
        template=NoteTemplate.REFERRAL_LETTER,
        system_prompt=(
            "You are a UK GP writing a referral letter to a secondary care consultant. "
            "This follows the standard NHS referral format.\n"
            "Rules:\n"
            "1. Use UK English and formal letter conventions.\n"
            "2. Address as 'Dear Colleague' if specialty is unknown.\n"
            "3. Include reason for referral, relevant history, examination findings, "
            "   current medications, and specific questions for the specialist.\n"
            "4. Include SNOMED CT codes for key diagnoses.\n"
            "5. ONLY include information from the transcript.\n\n"
            f"{COMMON_SNOMED_CODES}"
        ),
        instruction=(
            "Write an NHS GP referral letter from this consultation transcript. "
            "Format as a formal letter with: Dear Colleague, reason for referral, "
            "clinical details, current medications, and request. "
            "Include SNOMED codes for diagnoses."
        ),
        example_output=(
            "Dear Colleague,\n\n"
            "Re: [Patient Name], DOB: [DOB], NHS No: [NHS Number]\n\n"
            "Thank you for seeing this patient who I am referring with a "
            "persistent cough [SNOMED: 49727002] of 6 weeks duration, "
            "unintentional weight loss of approximately 3kg, and a 25 pack-year "
            "smoking history.\n\n"
            "History:\n"
            "Patient presented with persistent dry cough progressing to productive cough. "
            "Associated weight loss noted over 2 months. No haemoptysis. "
            "Increasing exertional breathlessness.\n\n"
            "Examination:\n"
            "Crackles at right base. SpO2 96% on room air. Temperature 37.1C. "
            "BMI 23.4.\n\n"
            "Investigations:\n"
            "FBC, CRP, and LFTs requested. Urgent chest X-ray arranged.\n\n"
            "Current Medications:\n"
            "Nil regular\n\n"
            "I would be grateful for your assessment and further investigation. "
            "I have explained to the patient that this referral is precautionary "
            "and does not necessarily indicate a serious diagnosis.\n\n"
            "Yours faithfully,\n"
            "Dr [Name], [Practice Name]"
        ),
    ),

    # ─── DISCHARGE / FOLLOW-UP SUMMARY ───────────────────────────────
    NoteTemplate.DISCHARGE_SUMMARY: TemplateConfig(
        template=NoteTemplate.DISCHARGE_SUMMARY,
        system_prompt=(
            "You are a UK GP writing a post-consultation discharge or follow-up summary. "
            "This is given to the patient or filed in their record after a consultation.\n"
            "Rules:\n"
            "1. Use UK English.\n"
            "2. Be clear and patient-friendly where appropriate.\n"
            "3. Include diagnosis, treatment, follow-up plan, and red flags.\n"
            "4. Include SNOMED CT codes for clinical terms.\n"
            "5. ONLY include information from the transcript.\n\n"
            f"{COMMON_SNOMED_CODES}"
        ),
        instruction=(
            "Write a discharge/follow-up summary from this GP consultation. "
            "Include: Diagnosis, Treatment Given, Medications, Follow-up Plan, "
            "and Red Flags / When to Seek Help. Include SNOMED codes."
        ),
        example_output=(
            "Discharge Summary\n\n"
            "Date of Consultation: [Date]\n"
            "Clinician: Dr [Name]\n\n"
            "Diagnosis:\n"
            "Lower respiratory tract infection [SNOMED: 10509002]\n\n"
            "Treatment Given:\n"
            "- Prescribed amoxicillin 500mg three times daily for 5 days [SNOMED: 372756006]\n"
            "- Advised regular paracetamol [SNOMED: 387517004] for fever and aches\n"
            "- Advised increased fluid intake and rest\n\n"
            "Current Medications:\n"
            "- Ramipril 5mg OD (continue) [SNOMED: 372913009]\n"
            "- Atorvastatin 20mg ON (continue) [SNOMED: 108600003]\n"
            "- Amoxicillin 500mg TDS x 5 days (NEW)\n\n"
            "Follow-up Plan:\n"
            "Review in 1 week if symptoms not improving. "
            "Repeat chest examination at follow-up.\n\n"
            "Red Flags - Seek Urgent Help If:\n"
            "- Worsening breathlessness\n"
            "- Coughing up blood (haemoptysis)\n"
            "- Chest pain\n"
            "- Fever not responding to paracetamol\n"
            "- Feeling significantly worse\n"
            "Contact: Call 111 for advice, or attend A&E / call 999 if acutely unwell."
        ),
    ),

    # ─── EMIS-STYLE CONSULTATION RECORD ──────────────────────────────
    NoteTemplate.EMIS_CONSULTATION: TemplateConfig(
        template=NoteTemplate.EMIS_CONSULTATION,
        system_prompt=(
            "You are a UK GP entering a consultation record into EMIS Web or SystmOne. "
            "This is the structured electronic health record format used in UK primary care.\n"
            "Rules:\n"
            "1. Use UK English.\n"
            "2. Use the structured format with Problem, History, Examination, Comment, "
            "   Medication Issued, and Follow-up.\n"
            "3. Include SNOMED/Read codes for all coded entries.\n"
            "4. Be concise - this is a working clinical record, not a letter.\n"
            "5. ONLY document what was discussed.\n\n"
            f"{COMMON_SNOMED_CODES}"
        ),
        instruction=(
            "Write an EMIS Web / SystmOne consultation record from this transcript. "
            "Use: Problem (with SNOMED code), History, Examination, Comment, "
            "Medication Issued (with codes), and Follow-up. "
            "Be concise and use clinical abbreviations where appropriate."
        ),
        example_output=(
            "Consultation Type: Face-to-face\n"
            "Duration: 10 minutes\n\n"
            "Problem:\n"
            "- Lower respiratory tract infection [SNOMED: 10509002]\n\n"
            "History:\n"
            "3/52 Hx cough [SNOMED: 49727002], initially dry now productive "
            "yellow sputum. Mild pyrexia last week. SOB on exertion. "
            "No haemoptysis, no CP. PMH: HTN, hyperlipidaemia. NKDA.\n\n"
            "Examination:\n"
            "T 37.8 | BP 128/82 | SpO2 97% RA\n"
            "Chest: crackles R base. No wheeze. HS I+II+0.\n\n"
            "Comment:\n"
            "Clinical picture consistent with LRTI. No red flags for pneumonia "
            "(well-appearing, normal SpO2, afebrile at review).\n\n"
            "Medication Issued:\n"
            "- Amoxicillin 500mg capsules [SNOMED: 372756006] - TDS x 5/7 - Acute Rx\n\n"
            "Follow-up:\n"
            "Review 1/52 if not improving. Safety-netting: return if worsening SOB, "
            "haemoptysis, CP. 111/A&E if acutely unwell."
        ),
    ),
}
