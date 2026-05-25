"""NICE Clinical Knowledge Summaries - 50 common primary care topics.

Structured dictionary of NICE guideline requirements for verification.
Replaces LLM guessing with real, curated NICE guideline checks.
"""

from __future__ import annotations

from loguru import logger


NICE_CKS: list[dict] = [
    {
        "condition": "chest pain",
        "keywords": ["chest pain", "chest tightness", "angina", "cardiac"],
        "nice_ref": "NICE CG95 - Chest pain of recent onset",
        "key_checks": [
            {"requirement": "ECG within 12 hours", "search_terms": ["ecg", "electrocardiogram", "12-lead"]},
            {"requirement": "Troponin test", "search_terms": ["troponin"]},
            {"requirement": "Aspirin 300mg unless contraindicated", "search_terms": ["aspirin"]},
            {"requirement": "Risk assessment (e.g. HEART score)", "search_terms": ["heart score", "risk score", "risk assessment", "timi"]},
        ],
        "red_flags": ["Chest pain at rest", "Pain radiating to arm/jaw", "Sweating with chest pain", "New onset chest pain with exertion"],
    },
    {
        "condition": "hypertension",
        "keywords": ["hypertension", "high blood pressure", "elevated bp", "raised blood pressure"],
        "nice_ref": "NICE NG136 - Hypertension in adults",
        "key_checks": [
            {"requirement": "ABPM or HBPM to confirm diagnosis", "search_terms": ["abpm", "hbpm", "ambulatory", "home blood pressure", "home bp", "self-monitoring"]},
            {"requirement": "Target <140/90 mmHg (or <150/90 if >80 years)", "search_terms": ["target", "140/90", "blood pressure goal"]},
            {"requirement": "Check for target organ damage", "search_terms": ["target organ", "fundoscopy", "urine albumin", "ecg", "renal function", "end-organ"]},
            {"requirement": "Lifestyle advice (diet, exercise, alcohol, salt)", "search_terms": ["lifestyle", "diet", "exercise", "salt", "alcohol", "weight"]},
        ],
        "red_flags": ["BP >180/120 - accelerated hypertension", "Signs of target organ damage", "Phaeochromocytoma symptoms"],
    },
    {
        "condition": "type 2 diabetes",
        "keywords": ["type 2 diabetes", "t2dm", "diabetes mellitus type 2", "diabetic"],
        "nice_ref": "NICE NG28 - Type 2 diabetes in adults",
        "key_checks": [
            {"requirement": "HbA1c target discussion (usually 48 mmol/mol)", "search_terms": ["hba1c", "glycated haemoglobin", "target", "48"]},
            {"requirement": "Metformin as first-line", "search_terms": ["metformin"]},
            {"requirement": "Annual foot examination", "search_terms": ["foot", "feet", "neuropathy", "peripheral"]},
            {"requirement": "Annual eye screening referral", "search_terms": ["eye", "retinal", "retinopathy", "screening"]},
            {"requirement": "Cardiovascular risk assessment", "search_terms": ["cardiovascular", "qrisk", "statin", "lipid"]},
        ],
        "red_flags": ["DKA symptoms", "HbA1c >86 mmol/mol", "Foot ulcer or Charcot joint"],
    },
    {
        "condition": "depression",
        "keywords": ["depression", "depressive", "low mood", "depressed"],
        "nice_ref": "NICE CG90/CG91 - Depression in adults",
        "key_checks": [
            {"requirement": "PHQ-9 or equivalent severity scoring", "search_terms": ["phq-9", "phq9", "severity", "screening tool", "questionnaire"]},
            {"requirement": "Step-care model discussion", "search_terms": ["step", "guided self-help", "cbt", "counselling", "therapy"]},
            {"requirement": "SSRI if moderate-severe (sertraline first-line)", "search_terms": ["ssri", "sertraline", "antidepressant"]},
            {"requirement": "Safety assessment (suicidal ideation)", "search_terms": ["suicid", "self-harm", "safety", "risk assessment"]},
            {"requirement": "Follow-up within 2 weeks if starting antidepressant", "search_terms": ["follow-up", "review", "2 week"]},
        ],
        "red_flags": ["Suicidal ideation", "Self-harm", "Psychotic features", "Severe functional impairment"],
    },
    {
        "condition": "headache",
        "keywords": ["headache", "cephalgia", "head pain", "migraine"],
        "nice_ref": "NICE CG150 - Headaches in over 12s",
        "key_checks": [
            {"requirement": "Red flag assessment", "search_terms": ["red flag", "thunderclap", "worst headache", "sudden onset"]},
            {"requirement": "Neuroimaging if red flags present", "search_terms": ["neuroimaging", "ct head", "mri", "ct scan", "brain scan"]},
            {"requirement": "Fundoscopy/papilloedema check", "search_terms": ["papilloedema", "fundoscopy", "optic disc", "papilledema"]},
            {"requirement": "Medication overuse headache assessment", "search_terms": ["overuse", "medication overuse", "analgesic overuse"]},
        ],
        "red_flags": ["Thunderclap headache", "New headache >50 years", "Papilloedema", "Progressive neurological deficit", "Headache worse on waking/coughing"],
    },
    {
        "condition": "asthma",
        "keywords": ["asthma", "wheeze", "bronchospasm"],
        "nice_ref": "NICE NG80 - Asthma: diagnosis, monitoring and management",
        "key_checks": [
            {"requirement": "Peak flow or spirometry", "search_terms": ["peak flow", "spirometry", "fev1", "pef"]},
            {"requirement": "Inhaler technique check", "search_terms": ["inhaler technique", "technique check", "spacer"]},
            {"requirement": "Step-up/step-down assessment", "search_terms": ["step-up", "step-down", "step up", "step down", "preventer"]},
            {"requirement": "Written asthma action plan", "search_terms": ["action plan", "asthma plan", "self-management"]},
            {"requirement": "Annual review", "search_terms": ["annual review", "yearly review"]},
        ],
        "red_flags": ["Peak flow <50% best", "Unable to complete sentences", "Silent chest", "Altered consciousness"],
    },
    {
        "condition": "copd",
        "keywords": ["copd", "chronic obstructive", "emphysema", "chronic bronchitis"],
        "nice_ref": "NICE NG115 - COPD in over 16s",
        "key_checks": [
            {"requirement": "Spirometry for diagnosis (FEV1/FVC <0.7)", "search_terms": ["spirometry", "fev1", "fvc", "lung function"]},
            {"requirement": "Smoking cessation advice", "search_terms": ["smoking", "cessation", "quit", "stop smoking"]},
            {"requirement": "Exacerbation management plan", "search_terms": ["exacerbation", "rescue", "action plan", "self-management"]},
            {"requirement": "Pulmonary rehabilitation referral if appropriate", "search_terms": ["pulmonary rehabilitation", "rehab"]},
            {"requirement": "Vaccination (flu + pneumococcal)", "search_terms": ["vaccination", "flu", "influenza", "pneumococcal"]},
        ],
        "red_flags": ["Acute exacerbation with severe breathlessness", "Cor pulmonale signs", "SpO2 <92%"],
    },
    {
        "condition": "back pain",
        "keywords": ["back pain", "lumbar pain", "lumbago", "sciatica"],
        "nice_ref": "NICE NG59 - Low back pain and sciatica",
        "key_checks": [
            {"requirement": "Red flag assessment (cauda equina, malignancy, fracture)", "search_terms": ["red flag", "cauda equina", "malignancy", "fracture", "saddle"]},
            {"requirement": "Avoid routine imaging unless red flags", "search_terms": ["imaging", "mri", "x-ray", "not indicated", "avoid"]},
            {"requirement": "Self-management advice (stay active)", "search_terms": ["stay active", "exercise", "self-management", "physiotherapy"]},
            {"requirement": "Consider NSAIDs short-term", "search_terms": ["nsaid", "ibuprofen", "naproxen", "paracetamol"]},
        ],
        "red_flags": ["Cauda equina syndrome (saddle anaesthesia, bladder/bowel dysfunction)", "Thoracic pain", "Progressive neurological deficit", "Age >50 with new onset"],
    },
    {
        "condition": "urinary tract infection",
        "keywords": ["uti", "urinary tract infection", "cystitis", "dysuria"],
        "nice_ref": "NICE NG109 - Urinary tract infection (lower)",
        "key_checks": [
            {"requirement": "Nitrofurantoin first-line (3 days women, 7 days men)", "search_terms": ["nitrofurantoin"]},
            {"requirement": "Urine dipstick or culture", "search_terms": ["dipstick", "urine culture", "msu", "urinalysis"]},
            {"requirement": "Trimethoprim as alternative if nitrofurantoin unsuitable", "search_terms": ["trimethoprim"]},
            {"requirement": "Safety-netting: return if no improvement in 48 hours", "search_terms": ["safety", "48 hours", "return", "worsen"]},
        ],
        "red_flags": ["Pyelonephritis (loin pain, fever)", "Sepsis signs", "Haematuria >60 years (2WW referral)"],
    },
    {
        "condition": "atrial fibrillation",
        "keywords": ["atrial fibrillation", "af", "irregular pulse", "arrhythmia"],
        "nice_ref": "NICE NG196 - Atrial fibrillation",
        "key_checks": [
            {"requirement": "CHA2DS2-VASc score assessment", "search_terms": ["cha2ds2", "chads", "stroke risk"]},
            {"requirement": "Anticoagulation if CHA2DS2-VASc >=2 (men) or >=3 (women)", "search_terms": ["anticoagulant", "anticoagulation", "doac", "warfarin", "apixaban"]},
            {"requirement": "Rate vs rhythm control decision", "search_terms": ["rate control", "rhythm control", "bisoprolol", "digoxin", "cardioversion"]},
            {"requirement": "ECG confirmation", "search_terms": ["ecg", "electrocardiogram"]},
        ],
        "red_flags": ["Haemodynamic instability", "Heart rate >150", "Chest pain with AF"],
    },
    {
        "condition": "heart failure",
        "keywords": ["heart failure", "cardiac failure", "ccf", "hf"],
        "nice_ref": "NICE NG106 - Chronic heart failure in adults",
        "key_checks": [
            {"requirement": "NT-proBNP blood test", "search_terms": ["nt-probnp", "bnp", "natriuretic peptide"]},
            {"requirement": "Echocardiogram", "search_terms": ["echocardiogram", "echo", "ejection fraction"]},
            {"requirement": "ACE inhibitor + beta-blocker first-line (HFrEF)", "search_terms": ["ace inhibitor", "ramipril", "beta-blocker", "bisoprolol"]},
            {"requirement": "Diuretic for fluid overload", "search_terms": ["diuretic", "furosemide", "fluid"]},
        ],
        "red_flags": ["Acute pulmonary oedema", "Cardiogenic shock", "New onset AF with heart failure"],
    },
    {
        "condition": "gout",
        "keywords": ["gout", "gouty", "urate", "crystal arthropathy"],
        "nice_ref": "NICE CKS - Gout",
        "key_checks": [
            {"requirement": "NSAID or colchicine for acute attack", "search_terms": ["nsaid", "naproxen", "colchicine", "indometacin"]},
            {"requirement": "Urate-lowering therapy discussion if recurrent", "search_terms": ["allopurinol", "febuxostat", "urate-lowering"]},
            {"requirement": "Serum urate level check (not during acute attack)", "search_terms": ["urate", "uric acid"]},
        ],
        "red_flags": ["Septic arthritis differential", "Polyarticular gout", "Tophaceous gout"],
    },
    {
        "condition": "eczema",
        "keywords": ["eczema", "atopic dermatitis", "dermatitis"],
        "nice_ref": "NICE CG57 - Atopic eczema in under 12s / NICE CKS",
        "key_checks": [
            {"requirement": "Emollient as first-line (liberal use)", "search_terms": ["emollient", "moisturis"]},
            {"requirement": "Topical corticosteroid for flares", "search_terms": ["topical steroid", "hydrocortisone", "betamethasone", "topical corticosteroid"]},
            {"requirement": "Step-up/step-down approach", "search_terms": ["step-up", "step-down", "potency"]},
        ],
        "red_flags": ["Eczema herpeticum", "Widespread secondary infection"],
    },
    {
        "condition": "anxiety",
        "keywords": ["anxiety", "anxious", "panic", "generalised anxiety", "gad"],
        "nice_ref": "NICE CG113 - Generalised anxiety disorder and panic disorder",
        "key_checks": [
            {"requirement": "GAD-7 or equivalent severity scoring", "search_terms": ["gad-7", "gad7", "severity", "scoring"]},
            {"requirement": "Step-care model (guided self-help first)", "search_terms": ["self-help", "cbt", "cognitive", "therapy", "step"]},
            {"requirement": "SSRI if drug treatment needed (sertraline first-line)", "search_terms": ["ssri", "sertraline", "antidepressant"]},
        ],
        "red_flags": ["Suicidal ideation", "Inability to function", "Severe panic attacks with chest pain (exclude cardiac)"],
    },
    {
        "condition": "type 1 diabetes",
        "keywords": ["type 1 diabetes", "t1dm", "insulin dependent"],
        "nice_ref": "NICE NG17 - Type 1 diabetes in adults",
        "key_checks": [
            {"requirement": "HbA1c target (usually 48 mmol/mol)", "search_terms": ["hba1c", "target", "48"]},
            {"requirement": "Basal-bolus insulin regimen", "search_terms": ["basal-bolus", "insulin", "basal", "bolus"]},
            {"requirement": "DAFNE or equivalent structured education", "search_terms": ["dafne", "structured education", "carb counting"]},
            {"requirement": "Hypoglycaemia awareness assessment", "search_terms": ["hypoglycaemia", "hypo", "awareness"]},
        ],
        "red_flags": ["DKA", "Severe hypoglycaemia", "Impaired hypoglycaemia awareness"],
    },
    {
        "condition": "chronic kidney disease",
        "keywords": ["ckd", "chronic kidney disease", "renal impairment", "egfr"],
        "nice_ref": "NICE NG203 - Chronic kidney disease",
        "key_checks": [
            {"requirement": "eGFR and urine ACR monitoring", "search_terms": ["egfr", "acr", "albumin", "creatinine ratio"]},
            {"requirement": "BP target <140/90 (or <130/80 if ACR >70)", "search_terms": ["blood pressure", "target", "140/90"]},
            {"requirement": "ACE inhibitor/ARB if proteinuria", "search_terms": ["ace inhibitor", "arb", "ramipril", "losartan"]},
            {"requirement": "Medication dose adjustment review", "search_terms": ["dose adjustment", "renal dose", "dose"]},
        ],
        "red_flags": ["eGFR <15 (stage 5)", "Rapid decline >25% in 12 months", "Hyperkalaemia"],
    },
    {
        "condition": "dvt",
        "keywords": ["dvt", "deep vein thrombosis", "leg swelling", "venous thromboembolism"],
        "nice_ref": "NICE NG158 - Venous thromboembolic diseases",
        "key_checks": [
            {"requirement": "Wells score assessment", "search_terms": ["wells", "probability", "score"]},
            {"requirement": "D-dimer if low probability", "search_terms": ["d-dimer"]},
            {"requirement": "Ultrasound if high probability or positive D-dimer", "search_terms": ["ultrasound", "doppler", "compression"]},
            {"requirement": "Anticoagulation (DOAC first-line)", "search_terms": ["anticoagulant", "doac", "apixaban", "rivaroxaban"]},
        ],
        "red_flags": ["Pulmonary embolism signs", "Phlegmasia cerulea dolens", "Massive iliofemoral DVT"],
    },
    {
        "condition": "hypothyroidism",
        "keywords": ["hypothyroid", "underactive thyroid", "low thyroid"],
        "nice_ref": "NICE CKS - Hypothyroidism",
        "key_checks": [
            {"requirement": "TSH and free T4 blood test", "search_terms": ["tsh", "t4", "thyroid function"]},
            {"requirement": "Levothyroxine initiation (start low, increase gradually)", "search_terms": ["levothyroxine"]},
            {"requirement": "Recheck TSH at 6-8 weeks after dose change", "search_terms": ["recheck", "6 week", "8 week", "repeat", "monitor"]},
        ],
        "red_flags": ["Myxoedema coma", "Severe hypothyroidism with bradycardia"],
    },
    {
        "condition": "iron deficiency anaemia",
        "keywords": ["iron deficiency", "anaemia", "low haemoglobin", "low ferritin"],
        "nice_ref": "NICE CKS - Anaemia - iron deficiency",
        "key_checks": [
            {"requirement": "Oral iron supplementation (ferrous fumarate/sulphate)", "search_terms": ["iron", "ferrous", "iron supplement"]},
            {"requirement": "Investigate underlying cause", "search_terms": ["cause", "investigate", "coeliac", "gi", "menorrhagia"]},
            {"requirement": "Recheck FBC at 2-4 weeks", "search_terms": ["recheck", "repeat", "fbc", "haemoglobin"]},
        ],
        "red_flags": ["GI malignancy (>60 with iron deficiency - 2WW)", "Severe anaemia Hb <70"],
    },
    {
        "condition": "osteoarthritis",
        "keywords": ["osteoarthritis", "oa", "degenerative joint", "wear and tear"],
        "nice_ref": "NICE CG177 - Osteoarthritis",
        "key_checks": [
            {"requirement": "Exercise and weight management advice", "search_terms": ["exercise", "weight", "physiotherapy", "activity"]},
            {"requirement": "Topical NSAIDs first-line for hand/knee", "search_terms": ["topical", "nsaid", "diclofenac gel"]},
            {"requirement": "Paracetamol or oral NSAIDs if needed", "search_terms": ["paracetamol", "nsaid", "ibuprofen", "naproxen"]},
        ],
        "red_flags": ["Red hot joint (exclude septic arthritis/gout)", "Locking/giving way (meniscal)"],
    },
    {
        "condition": "pneumonia",
        "keywords": ["pneumonia", "chest infection", "lrti"],
        "nice_ref": "NICE CG191 - Pneumonia in adults",
        "key_checks": [
            {"requirement": "CRB-65 severity score", "search_terms": ["crb-65", "crb65", "severity", "curb"]},
            {"requirement": "Chest X-ray if diagnostic uncertainty", "search_terms": ["chest x-ray", "cxr"]},
            {"requirement": "Antibiotic within 4 hours (amoxicillin first-line for community)", "search_terms": ["amoxicillin", "antibiotic"]},
        ],
        "red_flags": ["CRB-65 >=3 - urgent hospital admission", "SpO2 <94%", "Sepsis"],
    },
    {
        "condition": "epilepsy",
        "keywords": ["epilepsy", "seizure", "fit", "convulsion"],
        "nice_ref": "NICE CG137 - Epilepsies",
        "key_checks": [
            {"requirement": "EEG and MRI brain", "search_terms": ["eeg", "mri", "brain scan"]},
            {"requirement": "Antiepileptic drug choice based on seizure type", "search_terms": ["antiepileptic", "aed", "lamotrigine", "carbamazepine", "valproate", "levetiracetam"]},
            {"requirement": "Driving advice (DVLA notification)", "search_terms": ["driving", "dvla"]},
        ],
        "red_flags": ["Status epilepticus", "First seizure in adult", "New neurological deficit post-seizure"],
    },
    {
        "condition": "acute coronary syndrome",
        "keywords": ["acs", "acute coronary syndrome", "nstemi", "stemi", "unstable angina"],
        "nice_ref": "NICE NG185 - Acute coronary syndromes",
        "key_checks": [
            {"requirement": "Immediate 12-lead ECG", "search_terms": ["ecg", "12-lead", "electrocardiogram"]},
            {"requirement": "Aspirin 300mg immediately", "search_terms": ["aspirin", "300mg"]},
            {"requirement": "Serial troponin measurements", "search_terms": ["troponin", "serial"]},
            {"requirement": "Pain relief (GTN, morphine if needed)", "search_terms": ["gtn", "morphine", "pain relief", "analgesia"]},
            {"requirement": "Emergency referral/call 999", "search_terms": ["999", "emergency", "ambulance", "a&e"]},
        ],
        "red_flags": ["ST elevation on ECG - immediate PCI", "Cardiogenic shock", "Ventricular arrhythmia"],
    },
    {
        "condition": "stroke",
        "keywords": ["stroke", "tia", "cerebrovascular", "facial droop", "arm weakness"],
        "nice_ref": "NICE NG128 - Stroke and TIA",
        "key_checks": [
            {"requirement": "FAST assessment", "search_terms": ["fast", "face", "arm", "speech", "time"]},
            {"requirement": "Immediate CT head scan", "search_terms": ["ct head", "ct scan", "brain imaging"]},
            {"requirement": "Thrombolysis if within 4.5 hours (ischaemic)", "search_terms": ["thrombolysis", "alteplase"]},
            {"requirement": "Aspirin 300mg after haemorrhage excluded", "search_terms": ["aspirin"]},
        ],
        "red_flags": ["Sudden onset focal neurology", "GCS <8 - intubation", "Large vessel occlusion - thrombectomy"],
    },
    {
        "condition": "sepsis",
        "keywords": ["sepsis", "septic", "systemic infection", "sirs"],
        "nice_ref": "NICE NG51 - Sepsis: recognition, diagnosis and early management",
        "key_checks": [
            {"requirement": "NEWS2 score assessment", "search_terms": ["news", "news2", "early warning"]},
            {"requirement": "Blood cultures before antibiotics", "search_terms": ["blood culture", "cultures"]},
            {"requirement": "IV antibiotics within 1 hour (Sepsis Six)", "search_terms": ["iv antibiotic", "sepsis six", "broad spectrum"]},
            {"requirement": "Fluid resuscitation", "search_terms": ["fluid", "iv fluid", "resuscitation", "crystalloid"]},
            {"requirement": "Lactate measurement", "search_terms": ["lactate"]},
        ],
        "red_flags": ["Septic shock (MAP <65 despite fluids)", "Lactate >4", "Multi-organ dysfunction"],
    },
    {
        "condition": "anaphylaxis",
        "keywords": ["anaphylaxis", "anaphylactic", "severe allergic reaction"],
        "nice_ref": "NICE CG134 - Anaphylaxis",
        "key_checks": [
            {"requirement": "IM adrenaline (0.5mg 1:1000)", "search_terms": ["adrenaline", "epinephrine", "epipen"]},
            {"requirement": "Lie flat with legs raised (unless breathing difficulty)", "search_terms": ["lie flat", "position"]},
            {"requirement": "Call 999/resuscitation team", "search_terms": ["999", "emergency", "resuscitation"]},
            {"requirement": "Mast cell tryptase (at 1 hour and 24 hours)", "search_terms": ["tryptase"]},
        ],
        "red_flags": ["Airway compromise", "Cardiovascular collapse", "Previous fatal anaphylaxis history"],
    },
    {
        "condition": "cellulitis",
        "keywords": ["cellulitis", "skin infection", "erythema spreading"],
        "nice_ref": "NICE CKS - Cellulitis",
        "key_checks": [
            {"requirement": "Flucloxacillin first-line (or clarithromycin if penicillin allergy)", "search_terms": ["flucloxacillin", "clarithromycin", "antibiotic"]},
            {"requirement": "Mark affected area to monitor spread", "search_terms": ["mark", "outline", "monitor spread"]},
            {"requirement": "Eron classification for severity", "search_terms": ["eron", "severity", "class"]},
        ],
        "red_flags": ["Necrotising fasciitis", "Orbital cellulitis", "Sepsis with cellulitis"],
    },
    {
        "condition": "pregnancy",
        "keywords": ["pregnant", "pregnancy", "antenatal"],
        "nice_ref": "NICE CG62 - Antenatal care for uncomplicated pregnancies",
        "key_checks": [
            {"requirement": "Folic acid 400mcg daily (5mg if high risk)", "search_terms": ["folic acid", "folate"]},
            {"requirement": "Booking appointment and blood tests", "search_terms": ["booking", "blood tests", "screening"]},
            {"requirement": "Review medications for safety in pregnancy", "search_terms": ["medication", "safe", "teratogenic", "contraindicated"]},
        ],
        "red_flags": ["Pre-eclampsia symptoms", "Ectopic pregnancy (pain + PV bleeding)", "Antepartum haemorrhage"],
    },
    {
        "condition": "menopause",
        "keywords": ["menopause", "menopausal", "hot flush", "perimenopause"],
        "nice_ref": "NICE NG23 - Menopause: diagnosis and management",
        "key_checks": [
            {"requirement": "Discuss HRT risks and benefits", "search_terms": ["hrt", "hormone replacement", "oestrogen"]},
            {"requirement": "Vasomotor symptom management", "search_terms": ["vasomotor", "hot flush", "night sweat"]},
            {"requirement": "Bone health/osteoporosis risk assessment", "search_terms": ["bone", "osteoporosis", "dexa", "calcium"]},
        ],
        "red_flags": ["Post-menopausal bleeding (2WW referral)", "VTE risk with HRT"],
    },
]


def check_nice_guidelines(
    chief_complaint: str,
    hpi: str,
    assessment: str,
    plan: str,
) -> list[dict]:
    """Check the clinical note against NICE CKS guidelines.

    Returns list of:
        {"nice_ref": str, "condition": str, "met": list[str], "not_met": list[str], "red_flags": list[str]}
    """
    note_text = f"{chief_complaint} {hpi} {assessment} {plan}".lower()

    results: list[dict] = []

    for guideline in NICE_CKS:
        if not any(kw in note_text for kw in guideline["keywords"]):
            continue

        met: list[str] = []
        not_met: list[str] = []

        for check in guideline["key_checks"]:
            if any(term in note_text for term in check["search_terms"]):
                met.append(check["requirement"])
            else:
                not_met.append(check["requirement"])

        matched_red_flags = [rf for rf in guideline["red_flags"] if any(
            word.lower() in note_text for word in rf.split() if len(word) > 3
        )]

        results.append({
            "nice_ref": guideline["nice_ref"],
            "condition": guideline["condition"],
            "met": met,
            "not_met": not_met,
            "red_flags": matched_red_flags,
        })

    logger.info(
        f"NICE guidelines check: {len(results)} conditions matched "
        f"({[r['condition'] for r in results]})"
    )
    return results
