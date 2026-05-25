"""Local drug interaction database - 102 common UK primary care interactions.

Curated from BNF and MHRA safety alerts. Pair-indexed for O(1) lookup.
Replaces LLM guesswork with clinically-grounded interaction data.
"""

from __future__ import annotations

from loguru import logger


INTERACTION_DB: list[dict] = [
    # --- Anticoagulants ---
    {"drug_a": "warfarin", "drug_b": "ibuprofen", "severity": "critical", "effect": "Major bleeding risk - NSAIDs inhibit platelet function and increase GI bleeding", "clinical_action": "Avoid combination. Use paracetamol for analgesia."},
    {"drug_a": "warfarin", "drug_b": "naproxen", "severity": "critical", "effect": "Major bleeding risk - NSAIDs inhibit platelet function and increase GI bleeding", "clinical_action": "Avoid combination. Use paracetamol for analgesia."},
    {"drug_a": "warfarin", "drug_b": "aspirin", "severity": "high", "effect": "Increased bleeding risk - dual anticoagulant/antiplatelet therapy", "clinical_action": "Only co-prescribe if clearly indicated (e.g. post-ACS). Add PPI gastroprotection."},
    {"drug_a": "warfarin", "drug_b": "fluconazole", "severity": "critical", "effect": "Fluconazole inhibits CYP2C9, markedly increasing warfarin levels and INR", "clinical_action": "Avoid or reduce warfarin dose by 25-50%. Monitor INR daily."},
    {"drug_a": "warfarin", "drug_b": "amoxicillin", "severity": "moderate", "effect": "Antibiotics may alter gut flora affecting vitamin K metabolism, increasing INR", "clinical_action": "Monitor INR within 3-5 days of starting antibiotic."},
    {"drug_a": "warfarin", "drug_b": "clarithromycin", "severity": "critical", "effect": "CYP3A4 inhibition - significantly increases warfarin levels", "clinical_action": "Avoid combination. Use alternative antibiotic. Monitor INR closely if unavoidable."},
    {"drug_a": "warfarin", "drug_b": "metronidazole", "severity": "high", "effect": "Metronidazole inhibits warfarin metabolism, increasing bleeding risk", "clinical_action": "Reduce warfarin dose. Monitor INR within 3 days."},
    {"drug_a": "apixaban", "drug_b": "clarithromycin", "severity": "high", "effect": "CYP3A4 and P-gp inhibition increases DOAC exposure", "clinical_action": "Avoid combination or reduce apixaban dose."},
    {"drug_a": "apixaban", "drug_b": "ibuprofen", "severity": "high", "effect": "Increased bleeding risk - NSAID + anticoagulant", "clinical_action": "Avoid NSAIDs. Use paracetamol."},
    # --- ACE inhibitors / ARBs ---
    {"drug_a": "ramipril", "drug_b": "spironolactone", "severity": "critical", "effect": "Hyperkalaemia risk - ACE inhibitor + potassium-sparing diuretic", "clinical_action": "Monitor potassium within 1 week. Avoid if K+ >5.0 mmol/L."},
    {"drug_a": "ramipril", "drug_b": "losartan", "severity": "critical", "effect": "Dual RAAS blockade - hyperkalaemia, renal impairment, hypotension", "clinical_action": "Do NOT combine ACE inhibitor + ARB. Choose one."},
    {"drug_a": "ramipril", "drug_b": "ibuprofen", "severity": "high", "effect": "NSAIDs reduce ACE inhibitor efficacy and increase renal impairment risk", "clinical_action": "Avoid long-term NSAID use. Monitor renal function if short course needed."},
    {"drug_a": "ramipril", "drug_b": "naproxen", "severity": "high", "effect": "NSAIDs reduce ACE inhibitor efficacy and increase renal impairment risk", "clinical_action": "Avoid long-term NSAID use. Monitor renal function."},
    {"drug_a": "ramipril", "drug_b": "trimethoprim", "severity": "high", "effect": "Trimethoprim inhibits potassium excretion - hyperkalaemia risk with ACE inhibitors", "clinical_action": "Monitor potassium within 3 days. Use alternative antibiotic if possible."},
    {"drug_a": "losartan", "drug_b": "spironolactone", "severity": "critical", "effect": "Hyperkalaemia risk - ARB + potassium-sparing diuretic", "clinical_action": "Monitor potassium within 1 week. Avoid if K+ >5.0 mmol/L."},
    {"drug_a": "losartan", "drug_b": "ibuprofen", "severity": "high", "effect": "NSAIDs reduce ARB efficacy and increase renal impairment risk", "clinical_action": "Avoid long-term NSAID use. Monitor renal function."},
    # --- Metformin ---
    {"drug_a": "metformin", "drug_b": "contrast dye", "severity": "critical", "effect": "Risk of lactic acidosis - contrast media can impair renal function", "clinical_action": "Withhold metformin 48 hours before and after IV contrast. Check eGFR before restarting."},
    {"drug_a": "metformin", "drug_b": "alcohol", "severity": "high", "effect": "Increased risk of lactic acidosis, especially with binge drinking", "clinical_action": "Advise moderate alcohol intake. Avoid binge drinking."},
    # --- SSRIs ---
    {"drug_a": "sertraline", "drug_b": "tramadol", "severity": "critical", "effect": "Serotonin syndrome risk - SSRI + serotonergic opioid", "clinical_action": "Avoid combination. Use alternative analgesic (paracetamol, codeine)."},
    {"drug_a": "sertraline", "drug_b": "ibuprofen", "severity": "high", "effect": "Increased GI bleeding risk - SSRI + NSAID", "clinical_action": "Co-prescribe PPI if NSAID essential. Prefer paracetamol."},
    {"drug_a": "sertraline", "drug_b": "warfarin", "severity": "high", "effect": "SSRIs increase bleeding risk via platelet inhibition", "clinical_action": "Monitor INR. Consider alternative antidepressant."},
    {"drug_a": "fluoxetine", "drug_b": "tramadol", "severity": "critical", "effect": "Serotonin syndrome risk - SSRI + serotonergic opioid", "clinical_action": "Avoid combination."},
    {"drug_a": "fluoxetine", "drug_b": "codeine", "severity": "high", "effect": "Fluoxetine inhibits CYP2D6, reducing codeine conversion to morphine - reduced analgesia", "clinical_action": "Codeine ineffective. Use alternative analgesic."},
    {"drug_a": "fluoxetine", "drug_b": "tamoxifen", "severity": "critical", "effect": "Fluoxetine inhibits CYP2D6 - blocks tamoxifen activation, reducing efficacy", "clinical_action": "Avoid combination. Use citalopram or venlafaxine instead."},
    {"drug_a": "citalopram", "drug_b": "tramadol", "severity": "critical", "effect": "Serotonin syndrome risk - SSRI + serotonergic opioid", "clinical_action": "Avoid combination."},
    {"drug_a": "citalopram", "drug_b": "amiodarone", "severity": "critical", "effect": "QT prolongation risk - both drugs prolong QT interval", "clinical_action": "Avoid combination. ECG monitoring if unavoidable."},
    {"drug_a": "mirtazapine", "drug_b": "tramadol", "severity": "high", "effect": "Serotonin syndrome risk - mirtazapine has serotonergic activity", "clinical_action": "Avoid combination if possible."},
    # --- Methotrexate ---
    {"drug_a": "methotrexate", "drug_b": "trimethoprim", "severity": "critical", "effect": "Both are folate antagonists - risk of fatal pancytopenia", "clinical_action": "NEVER co-prescribe. Use alternative antibiotic (nitrofurantoin for UTI)."},
    {"drug_a": "methotrexate", "drug_b": "ibuprofen", "severity": "critical", "effect": "NSAIDs reduce methotrexate clearance - risk of severe toxicity", "clinical_action": "Avoid NSAIDs. Use paracetamol for pain."},
    {"drug_a": "methotrexate", "drug_b": "naproxen", "severity": "critical", "effect": "NSAIDs reduce methotrexate clearance - risk of severe toxicity", "clinical_action": "Avoid NSAIDs."},
    {"drug_a": "methotrexate", "drug_b": "omeprazole", "severity": "moderate", "effect": "PPIs may reduce methotrexate clearance at high doses", "clinical_action": "Monitor for methotrexate toxicity. Usually safe at low rheumatology doses."},
    # --- Lithium ---
    {"drug_a": "lithium", "drug_b": "ibuprofen", "severity": "critical", "effect": "NSAIDs increase lithium levels by reducing renal clearance - toxicity risk", "clinical_action": "Avoid NSAIDs. If essential, monitor lithium levels within 5 days."},
    {"drug_a": "lithium", "drug_b": "naproxen", "severity": "critical", "effect": "NSAIDs increase lithium levels - toxicity risk", "clinical_action": "Avoid NSAIDs."},
    {"drug_a": "lithium", "drug_b": "ramipril", "severity": "high", "effect": "ACE inhibitors increase lithium levels by reducing renal clearance", "clinical_action": "Monitor lithium levels within 1 week. Dose adjustment may be needed."},
    {"drug_a": "lithium", "drug_b": "losartan", "severity": "high", "effect": "ARBs increase lithium levels", "clinical_action": "Monitor lithium levels within 1 week."},
    {"drug_a": "lithium", "drug_b": "furosemide", "severity": "high", "effect": "Diuretics increase lithium levels by reducing renal clearance", "clinical_action": "Monitor lithium levels. Avoid dehydration."},
    {"drug_a": "lithium", "drug_b": "bendroflumethiazide", "severity": "high", "effect": "Thiazides increase lithium levels", "clinical_action": "Monitor lithium levels closely. Consider dose reduction."},
    # --- Digoxin ---
    {"drug_a": "digoxin", "drug_b": "amiodarone", "severity": "critical", "effect": "Amiodarone increases digoxin levels by 70-100% - toxicity risk", "clinical_action": "Halve digoxin dose when starting amiodarone. Monitor levels."},
    {"drug_a": "digoxin", "drug_b": "clarithromycin", "severity": "high", "effect": "Macrolides increase digoxin absorption - toxicity risk", "clinical_action": "Monitor digoxin levels. Use alternative antibiotic if possible."},
    {"drug_a": "digoxin", "drug_b": "furosemide", "severity": "high", "effect": "Diuretic-induced hypokalaemia increases sensitivity to digoxin toxicity", "clinical_action": "Monitor potassium. Supplement potassium or add potassium-sparing diuretic."},
    {"drug_a": "digoxin", "drug_b": "verapamil", "severity": "critical", "effect": "Verapamil increases digoxin levels and additive AV nodal depression", "clinical_action": "Avoid combination. If essential, halve digoxin dose."},
    # --- Statins ---
    {"drug_a": "simvastatin", "drug_b": "amlodipine", "severity": "high", "effect": "Amlodipine increases simvastatin exposure - myopathy/rhabdomyolysis risk", "clinical_action": "Max simvastatin 20mg with amlodipine. Consider atorvastatin instead."},
    {"drug_a": "simvastatin", "drug_b": "clarithromycin", "severity": "critical", "effect": "CYP3A4 inhibition - rhabdomyolysis risk", "clinical_action": "Withhold statin during macrolide course."},
    {"drug_a": "simvastatin", "drug_b": "erythromycin", "severity": "critical", "effect": "CYP3A4 inhibition - rhabdomyolysis risk", "clinical_action": "Withhold statin during antibiotic course."},
    {"drug_a": "atorvastatin", "drug_b": "clarithromycin", "severity": "high", "effect": "CYP3A4 inhibition increases statin exposure - myopathy risk", "clinical_action": "Withhold statin during macrolide course. Or use azithromycin (no interaction)."},
    {"drug_a": "simvastatin", "drug_b": "grapefruit", "severity": "moderate", "effect": "Grapefruit inhibits CYP3A4 - increases simvastatin levels", "clinical_action": "Avoid grapefruit with simvastatin. Atorvastatin less affected."},
    # --- Beta-blockers ---
    {"drug_a": "bisoprolol", "drug_b": "verapamil", "severity": "critical", "effect": "Additive AV nodal depression - risk of heart block, asystole", "clinical_action": "NEVER co-prescribe beta-blocker + verapamil IV. Oral combination only with extreme caution."},
    {"drug_a": "bisoprolol", "drug_b": "diltiazem", "severity": "high", "effect": "Additive bradycardia and AV block risk", "clinical_action": "Avoid combination. If essential, monitor ECG and heart rate closely."},
    {"drug_a": "bisoprolol", "drug_b": "amlodipine", "severity": "moderate", "effect": "Additive hypotension and bradycardia", "clinical_action": "Monitor blood pressure and heart rate. Usually well-tolerated."},
    # --- Clopidogrel ---
    {"drug_a": "clopidogrel", "drug_b": "omeprazole", "severity": "high", "effect": "Omeprazole inhibits CYP2C19, reducing clopidogrel activation - reduced antiplatelet effect", "clinical_action": "Use lansoprazole or pantoprazole instead (no CYP2C19 interaction)."},
    {"drug_a": "clopidogrel", "drug_b": "esomeprazole", "severity": "high", "effect": "Esomeprazole inhibits CYP2C19, reducing clopidogrel activation", "clinical_action": "Use lansoprazole or pantoprazole instead."},
    # --- Quinolones ---
    {"drug_a": "ciprofloxacin", "drug_b": "prednisolone", "severity": "high", "effect": "Fluoroquinolone + corticosteroid - increased tendon rupture risk (MHRA warning)", "clinical_action": "Avoid combination. Use alternative antibiotic."},
    {"drug_a": "ciprofloxacin", "drug_b": "tizanidine", "severity": "critical", "effect": "Ciprofloxacin inhibits CYP1A2 - tizanidine levels increase 10-fold", "clinical_action": "Contraindicated. Use alternative antibiotic."},
    {"drug_a": "ciprofloxacin", "drug_b": "theophylline", "severity": "critical", "effect": "Ciprofloxacin inhibits CYP1A2 - theophylline toxicity risk", "clinical_action": "Monitor theophylline levels. Reduce dose or use alternative antibiotic."},
    {"drug_a": "ciprofloxacin", "drug_b": "warfarin", "severity": "high", "effect": "Ciprofloxacin increases warfarin effect - bleeding risk", "clinical_action": "Monitor INR within 3 days."},
    # --- Benzodiazepines + Opioids ---
    {"drug_a": "diazepam", "drug_b": "codeine", "severity": "critical", "effect": "CNS and respiratory depression - benzodiazepine + opioid (MHRA warning)", "clinical_action": "Avoid combination. If essential, use lowest doses for shortest duration."},
    {"drug_a": "diazepam", "drug_b": "morphine", "severity": "critical", "effect": "CNS and respiratory depression - high mortality risk", "clinical_action": "Avoid combination. Specialist supervision only."},
    {"drug_a": "diazepam", "drug_b": "tramadol", "severity": "critical", "effect": "CNS depression + serotonin syndrome risk", "clinical_action": "Avoid combination."},
    {"drug_a": "diazepam", "drug_b": "gabapentin", "severity": "high", "effect": "Additive CNS depression - respiratory depression risk (MHRA 2019)", "clinical_action": "Avoid combination. If essential, monitor closely."},
    {"drug_a": "diazepam", "drug_b": "pregabalin", "severity": "high", "effect": "Additive CNS depression - respiratory depression risk (MHRA 2019)", "clinical_action": "Avoid combination."},
    # --- Allopurinol ---
    {"drug_a": "allopurinol", "drug_b": "azathioprine", "severity": "critical", "effect": "Allopurinol inhibits azathioprine metabolism - risk of fatal myelosuppression", "clinical_action": "If essential, reduce azathioprine dose by 75%. Specialist supervision only."},
    {"drug_a": "allopurinol", "drug_b": "mercaptopurine", "severity": "critical", "effect": "Allopurinol inhibits mercaptopurine metabolism - severe myelosuppression", "clinical_action": "Reduce mercaptopurine dose by 75%."},
    # --- Colchicine ---
    {"drug_a": "colchicine", "drug_b": "clarithromycin", "severity": "critical", "effect": "CYP3A4 and P-gp inhibition - colchicine toxicity (potentially fatal)", "clinical_action": "Avoid combination. Use alternative antibiotic."},
    {"drug_a": "colchicine", "drug_b": "erythromycin", "severity": "critical", "effect": "CYP3A4 inhibition - colchicine toxicity risk", "clinical_action": "Avoid combination."},
    {"drug_a": "colchicine", "drug_b": "simvastatin", "severity": "high", "effect": "Increased risk of myopathy/rhabdomyolysis", "clinical_action": "Monitor for muscle symptoms."},
    # --- Insulin / Diabetes ---
    {"drug_a": "insulin glargine", "drug_b": "gliclazide", "severity": "high", "effect": "Additive hypoglycaemia risk - insulin + sulfonylurea", "clinical_action": "Monitor glucose closely. Reduce sulfonylurea dose when starting insulin."},
    {"drug_a": "gliclazide", "drug_b": "fluconazole", "severity": "high", "effect": "Fluconazole inhibits CYP2C9 - increased gliclazide levels, hypoglycaemia risk", "clinical_action": "Monitor glucose daily. Consider dose reduction."},
    {"drug_a": "metformin", "drug_b": "gliclazide", "severity": "moderate", "effect": "Additive hypoglycaemia risk, but generally well-tolerated combination", "clinical_action": "Standard combination. Monitor for hypoglycaemia."},
    {"drug_a": "empagliflozin", "drug_b": "furosemide", "severity": "moderate", "effect": "Additive diuretic effect - risk of dehydration and hypotension", "clinical_action": "Monitor hydration status and blood pressure."},
    # --- Corticosteroids ---
    {"drug_a": "prednisolone", "drug_b": "ibuprofen", "severity": "high", "effect": "Increased GI bleeding risk - corticosteroid + NSAID", "clinical_action": "Avoid combination. If essential, add PPI gastroprotection."},
    {"drug_a": "prednisolone", "drug_b": "naproxen", "severity": "high", "effect": "Increased GI bleeding risk - corticosteroid + NSAID", "clinical_action": "Avoid NSAID. Use paracetamol."},
    {"drug_a": "prednisolone", "drug_b": "warfarin", "severity": "moderate", "effect": "Corticosteroids may increase INR unpredictably", "clinical_action": "Monitor INR within 1 week of starting/stopping prednisolone."},
    {"drug_a": "prednisolone", "drug_b": "metformin", "severity": "moderate", "effect": "Corticosteroids increase blood glucose - may worsen diabetic control", "clinical_action": "Monitor blood glucose. May need to increase metformin dose."},
    # --- Antiepileptics ---
    {"drug_a": "carbamazepine", "drug_b": "fluoxetine", "severity": "high", "effect": "Fluoxetine inhibits CYP3A4 - increased carbamazepine levels, toxicity risk", "clinical_action": "Monitor carbamazepine levels. Consider alternative antidepressant."},
    {"drug_a": "carbamazepine", "drug_b": "clarithromycin", "severity": "critical", "effect": "CYP3A4 inhibition - carbamazepine toxicity", "clinical_action": "Avoid combination. Use azithromycin."},
    {"drug_a": "carbamazepine", "drug_b": "oral contraceptive", "severity": "critical", "effect": "CYP3A4 induction - reduced contraceptive efficacy, pregnancy risk", "clinical_action": "Use alternative contraception (IUD, depo). Not effective with oral pill."},
    {"drug_a": "phenytoin", "drug_b": "fluoxetine", "severity": "high", "effect": "Fluoxetine inhibits phenytoin metabolism - toxicity risk", "clinical_action": "Monitor phenytoin levels."},
    {"drug_a": "gabapentin", "drug_b": "morphine", "severity": "high", "effect": "Additive CNS and respiratory depression", "clinical_action": "Use lowest effective doses. Monitor for sedation."},
    {"drug_a": "pregabalin", "drug_b": "morphine", "severity": "high", "effect": "Additive CNS and respiratory depression (MHRA warning)", "clinical_action": "Avoid combination if possible. Monitor closely."},
    # --- Thyroid ---
    {"drug_a": "levothyroxine", "drug_b": "omeprazole", "severity": "moderate", "effect": "PPIs reduce levothyroxine absorption", "clinical_action": "Take levothyroxine 4 hours before PPI. Monitor TSH."},
    {"drug_a": "levothyroxine", "drug_b": "iron", "severity": "moderate", "effect": "Iron reduces levothyroxine absorption by forming complexes", "clinical_action": "Separate doses by at least 4 hours."},
    {"drug_a": "levothyroxine", "drug_b": "calcium", "severity": "moderate", "effect": "Calcium reduces levothyroxine absorption", "clinical_action": "Separate doses by at least 4 hours."},
    # --- Aminoglycosides ---
    {"drug_a": "gentamicin", "drug_b": "furosemide", "severity": "critical", "effect": "Additive ototoxicity and nephrotoxicity", "clinical_action": "Avoid combination if possible. Monitor renal function and drug levels."},
    {"drug_a": "gentamicin", "drug_b": "vancomycin", "severity": "high", "effect": "Additive nephrotoxicity", "clinical_action": "Monitor renal function daily. Monitor drug levels."},
    # --- Potassium ---
    {"drug_a": "spironolactone", "drug_b": "potassium supplements", "severity": "critical", "effect": "Life-threatening hyperkalaemia - potassium-sparing diuretic + potassium", "clinical_action": "NEVER co-prescribe without close monitoring. Usually contraindicated."},
    {"drug_a": "ramipril", "drug_b": "potassium supplements", "severity": "high", "effect": "Hyperkalaemia risk - ACE inhibitors promote potassium retention", "clinical_action": "Avoid routine potassium supplementation with ACE inhibitors."},
    # --- Antiplatelets ---
    {"drug_a": "aspirin", "drug_b": "ibuprofen", "severity": "high", "effect": "Ibuprofen antagonises aspirin's antiplatelet effect if taken first; also increased GI bleeding", "clinical_action": "Take aspirin 30 minutes before ibuprofen. Prefer paracetamol."},
    {"drug_a": "aspirin", "drug_b": "clopidogrel", "severity": "moderate", "effect": "Dual antiplatelet - increased bleeding risk but often indicated post-ACS/PCI", "clinical_action": "Only use dual antiplatelet when indicated. Add PPI gastroprotection."},
    # --- Misc common interactions ---
    {"drug_a": "doxycycline", "drug_b": "antacids", "severity": "moderate", "effect": "Antacids chelate doxycycline, reducing absorption", "clinical_action": "Separate doses by at least 2 hours."},
    {"drug_a": "doxycycline", "drug_b": "iron", "severity": "moderate", "effect": "Iron chelates doxycycline, reducing absorption", "clinical_action": "Separate doses by at least 2-3 hours."},
    {"drug_a": "ciprofloxacin", "drug_b": "iron", "severity": "moderate", "effect": "Iron chelates ciprofloxacin, reducing absorption", "clinical_action": "Separate doses by at least 2 hours."},
    {"drug_a": "nitrofurantoin", "drug_b": "antacids", "severity": "moderate", "effect": "Reduced nitrofurantoin absorption with magnesium-containing antacids", "clinical_action": "Avoid co-administration."},
    {"drug_a": "amlodipine", "drug_b": "simvastatin", "severity": "high", "effect": "Amlodipine increases simvastatin exposure via CYP3A4 - myopathy risk", "clinical_action": "Max simvastatin 20mg. Consider switching to atorvastatin."},
    {"drug_a": "furosemide", "drug_b": "ramipril", "severity": "moderate", "effect": "First-dose hypotension when ACE inhibitor added to diuretic", "clinical_action": "Start ACE inhibitor at low dose. Consider withholding diuretic on first day."},
    {"drug_a": "omeprazole", "drug_b": "methotrexate", "severity": "moderate", "effect": "PPIs may reduce methotrexate clearance at high doses", "clinical_action": "Monitor for methotrexate toxicity at high doses."},
    {"drug_a": "bendroflumethiazide", "drug_b": "lithium", "severity": "high", "effect": "Thiazides increase lithium levels - toxicity risk", "clinical_action": "Monitor lithium levels closely."},
    {"drug_a": "amiodarone", "drug_b": "simvastatin", "severity": "critical", "effect": "Amiodarone inhibits CYP3A4 - rhabdomyolysis risk with simvastatin", "clinical_action": "Max simvastatin 20mg. Prefer atorvastatin (lower interaction risk)."},
    {"drug_a": "verapamil", "drug_b": "simvastatin", "severity": "critical", "effect": "Verapamil inhibits CYP3A4 - rhabdomyolysis risk", "clinical_action": "Max simvastatin 10mg with verapamil. Prefer atorvastatin."},
    {"drug_a": "diltiazem", "drug_b": "simvastatin", "severity": "high", "effect": "Diltiazem inhibits CYP3A4 - increased simvastatin exposure", "clinical_action": "Max simvastatin 20mg with diltiazem."},
    {"drug_a": "tramadol", "drug_b": "carbamazepine", "severity": "moderate", "effect": "Carbamazepine induces CYP3A4 - reduced tramadol efficacy and seizure threshold lowered", "clinical_action": "Use alternative analgesic. Avoid in epilepsy."},
    {"drug_a": "co-amoxiclav", "drug_b": "warfarin", "severity": "moderate", "effect": "May increase INR unpredictably", "clinical_action": "Monitor INR within 3-5 days."},
    {"drug_a": "flucloxacillin", "drug_b": "warfarin", "severity": "moderate", "effect": "May reduce warfarin effect", "clinical_action": "Monitor INR."},
]


_PAIR_INDEX: dict[tuple[str, str], dict] = {}
for _entry in INTERACTION_DB:
    a = _entry["drug_a"].lower()
    b = _entry["drug_b"].lower()
    _PAIR_INDEX[(a, b)] = _entry
    _PAIR_INDEX[(b, a)] = _entry


def check_interactions(medications: list[str]) -> list[dict]:
    """Check all medication pairs for known interactions.

    Args:
        medications: List of drug names (case-insensitive).

    Returns:
        List of interaction dicts with drug_a, drug_b, severity, effect, clinical_action.
    """
    meds = [m.lower().strip() for m in medications]
    found: list[dict] = []
    checked: set[tuple[str, str]] = set()

    for i, med_a in enumerate(meds):
        for med_b in meds[i + 1:]:
            pair = (min(med_a, med_b), max(med_a, med_b))
            if pair in checked:
                continue
            checked.add(pair)

            interaction = _PAIR_INDEX.get((med_a, med_b))
            if interaction:
                found.append(interaction)

    severity_order = {"critical": 0, "high": 1, "moderate": 2}
    found.sort(key=lambda x: severity_order.get(x["severity"], 3))

    logger.info(
        f"Drug interaction check: {len(meds)} medications, {len(found)} interactions found"
    )
    return found
