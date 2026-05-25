#!/usr/bin/env python3
"""Run all 8 transcripts through all 4 models. Produce comparison data for dissertation.

Output: docs/model_comparison_results.json + docs/model_comparison_report.md
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clinical.evaluator import ClinicalEvaluator, QualityScore
from src.soap_generation.generator import SOAPGenerator, SOAPModel, display_name

TRANSCRIPTS = {
    "Lower Back Pain": (
        "Doctor: Good morning, what brings you in today?\n"
        "Patient: I've been having terrible lower back pain for about three weeks now. "
        "It started after I was lifting some heavy boxes at work.\n"
        "Doctor: Can you describe the pain? Is it sharp, dull, or aching?\n"
        "Patient: It's a dull ache most of the time but gets sharp when I bend forward. "
        "It sometimes goes down my left leg.\n"
        "Doctor: Any numbness or tingling in your legs?\n"
        "Patient: Yes, I get pins and needles in my left foot sometimes.\n"
        "Doctor: Any problems with your bladder or bowels?\n"
        "Patient: No, nothing like that.\n"
        "Doctor: Are you taking anything for the pain?\n"
        "Patient: Just paracetamol, two tablets four times a day, but it's not really helping.\n"
        "Doctor: Let me examine you. Stand up for me... Right, I can see some muscle spasm "
        "on the left side. Straight leg raise is positive on the left at 40 degrees. "
        "Reflexes and power are normal. Temperature is 36.8.\n"
        "Doctor: Based on my examination, this looks like a lumbar disc protrusion with "
        "left-sided sciatica. The good news is your neurological examination is largely normal. "
        "I'm going to prescribe naproxen 500mg twice daily with food, and refer you for "
        "physiotherapy. Continue the paracetamol as well.\n"
        "Doctor: I'd like you to come back in four weeks. If the pain gets significantly worse, "
        "or if you develop any weakness in your legs, or any bladder or bowel problems, "
        "please come to A&E immediately or call 999. Those would be red flag symptoms."
    ),
    "Persistent Cough": (
        "Doctor: Hello, how can I help you today?\n"
        "Patient: I've had this cough that won't go away. It's been about six weeks now.\n"
        "Doctor: Is it a dry cough or are you bringing anything up?\n"
        "Patient: Mostly dry, but sometimes I get a bit of clear phlegm in the mornings.\n"
        "Doctor: Any blood in the phlegm?\n"
        "Patient: No, no blood.\n"
        "Doctor: Any fever, night sweats, or weight loss?\n"
        "Patient: No fever. I have lost a bit of weight actually, maybe half a stone. "
        "I thought it was just stress.\n"
        "Doctor: Are you a smoker?\n"
        "Patient: Yes, I've smoked about 20 a day for 25 years.\n"
        "Doctor: Any shortness of breath?\n"
        "Patient: Yes, especially going up stairs. That's got worse recently.\n"
        "Doctor: Let me listen to your chest. There are some crackles at the right base. "
        "Blood pressure is 138 over 85. Oxygen saturation is 96%. Temperature 37.1.\n"
        "Doctor: Given your smoking history, the persistent cough, weight loss, and what "
        "I'm hearing on examination, I want to arrange an urgent chest X-ray. "
        "I'm going to refer you on the two-week wait pathway.\n"
        "Doctor: I'll also do some blood tests - full blood count, CRP, and liver function. "
        "If you develop haemoptysis or worsening breathlessness, "
        "please call 111 or attend A&E."
    ),
    "Diabetes Review": (
        "Doctor: Good afternoon. You're here for your annual diabetes review.\n"
        "Patient: Yes, that's right.\n"
        "Doctor: How have you been managing? Any problems with your medications?\n"
        "Patient: Not too bad. I'm taking my metformin 1g twice daily and the "
        "atorvastatin 20mg at night. Sometimes the metformin upsets my stomach.\n"
        "Doctor: Are you checking your blood sugars at home?\n"
        "Patient: Yes, they're usually between 7 and 10 before breakfast.\n"
        "Doctor: Your HbA1c has come down to 58 mmol/mol from 65 last time. "
        "Cholesterol is 4.2. Kidney function is normal. eGFR is 85.\n"
        "Doctor: Blood pressure today is 134 over 82. BMI is 31.2. "
        "Feet examination: pulses present, sensation intact with monofilament.\n"
        "Doctor: Overall you're doing well. Keep medications the same. "
        "Take metformin with food for the stomach issues. "
        "Review in six months for repeat HbA1c. "
        "If blood sugars run consistently above 12, book an urgent appointment."
    ),
    "Mental Health - Anxiety": (
        "Doctor: Hello, I can see you'd like to talk about how you've been feeling.\n"
        "Patient: Yes, I've been really struggling. I feel anxious all the time, "
        "I can't sleep, and I'm finding it hard to concentrate at work.\n"
        "Doctor: How long has this been going on?\n"
        "Patient: About two months. It started with problems at work - redundancies.\n"
        "Doctor: Tell me about the anxiety. What does it feel like?\n"
        "Patient: My heart races, I feel sick, I get headaches. "
        "I've had a few panic attacks.\n"
        "Doctor: Are you sleeping okay?\n"
        "Patient: No, about four hours a night.\n"
        "Doctor: How about your mood? Any thoughts of harming yourself?\n"
        "Patient: My mood is low but no thoughts of harming myself.\n"
        "Doctor: I think you're experiencing generalised anxiety disorder. "
        "I'm going to refer you to IAPT for CBT therapy. "
        "Reduce the alcohol as it makes anxiety worse. "
        "If things don't improve in four to six weeks, we can discuss sertraline.\n"
        "Doctor: If you experience any thoughts of self-harm, "
        "please call 111, the Samaritans on 116 123, or attend A&E."
    ),
    "Chest Infection": (
        "Doctor: Good morning. What's the matter today?\n"
        "Patient: I've been feeling really unwell for about five days. "
        "Terrible cough with green phlegm, and I feel hot and cold.\n"
        "Doctor: Any chest pain or shortness of breath?\n"
        "Patient: A bit of chest pain when I cough, and I'm a bit wheezy.\n"
        "Doctor: Past medical history?\n"
        "Patient: I've got asthma. I use a salbutamol inhaler and a brown preventer.\n"
        "Doctor: Temperature is 38.4. BP 125 over 78. Sats 97%. "
        "On auscultation, coarse crackles and wheeze at the right lower zone.\n"
        "Doctor: This looks like a lower respiratory tract infection. "
        "Prescribing amoxicillin 500mg three times daily for five days. "
        "Keep using your salbutamol. Take paracetamol for the fever.\n"
        "Doctor: Come back in one week. If you become more breathless, "
        "the fever doesn't settle in 48 hours, or you cough up blood, "
        "call 111 or go to A&E."
    ),
    "Hypertension Review": (
        "Doctor: Hello, you're here for your blood pressure check.\n"
        "Patient: Yes, the nurse asked me to come back because it was high.\n"
        "Doctor: It's 158 over 96 today. Ambulatory readings averaged 152 over 92.\n"
        "Patient: Is that bad?\n"
        "Doctor: Higher than we'd like - we aim for below 140 over 90. "
        "Any family history of heart disease?\n"
        "Patient: My dad had a heart attack at 55.\n"
        "Doctor: I'd like to start you on ramipril 2.5mg once daily. "
        "We'll check kidney function in two weeks. "
        "Reduce salt intake, 30 minutes exercise most days.\n"
        "Doctor: Come back in four weeks to recheck. "
        "If you get a persistent dry cough, let us know. "
        "If you develop sudden severe headache, chest pain, or visual changes, "
        "call 999 immediately."
    ),
    "Paediatric Ear Infection": (
        "Doctor: Hello mum, and hello there! What's been going on?\n"
        "Patient's mother: He's been really unsettled for two days. "
        "Pulling at his right ear and running a temperature.\n"
        "Doctor: How high has the temperature been?\n"
        "Patient's mother: Up to 38.9. Calpol brings it down a bit.\n"
        "Doctor: Any cough, runny nose, or vomiting?\n"
        "Patient's mother: He had a cold last week. No vomiting.\n"
        "Doctor: Temperature now is 38.2. Right tympanic membrane is red and bulging. "
        "Left ear normal. Throat red but no pus.\n"
        "Doctor: This is acute otitis media. Most clear within three days. "
        "Continue paracetamol and ibuprofen for pain and fever. "
        "I'll give a delayed prescription for amoxicillin - use it if not improving after three days.\n"
        "Doctor: If he develops a high fever that won't come down, "
        "becomes very drowsy, or you see fluid from the ear, "
        "bring him straight back or call 111."
    ),
    "Skin Lesion Assessment": (
        "Doctor: You've come about a mole that's been changing?\n"
        "Patient: Yes, on my back. My wife noticed it's got bigger "
        "and the colour has changed - different shades of brown now.\n"
        "Doctor: How long has it been changing?\n"
        "Patient: About three months.\n"
        "Doctor: Any itching, bleeding, or crusting?\n"
        "Patient: It's been a bit itchy. No bleeding.\n"
        "Doctor: It's about 8mm, asymmetric, irregular borders, "
        "multiple colours including a bluish area. "
        "Irregular pigment network on dermatoscopy.\n"
        "Doctor: This meets several ABCDE criteria for a suspicious mole. "
        "I'm referring urgently on the two-week wait pathway to dermatology "
        "for excision biopsy.\n"
        "Doctor: Many suspicious moles turn out benign, but we must check properly. "
        "If it starts bleeding or changes rapidly before your appointment, "
        "please call us or attend A&E."
    ),
}

MODELS = [
    SOAPModel.PRAXIS_PRIMARY,
    SOAPModel.MED42,
    SOAPModel.MEDITRON,
    SOAPModel.MISTRAL,
]


def main():
    evaluator = ClinicalEvaluator()
    output_dir = Path(__file__).resolve().parent.parent / "docs"
    output_dir.mkdir(exist_ok=True)

    all_results = []
    model_totals = {m.value: {"scores": [], "times": [], "tokens": []} for m in MODELS}

    gen = SOAPGenerator()
    available = gen.list_models()
    print(f"Ollama models: {available}")
    print(f"Running {len(TRANSCRIPTS)} transcripts x {len(MODELS)} models = {len(TRANSCRIPTS) * len(MODELS)} generations\n")

    for case_name, transcript in TRANSCRIPTS.items():
        print(f"\n{'='*70}")
        print(f"  {case_name}")
        print(f"{'='*70}")

        for model in MODELS:
            model_name = display_name(model.value)
            # Check model is available
            if not any(model.value in m for m in available):
                print(f"  [{model_name}] SKIPPED - not in Ollama")
                continue

            try:
                generator = SOAPGenerator(model=model)
                t0 = time.perf_counter()
                result = generator.generate(transcript)
                wall_time = round(time.perf_counter() - t0, 2)

                score = evaluator.evaluate(
                    result.soap_note,
                    model=result.model,
                    time_secs=result.generation_time_secs,
                    tokens=result.token_count,
                )

                entry = {
                    "case": case_name,
                    "model_id": model.value,
                    "model_display": model_name,
                    "soap_note": result.soap_note,
                    "generation_time_secs": result.generation_time_secs,
                    "wall_time_secs": wall_time,
                    "token_count": result.token_count,
                    "structure_score": round(score.structure_score, 2),
                    "content_score": round(score.content_score, 2),
                    "overall_score": round(score.overall_score, 2),
                    "has_snomed": score.has_snomed_codes,
                    "snomed_count": score.snomed_code_count,
                    "has_safety_netting": score.has_safety_netting,
                    "uses_uk_english": score.uses_uk_english,
                }
                all_results.append(entry)

                model_totals[model.value]["scores"].append(score.overall_score)
                model_totals[model.value]["times"].append(result.generation_time_secs)
                model_totals[model.value]["tokens"].append(result.token_count)

                print(f"  [{model_name:<40}] overall={score.overall_score:.2f}  struct={score.structure_score:.2f}  content={score.content_score:.2f}  snomed={score.snomed_code_count}  time={result.generation_time_secs}s  tokens={result.token_count}")

            except Exception as e:
                print(f"  [{model_name}] ERROR: {e}")

    # Save raw JSON
    json_path = output_dir / "model_comparison_results.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nRaw results saved: {json_path}")

    # Generate markdown report
    md_lines = [
        "# PRAXIS Model Comparison Results",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Consultations:** {len(TRANSCRIPTS)}",
        f"**Models:** {len(MODELS)}",
        f"**Total generations:** {len(all_results)}",
        "",
        "## Summary by Model",
        "",
        "| Model | Avg Overall | Avg Structure | Avg Content | Avg Time (s) | Avg Tokens | SNOMED Rate | UK English Rate | Safety-Net Rate |",
        "|-------|------------|--------------|-------------|-------------|-----------|-------------|----------------|-----------------|",
    ]

    for model in MODELS:
        mid = model.value
        entries = [r for r in all_results if r["model_id"] == mid]
        if not entries:
            continue
        n = len(entries)
        avg_overall = sum(r["overall_score"] for r in entries) / n
        avg_struct = sum(r["structure_score"] for r in entries) / n
        avg_content = sum(r["content_score"] for r in entries) / n
        avg_time = sum(r["generation_time_secs"] for r in entries) / n
        avg_tokens = sum(r["token_count"] for r in entries) / n
        snomed_rate = sum(1 for r in entries if r["has_snomed"]) / n * 100
        uk_rate = sum(1 for r in entries if r["uses_uk_english"]) / n * 100
        safety_rate = sum(1 for r in entries if r["has_safety_netting"]) / n * 100
        md_lines.append(
            f"| {display_name(mid)} | {avg_overall:.2f} | {avg_struct:.2f} | {avg_content:.2f} | "
            f"{avg_time:.1f} | {avg_tokens:.0f} | {snomed_rate:.0f}% | {uk_rate:.0f}% | {safety_rate:.0f}% |"
        )

    md_lines += [
        "",
        "## Results by Consultation",
        "",
    ]

    for case_name in TRANSCRIPTS:
        md_lines.append(f"### {case_name}")
        md_lines.append("")
        md_lines.append("| Model | Overall | Structure | Content | SNOMED | Safety-Net | UK English | Time (s) | Tokens |")
        md_lines.append("|-------|---------|-----------|---------|--------|-----------|-----------|---------|--------|")
        entries = [r for r in all_results if r["case"] == case_name]
        for r in entries:
            snomed_str = f"{r['snomed_count']}" if r["has_snomed"] else "No"
            safety_str = "Yes" if r["has_safety_netting"] else "No"
            uk_str = "Yes" if r["uses_uk_english"] else "No"
            md_lines.append(
                f"| {r['model_display']} | {r['overall_score']:.2f} | {r['structure_score']:.2f} | "
                f"{r['content_score']:.2f} | {snomed_str} | {safety_str} | {uk_str} | "
                f"{r['generation_time_secs']} | {r['token_count']} |"
            )
        md_lines.append("")

    # Winning model per case
    md_lines += [
        "## Best Model per Consultation",
        "",
        "| Consultation | Best Model | Score |",
        "|-------------|-----------|-------|",
    ]
    for case_name in TRANSCRIPTS:
        entries = [r for r in all_results if r["case"] == case_name]
        if entries:
            best = max(entries, key=lambda r: r["overall_score"])
            md_lines.append(f"| {case_name} | {best['model_display']} | {best['overall_score']:.2f} |")

    md_lines += [
        "",
        "---",
        "*Generated by PRAXIS model comparison pipeline*",
    ]

    md_path = output_dir / "model_comparison_report.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Report saved: {md_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for model in MODELS:
        mid = model.value
        entries = [r for r in all_results if r["model_id"] == mid]
        if not entries:
            continue
        n = len(entries)
        avg = sum(r["overall_score"] for r in entries) / n
        avg_t = sum(r["generation_time_secs"] for r in entries) / n
        snomed_pct = sum(1 for r in entries if r["has_snomed"]) / n * 100
        print(f"  {display_name(mid):<42} avg={avg:.2f}  time={avg_t:.1f}s  snomed={snomed_pct:.0f}%")


if __name__ == "__main__":
    main()
