"""Synthetic data + augmentation + retrain. Superseded by retrain_full_data.py."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset
from loguru import logger

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# paths
SYNTH_DIR = Path("data/raw/synthetic_british")
AUGMENTED_DIR = Path("data/processed/augmented_primock57")
COMBINED_DIR = Path("data/processed/combined_v3")
PRIMOCK_DIR = Path("data/processed/primock57")

WHISPER_V3_DIR = "data/models/whisper-british-medical-v3"
MEDASR_V2_DIR = "data/models/medasr-british-v2"

# voices
VOICES = [
    ("en-GB-SoniaNeural", "female"),
    ("en-GB-RyanNeural", "male"),
]

# ══════════════════════════════════════════════════════════════
# STEP 1: Generate synthetic British medical sentences
# ══════════════════════════════════════════════════════════════

MEDICAL_SENTENCES = {
    "gp_consultation": [
        "The patient reports chest pain radiating to the left arm",
        "She describes the pain as a dull ache that comes and goes",
        "He mentions feeling short of breath when climbing stairs",
        "The patient has been experiencing persistent headaches for two weeks",
        "She reports nausea and vomiting since yesterday morning",
        "He complains of lower back pain that worsens with movement",
        "The patient describes a burning sensation when passing urine",
        "She has noticed blood in her stool over the past three days",
        "He reports difficulty sleeping and feeling anxious most days",
        "The patient mentions a persistent dry cough lasting four weeks",
        "She describes pins and needles in both hands especially at night",
        "He reports dizziness when standing up quickly",
        "The patient has been feeling extremely tired for the past month",
        "She mentions her appetite has decreased significantly",
        "He describes intermittent abdominal pain after eating",
        "The patient reports joint stiffness particularly in the morning",
        "She has noticed swelling in her ankles over the past week",
        "He mentions a rash on his forearms that appeared three days ago",
        "The patient describes episodes of heart palpitations",
        "She reports blurred vision in her right eye",
        "He complains of a sore throat and difficulty swallowing",
        "The patient mentions frequent urination during the night",
        "She describes a throbbing pain behind her left eye",
        "He reports weight loss of about half a stone over six weeks",
        "The patient has been experiencing muscle cramps in her calves",
        "She mentions feeling lightheaded after meals",
        "He describes a tight feeling in his chest when exercising",
        "The patient reports numbness running down her left leg",
        "She has noticed her hair has been thinning recently",
        "He mentions his sleep has been quite poor lately",
    ],
    "drug_prescriptions": [
        "Start amoxicillin 500mg three times daily for seven days",
        "I'm going to prescribe omeprazole 20mg once daily before breakfast",
        "Take paracetamol 1 gram four times a day as needed for pain",
        "We'll start you on metformin 500mg twice daily with meals",
        "I'm prescribing prednisolone 40mg daily for five days then stop",
        "Start atorvastatin 20mg once daily in the evening",
        "Take ibuprofen 400mg three times daily with food",
        "I'll prescribe ramipril 2.5mg once daily for your blood pressure",
        "Start sertraline 50mg once daily in the morning",
        "Take co-amoxiclav 625mg three times daily for five days",
        "I'm starting you on amlodipine 5mg once daily",
        "Use the salbutamol inhaler two puffs as needed for wheeze",
        "Take lansoprazole 30mg once daily before food",
        "I'll prescribe fluoxetine 20mg once daily in the morning",
        "Start bisoprolol 2.5mg once daily for your heart rate",
        "Take naproxen 500mg twice daily with food",
        "I'm prescribing doxycycline 100mg twice daily for seven days",
        "Use budesonide inhaler 200 micrograms twice daily",
        "Take gabapentin 300mg three times daily for the nerve pain",
        "Start apixaban 5mg twice daily as a blood thinner",
        "I'll prescribe trimethoprim 200mg twice daily for three days",
        "Take duloxetine 30mg once daily for two weeks then increase to 60mg",
        "I'm starting you on empagliflozin 10mg once daily",
        "Use the tiotropium inhaler once daily in the morning",
        "Take nitrofurantoin 100mg twice daily for seven days",
        "Start pregabalin 75mg twice daily and we'll review in two weeks",
        "I'll prescribe cefalexin 500mg three times daily for five days",
        "Take mirtazapine 15mg at bedtime",
        "Start levothyroxine 50 micrograms once daily on an empty stomach",
        "I'm prescribing codeine 30mg four times daily for severe pain",
    ],
    "history_taking": [
        "Past medical history includes type 2 diabetes and hypertension",
        "The patient has a history of asthma diagnosed in childhood",
        "Family history is significant for ischaemic heart disease",
        "She was diagnosed with rheumatoid arthritis five years ago",
        "No known drug allergies documented in the records",
        "Previous surgical history includes appendicectomy in 2015",
        "The patient has a history of depression managed with medication",
        "He was diagnosed with COPD three years ago following spirometry",
        "Family history of breast cancer in her mother and maternal aunt",
        "Past medical history includes a DVT treated with anticoagulation",
        "The patient has a history of recurrent urinary tract infections",
        "She was diagnosed with coeliac disease after blood tests and biopsy",
        "Previous history of gestational diabetes during her second pregnancy",
        "He has a background of atrial fibrillation diagnosed two years ago",
        "No significant past medical history prior to this presentation",
        "The patient reports allergies to penicillin causing a rash",
        "Family history of type 2 diabetes in both parents",
        "Past surgical history includes a right total knee replacement",
        "She has a history of migraine with aura since her twenties",
        "Previous history of an ectopic pregnancy treated surgically",
        "The patient was previously treated for Helicobacter pylori infection",
        "He has a background history of gout affecting the big toe",
        "Social history includes smoking 10 cigarettes daily for 20 years",
        "She drinks approximately 14 units of alcohol per week",
        "Occupational history as a teacher with no known occupational exposures",
        "The patient is a retired builder with a history of asbestos exposure",
        "He lives alone and is independently mobile",
        "She is the primary carer for her elderly mother",
        "No family history of bowel or ovarian cancer",
        "Previous mental health history includes anxiety and panic attacks",
    ],
    "examination_findings": [
        "On examination, bilateral ankle oedema noted",
        "Blood pressure is 148 over 92, heart rate 78 regular",
        "Chest is clear on auscultation bilaterally",
        "Abdomen is soft and non-tender with no organomegaly",
        "Heart sounds are normal, no murmurs detected",
        "There is a two centimetre lymph node palpable in the left axilla",
        "Throat examination reveals erythema and tonsillar exudate",
        "Skin examination shows an erythematous maculopapular rash on the trunk",
        "Neurological examination is grossly normal",
        "Range of motion in the right shoulder is reduced to 90 degrees abduction",
        "Peripheral pulses are present and equal bilaterally",
        "Body mass index is calculated at 32.4",
        "Oxygen saturations are 97% on room air",
        "Temperature is 38.2 degrees Celsius",
        "Respiratory rate is 18 breaths per minute",
        "There is tenderness over the right iliac fossa on palpation",
        "Ear examination shows a bulging tympanic membrane on the right",
        "Fundoscopy reveals cotton wool spots and flame haemorrhages",
        "Knee examination shows a small effusion on the left side",
        "Skin turgor is reduced suggesting mild dehydration",
        "The thyroid gland feels diffusely enlarged on palpation",
        "Cranial nerve examination is unremarkable",
        "There is crepitus felt over the right knee on flexion",
        "Peak flow reading is 320 litres per minute",
        "The patient appears comfortable at rest but becomes breathless on exertion",
        "Capillary refill time is less than two seconds",
        "There are bibasal crackles heard on auscultation",
        "Abdominal examination reveals a palpable mass in the left lower quadrant",
        "The wound appears clean with no signs of infection",
        "Reflexes are symmetrical and normal throughout",
    ],
    "nhs_specific": [
        "Referred to cardiology for further investigation",
        "I'll do a two week wait referral to the breast clinic",
        "We need to arrange an urgent CT scan of the abdomen",
        "I'll refer you to the musculoskeletal physiotherapy service",
        "Let's arrange an ECG and some blood tests today",
        "I'll book you in for a follow-up appointment in two weeks",
        "We'll request an MRI of the lumbar spine",
        "I'm going to refer you to the community mental health team",
        "You'll need to attend the phlebotomy clinic for your blood tests",
        "I'll send a referral to the pain management clinic",
        "We should check your HbA1c and fasting glucose",
        "I'll arrange a chest X-ray at the hospital",
        "You need to be seen in A&E if the symptoms worsen",
        "The blood results show your CRP is elevated at 45",
        "Your FBC shows a haemoglobin of 110, which is slightly low",
        "LFTs are within normal range",
        "U and Es are satisfactory with normal renal function",
        "TFTs show a slightly elevated TSH at 5.8",
        "I'll refer you to the diabetes specialist nurse",
        "We'll arrange a spirometry test to assess your lung function",
        "Your cholesterol ratio is 5.2 which we need to bring down",
        "I'll make a safeguarding referral given the concerns raised",
        "You're entitled to a free NHS health check at your age",
        "I'll add you to the waiting list for the orthopaedic clinic",
        "We need to do an annual medication review",
        "Your prescription is ready to collect from the pharmacy",
        "I'll send a task to the practice nurse to arrange your smear test",
        "The district nurse will visit you at home for the wound dressing",
        "I'll complete a fit note for two weeks off work",
        "Your QRISK score puts you at moderate cardiovascular risk",
    ],
    "informal_gp": [
        "Right, let's have a look at your blood results",
        "OK so tell me what's been going on",
        "How long has this been troubling you then",
        "And when did you first notice the lump",
        "Have you tried anything for it so far",
        "Let me just check your blood pressure quickly",
        "Pop yourself up on the couch for me",
        "Take a deep breath in and out for me please",
        "Any questions before you go",
        "Right let me just pull up your notes here",
        "So the good news is your results are all normal",
        "I'm not too worried about this but let's keep an eye on it",
        "Let's give it another couple of weeks and see how you get on",
        "If things get worse over the weekend pop down to A and E",
        "I think we should get some bloods done just to be safe",
        "The tablets might make you feel a bit queasy at first",
        "Have you been under a lot of stress recently",
        "Are you still taking the medication we started last time",
        "Let me just feel your tummy if that's alright",
        "And how's your mood been generally",
        "OK I'm going to have a listen to your chest",
        "Have you noticed any change in your bowel habits",
        "Right I think we need to refer you on for this one",
        "The pain should settle down within a few days",
        "Make sure you're drinking plenty of fluids",
        "I'd suggest avoiding alcohol while you're on these tablets",
        "Come back and see me if it hasn't improved in a week",
        "It's nothing to worry about but we should investigate further",
        "Let's try a different medication and see if that helps",
        "I'll give you a sick note for a couple of weeks",
    ],
    "clinical_dialogue_doctor": [
        "Good morning, please take a seat and tell me how I can help today",
        "I understand that must be quite concerning for you",
        "When you say the pain is severe, can you rate it out of ten",
        "Does the pain go anywhere else or stay in one spot",
        "Have you experienced anything like this before",
        "Are you currently taking any medications, including over the counter",
        "Do you have any allergies that we should know about",
        "I'd like to examine you now if that's OK",
        "Everything looks fine on examination which is reassuring",
        "Based on what you've told me I think this is likely a viral infection",
        "I want to rule out anything more serious so we'll do some tests",
        "The results should be back within five working days",
        "I'll call you if there's anything abnormal on the results",
        "Do you have any other concerns you'd like to discuss today",
        "We'll keep this under review and I'll see you again in a month",
        "I think it would be helpful to refer you to a specialist",
        "In the meantime try to rest and take regular paracetamol",
        "If you develop a fever or the symptoms change significantly come straight back",
        "Is there anything else I can help you with today",
        "Take care and don't hesitate to book another appointment if needed",
        "I can see from your records you're due for a medication review",
        "Let me explain what this diagnosis means for you going forward",
        "We need to monitor this regularly so book in every three months",
        "I'm going to increase your dose slightly and see how you respond",
        "This is a common condition and there are several treatment options",
        "I want to make sure we're managing your condition properly",
        "Have you considered any lifestyle changes that might help",
        "Your blood sugar has been creeping up so we need to address that",
        "I'd recommend reducing your salt intake to help with the blood pressure",
        "Let's go through your medications to make sure everything is still appropriate",
    ],
    "clinical_dialogue_patient": [
        "I've been having this pain in my chest for about a week now",
        "It's a sharp pain that gets worse when I breathe in deeply",
        "I've been feeling really tired and run down for months",
        "My joints have been really stiff and painful in the mornings",
        "I've noticed some blood when I go to the toilet",
        "The headaches are getting more frequent and lasting longer",
        "I can't seem to shift this cough, it's been going on for weeks",
        "I've been feeling really anxious and I can't sleep properly",
        "My skin has come up in this rash and it's really itchy",
        "I've been getting these dizzy spells especially in the morning",
        "The medication doesn't seem to be helping with the pain",
        "I've tried ibuprofen but it upsets my stomach",
        "My mum had the same thing and she ended up needing surgery",
        "I'm worried it might be something serious",
        "The pain keeps waking me up in the middle of the night",
        "I've lost about a stone in weight without really trying",
        "I've been getting these hot flushes and night sweats",
        "My ankle has been swollen since I twisted it last week",
        "I've been having trouble with my memory and concentration",
        "The tablets you gave me last time made me feel really sick",
        "I've been under a lot of pressure at work recently",
        "My back has been killing me and I can barely move",
        "I keep getting these sharp pains in my stomach after eating",
        "I've noticed my heart racing for no apparent reason",
        "I can't walk as far as I used to without getting out of breath",
        "My eyes have been really dry and gritty feeling",
        "I've been getting pins and needles in my feet",
        "The cream you prescribed hasn't made any difference",
        "I think I might need my medication changed",
        "I'm having trouble hearing properly especially in noisy places",
    ],
    "assessment_plan": [
        "Assessment: likely viral upper respiratory tract infection",
        "Plan: conservative management with supportive care",
        "Working diagnosis of gastro-oesophageal reflux disease",
        "Differential diagnosis includes pulmonary embolism",
        "Assessment: acute exacerbation of COPD",
        "Plan: commence oral antibiotics and increase inhaler use",
        "Likely musculoskeletal in origin given the examination findings",
        "Assessment: poorly controlled type 2 diabetes",
        "Plan: increase metformin and arrange dietitian referral",
        "Suspected urinary tract infection based on symptoms and dipstick",
        "Assessment: new diagnosis of atrial fibrillation",
        "Plan: start rate control and arrange echocardiogram",
        "Working diagnosis of iron deficiency anaemia",
        "Plan: start oral iron supplementation and investigate cause",
        "Assessment: moderate depressive episode",
        "Plan: start SSRI and refer to talking therapies",
        "Likely osteoarthritis of the right knee",
        "Plan: trial of topical anti-inflammatory and physiotherapy",
        "Assessment: acute otitis media",
        "Plan: safety netting advice and delayed antibiotic prescription",
        "Suspected coeliac disease based on symptoms",
        "Plan: check tissue transglutaminase antibodies and refer if positive",
        "Assessment: hypertension not at target",
        "Plan: add second antihypertensive agent and recheck in four weeks",
        "Working diagnosis of carpal tunnel syndrome",
        "Plan: trial of wrist splints and nerve conduction studies if no improvement",
        "Assessment: stable angina pectoris",
        "Plan: refer cardiology and start GTN spray",
        "Likely benign positional vertigo based on history",
        "Plan: teach Epley manoeuvre and review in two weeks",
    ],
    "results_discussion": [
        "Your haemoglobin is 125 which is within the normal range",
        "The white cell count is slightly raised at 12.3",
        "Your kidney function tests are all normal which is good",
        "The liver function tests show a mildly elevated ALT at 55",
        "Your thyroid function is normal so we can rule that out",
        "The HbA1c has come down from 62 to 54 which is excellent progress",
        "Your cholesterol has improved but the LDL is still a bit high",
        "The chest X-ray came back as normal with no concerning features",
        "Your PSA level is 3.2 which is within the expected range for your age",
        "The urine culture has grown E coli so the antibiotics were appropriate",
        "Your vitamin D level is low at 22 so we need to supplement that",
        "The ECG shows normal sinus rhythm with no abnormalities",
        "Your ferritin is very low at 8 which explains the tiredness",
        "The ultrasound of the abdomen showed a few small gallstones",
        "Your inflammatory markers are back to normal now",
    ],
}

def get_all_sentences() -> list[tuple[str, str]]:
    """Return (sentence, category) pairs."""
    pairs = []
    for category, sentences in MEDICAL_SENTENCES.items():
        for s in sentences:
            pairs.append((s, category))
    return pairs

async def generate_synthetic_audio():
    """Generate TTS audio using edge-tts for all sentences x voices."""
    import edge_tts

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    sentences = get_all_sentences()
    total = len(sentences)
    logger.info(f"STEP 1: Generating {total} sentences x {len(VOICES)} voices = {total * len(VOICES)} audio files")

    metadata = []
    generated = 0

    for idx, (sentence, category) in enumerate(sentences):
        for voice_name, gender in VOICES:
            safe_name = f"{category}_{idx:03d}_{gender}"
            wav_path = SYNTH_DIR / f"{safe_name}.wav"
            mp3_path = SYNTH_DIR / f"{safe_name}.mp3"

            if wav_path.exists():
                metadata.append({
                    "filename": wav_path.name,
                    "text": sentence,
                    "voice": voice_name,
                    "gender": gender,
                    "category": category,
                })
                generated += 1
                continue

            try:
                communicate = edge_tts.Communicate(sentence, voice_name)
                await communicate.save(str(mp3_path))

                # Convert mp3 to wav 16kHz mono
                audio, sr = librosa.load(str(mp3_path), sr=16000, mono=True)
                sf.write(str(wav_path), audio, 16000)
                mp3_path.unlink()  # Remove mp3

                metadata.append({
                    "filename": wav_path.name,
                    "text": sentence,
                    "voice": voice_name,
                    "gender": gender,
                    "category": category,
                })
                generated += 1

            except Exception as e:
                logger.warning(f"Failed to generate {safe_name}: {e}")

        if (idx + 1) % 50 == 0:
            logger.info(f"  Progress: {idx + 1}/{total} sentences ({generated} files)")

    # Save metadata CSV
    csv_path = SYNTH_DIR / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text", "voice", "gender", "category"])
        writer.writeheader()
        writer.writerows(metadata)

    logger.info(f"STEP 1 DONE: {generated} synthetic audio files saved to {SYNTH_DIR}")
    logger.info(f"Metadata: {csv_path}")
    return metadata

# ══════════════════════════════════════════════════════════════
# STEP 2: Data augmentation (speed perturbation)
# ══════════════════════════════════════════════════════════════

def augment_primock57():
    """Create 0.9x and 1.1x speed versions of PriMock57 training data."""
    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

    with open(PRIMOCK_DIR / "train.json") as f:
        train_records = json.load(f)

    logger.info(f"STEP 2: Augmenting {len(train_records)} PriMock57 training samples (0.9x + 1.1x speed)")

    augmented_records = []
    for i, rec in enumerate(train_records):
        audio_path = rec["audio_path"]
        if not Path(audio_path).exists():
            logger.warning(f"Audio not found: {audio_path}, skipping")
            continue

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        for speed, label in [(0.9, "slow"), (1.1, "fast")]:
            out_name = f"aug_{label}_{Path(audio_path).stem}.wav"
            out_path = AUGMENTED_DIR / out_name

            if not out_path.exists():
                # Speed change via resampling
                rate = 1.0 / speed
                stretched = librosa.effects.time_stretch(audio, rate=rate)
                sf.write(str(out_path), stretched, 16000)

            augmented_records.append({
                "audio_path": str(out_path.resolve()),
                "text": rec["text"],
                "consultation": rec["consultation"],
                "speaker": rec["speaker"],
                "source": f"primock57_aug_{label}",
            })

        if (i + 1) % 20 == 0:
            logger.info(f"  Augmented {i + 1}/{len(train_records)} files")

    # Save augmented records
    aug_json = AUGMENTED_DIR / "augmented.json"
    aug_json.write_text(json.dumps(augmented_records, indent=2))
    logger.info(f"STEP 2 DONE: {len(augmented_records)} augmented samples saved")
    return augmented_records

# ══════════════════════════════════════════════════════════════
# STEP 3: Combine datasets
# ══════════════════════════════════════════════════════════════

def combine_datasets(synth_metadata: list[dict], augmented_records: list[dict]):
    """Merge original PriMock57 train + augmented + synthetic into one training set."""
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    # Original PriMock57 training data
    with open(PRIMOCK_DIR / "train.json") as f:
        original_train = json.load(f)
    for r in original_train:
        r["source"] = "primock57_original"

    # Original PriMock57 validation data
    with open(PRIMOCK_DIR / "val.json") as f:
        original_val = json.load(f)
    for r in original_val:
        r["source"] = "primock57_original"

    # Synthetic data
    synth_records = []
    for m in synth_metadata:
        wav_path = SYNTH_DIR / m["filename"]
        if wav_path.exists():
            synth_records.append({
                "audio_path": str(wav_path.resolve()),
                "text": m["text"],
                "consultation": f"synthetic_{m['category']}",
                "speaker": m["gender"],
                "source": "synthetic_british",
            })

    # Combine all for training
    all_train = original_train + augmented_records + synth_records

    # Use original val set + some synthetic for validation
    random.seed(42)
    random.shuffle(synth_records)
    synth_val = synth_records[:30]  # 30 synthetic for val
    synth_train = synth_records[30:]

    # Rebuild: train = original + augmented + most synthetic
    all_train = original_train + augmented_records + synth_train
    all_val = original_val + synth_val

    # Original test set stays untouched
    with open(PRIMOCK_DIR / "test.json") as f:
        test_data = json.load(f)

    # Save
    (COMBINED_DIR / "train.json").write_text(json.dumps(all_train, indent=2))
    (COMBINED_DIR / "val.json").write_text(json.dumps(all_val, indent=2))
    (COMBINED_DIR / "test.json").write_text(json.dumps(test_data, indent=2))

    # Duration estimate
    total_dur = 0
    for r in all_train:
        try:
            info = sf.info(r["audio_path"])
            total_dur += info.duration
        except Exception:
            pass

    logger.info(f"STEP 3 DONE: Combined dataset")
    logger.info(f"  Train: {len(all_train)} samples (~{total_dur/3600:.1f}h)")
    logger.info(f"  Val: {len(all_val)} samples")
    logger.info(f"  Test: {len(test_data)} samples (original PriMock57, untouched)")

    return all_train, all_val, test_data

# ══════════════════════════════════════════════════════════════
# STEP 4: Retrain Whisper v3
# ══════════════════════════════════════════════════════════════

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: bool = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_feats, padding=self.padding, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch

def _normalise(text: str) -> str:
    text = text.lower()
    text = text.replace("</s>", "")
    text = re.sub(r"[^ a-z0-9']", " ", text)
    return " ".join(text.split())

def prepare_whisper_dataset(split_path: Path, processor, max_audio_secs: float = 20.0) -> Dataset:
    with open(split_path) as f:
        records = json.load(f)

    # Filter to files that exist
    records = [r for r in records if Path(r["audio_path"]).exists()]
    logger.info(f"  Loading {split_path.name}: {len(records)} valid samples")

    ds = Dataset.from_dict({
        "audio": [r["audio_path"] for r in records],
        "text": [r["text"] for r in records],
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    def process_fn(batch):
        audio = batch["audio"]
        max_samples = int(max_audio_secs * 16_000)
        samples = np.array(audio["array"][:max_samples], dtype=np.float32)

        inputs = processor.feature_extractor(
            samples, sampling_rate=16_000, return_tensors="np"
        )
        labels = processor.tokenizer(batch["text"]).input_ids
        labels = labels[:448]

        return {
            "input_features": inputs.input_features[0],
            "labels": labels,
        }

    ds = ds.map(process_fn, remove_columns=ds.column_names)
    return ds

def make_whisper_compute_metrics(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_norm = [_normalise(p) for p in pred_str]
        label_norm = [_normalise(l) for l in label_str]

        pairs = [(r, h) for r, h in zip(label_norm, pred_norm) if r.strip()]
        if not pairs:
            return {"wer": 1.0}
        refs, hyps = zip(*pairs)
        return {"wer": jiwer.wer(list(refs), list(hyps))}

    return compute_metrics

def train_whisper_v3():
    """Retrain Whisper Small with expanded data, v3."""
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    device = get_device()
    logger.info(f"STEP 4: Training Whisper v3 on expanded data (device={device})")

    MODEL_ID = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )
    model.generation_config.suppress_tokens = []

    # Freeze encoder
    model.freeze_encoder()
    logger.info("Encoder frozen")

    # LoRA on decoder
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.2,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare datasets
    train_ds = prepare_whisper_dataset(COMBINED_DIR / "train.json", processor)
    val_ds = prepare_whisper_dataset(COMBINED_DIR / "val.json", processor)
    logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=WHISPER_V3_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=False,
        bf16=False,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=make_whisper_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        processing_class=processor.feature_extractor,
    )

    logger.info("=" * 60)
    logger.info("Starting Whisper v3 LoRA fine-tuning (expanded data)")
    logger.info("=" * 60)
    train_result = trainer.train()

    trainer.save_model(WHISPER_V3_DIR)
    processor.save_pretrained(WHISPER_V3_DIR)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info(f"STEP 4 DONE: Whisper v3 saved to {WHISPER_V3_DIR}")
    logger.info(f"  Final eval WER: {eval_metrics.get('eval_wer', 'N/A')}")

# ══════════════════════════════════════════════════════════════
# STEP 5: Retrain MedASR v2
# ══════════════════════════════════════════════════════════════

def prepare_medasr_dataset(split_path: Path, processor, max_audio_secs: float = 30.0) -> Dataset:
    with open(split_path) as f:
        records = json.load(f)

    records = [r for r in records if Path(r["audio_path"]).exists()]
    logger.info(f"  Loading {split_path.name}: {len(records)} valid samples")

    ds = Dataset.from_dict({
        "audio": [r["audio_path"] for r in records],
        "text": [r["text"] for r in records],
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    def process_fn(batch):
        audio = batch["audio"]
        max_samples = int(max_audio_secs * 16_000)
        samples = np.array(audio["array"][:max_samples], dtype=np.float32)

        inputs = processor(samples, sampling_rate=16_000, return_tensors="np")
        labels = processor.tokenizer(batch["text"]).input_ids

        return {
            "input_features": inputs.input_features[0],
            "labels": labels,
        }

    ds = ds.map(process_fn, remove_columns=ds.column_names)
    return ds

def make_medasr_compute_metrics(processor):
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred_strs = []
        for seq in pred_ids:
            collapsed = []
            prev = -1
            for idx in seq:
                if idx != prev and idx != 0:
                    collapsed.append(int(idx))
                prev = idx
            pred_strs.append(processor.tokenizer.decode(collapsed))

        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_strs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_norm = [_normalise(p) for p in pred_strs]
        label_norm = [_normalise(l) for l in label_strs]

        pairs = [(r, h) for r, h in zip(label_norm, pred_norm) if r.strip() and h.strip()]
        if not pairs:
            return {"wer": 1.0}
        refs, hyps = zip(*pairs)
        return {"wer": jiwer.wer(list(refs), list(hyps))}

    return compute_metrics

def freeze_encoder_layers(model, freeze_ratio: float = 0.7):
    """Freeze the bottom fraction of encoder parameters."""
    all_params = list(model.named_parameters())
    encoder_params = [(n, p) for n, p in all_params if "encoder" in n]
    n_freeze = int(len(encoder_params) * freeze_ratio)
    for i, (name, param) in enumerate(encoder_params):
        if i < n_freeze:
            param.requires_grad = False
    frozen = sum(1 for _, p in all_params if not p.requires_grad)
    total = len(all_params)
    logger.info(f"Froze {frozen}/{total} parameters ({frozen/total:.0%})")

def train_medasr_v2():
    """Retrain MedASR with expanded data, top 30% unfrozen."""
    from transformers import (
        AutoModelForCTC,
        AutoProcessor,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    device = get_device()
    logger.info(f"STEP 5: Training MedASR v2 on expanded data (device={device})")

    MODEL_ID = "google/medasr"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(MODEL_ID)

    # Freeze bottom 70% (train top 30%)
    freeze_encoder_layers(model, freeze_ratio=0.7)

    train_ds = prepare_medasr_dataset(COMBINED_DIR / "train.json", processor)
    val_ds = prepare_medasr_dataset(COMBINED_DIR / "val.json", processor)
    logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    training_args = TrainingArguments(
        output_dir=MEDASR_V2_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        fp16=device == "cuda",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=make_medasr_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    logger.info("=" * 60)
    logger.info("Starting MedASR v2 fine-tuning (expanded data, top 30% unfrozen)")
    logger.info("=" * 60)
    train_result = trainer.train()

    trainer.save_model(MEDASR_V2_DIR)
    processor.save_pretrained(MEDASR_V2_DIR)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info(f"STEP 5 DONE: MedASR v2 saved to {MEDASR_V2_DIR}")
    logger.info(f"  Final eval WER: {eval_metrics.get('eval_wer', 'N/A')}")

# ══════════════════════════════════════════════════════════════
# STEP 6: Benchmark all models
# ══════════════════════════════════════════════════════════════

def benchmark_all():
    """Benchmark all models on PriMock57 test set."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console(width=120)
    logger.info("STEP 6: Benchmarking all models on PriMock57 test set")

    with open(PRIMOCK_DIR / "test.json") as f:
        test_records = json.load(f)

    console.print(f"\nTest samples: {len(test_records)}")

    # Define models to test
    model_configs = []

    # Base Whisper Small
    model_configs.append(("Base Whisper Small", "whisper", "openai/whisper-small", {}))

    # Whisper v2 (if exists)
    v2_path = Path("data/models/whisper-small-british-medical-v2")
    if v2_path.exists() and (v2_path / "adapter_config.json").exists():
        model_configs.append(("Whisper v2 (LoRA)", "whisper_lora", str(v2_path), {
            "no_repeat_ngram_size": 3, "repetition_penalty": 1.3, "max_new_tokens": 225,
        }))

    # Whisper v3 (if exists)
    v3_path = Path(WHISPER_V3_DIR)
    if v3_path.exists() and (v3_path / "adapter_config.json").exists():
        model_configs.append(("Whisper v3 (expanded)", "whisper_lora", str(v3_path), {
            "no_repeat_ngram_size": 3, "repetition_penalty": 1.3, "max_new_tokens": 225,
        }))

    # Base MedASR
    model_configs.append(("Base MedASR", "medasr", "google/medasr", {}))

    # MedASR v1 fine-tune (if exists)
    v1_medasr = Path("data/models/medasr-british-medical")
    if v1_medasr.exists() and (v1_medasr / "model.safetensors").exists():
        model_configs.append(("MedASR v1 (fine-tuned)", "medasr", str(v1_medasr), {}))

    # MedASR v2 (if exists)
    v2_medasr = Path(MEDASR_V2_DIR)
    if v2_medasr.exists():
        model_configs.append(("MedASR v2 (expanded)", "medasr", str(v2_medasr), {}))

    # Load and run each model
    all_results = {}

    for name, mtype, path, gen_kwargs in model_configs:
        console.print(f"\nLoading [bold]{name}[/bold]...")
        try:
            if mtype == "whisper":
                from transformers import WhisperForConditionalGeneration, WhisperProcessor
                processor = WhisperProcessor.from_pretrained(path)
                model = WhisperForConditionalGeneration.from_pretrained(path)
                model.eval()
            elif mtype == "whisper_lora":
                from transformers import WhisperForConditionalGeneration, WhisperProcessor
                from peft import PeftModel
                with open(Path(path) / "adapter_config.json") as f:
                    ac = json.load(f)
                base_id = ac.get("base_model_name_or_path", "openai/whisper-small")
                processor = WhisperProcessor.from_pretrained(base_id)
                base_model = WhisperForConditionalGeneration.from_pretrained(base_id)
                model = PeftModel.from_pretrained(base_model, path)
                model = model.merge_and_unload()
                model.eval()
            elif mtype == "medasr":
                from transformers import AutoModelForCTC, AutoProcessor
                processor = AutoProcessor.from_pretrained(path)
                model = AutoModelForCTC.from_pretrained(path)
                model.eval()

            console.print(f"  [green]Loaded[/green]")
        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")
            continue

        results = []
        for i, rec in enumerate(test_records):
            audio_path = rec["audio_path"]
            full_ref = rec["text"]
            full_dur = librosa.get_duration(path=audio_path)
            segment_dur = min(30.0, full_dur)

            audio, sr = librosa.load(audio_path, sr=16000, duration=30.0)

            words_per_sec = max(len(full_ref.split()) / full_dur, 1)
            approx_words = int(words_per_sec * segment_dur)
            ref_text = " ".join(full_ref.split()[:approx_words])
            ref_norm = _normalise(ref_text)

            try:
                t0 = time.perf_counter()
                if "medasr" in mtype:
                    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    pred_ids = torch.argmax(logits, dim=-1)[0]
                    collapsed = []
                    prev = -1
                    for idx in pred_ids.tolist():
                        if idx != prev and idx != 0:
                            collapsed.append(idx)
                        prev = idx
                    hyp_text = processor.tokenizer.decode(collapsed)
                else:
                    inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
                    with torch.no_grad():
                        gen_ids = model.generate(
                            **inputs, language="en", task="transcribe", **gen_kwargs,
                        )
                    hyp_text = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

                elapsed = time.perf_counter() - t0
                hyp_norm = _normalise(hyp_text.strip())
                wer = jiwer.wer(ref_norm, hyp_norm) if ref_norm and hyp_norm else 1.0
                cer = jiwer.cer(ref_norm, hyp_norm) if ref_norm and hyp_norm else 1.0
            except Exception as e:
                logger.warning(f"{name} failed on sample {i}: {e}")
                hyp_text, elapsed, wer, cer = "", 0.0, 1.0, 1.0

            results.append({"file": Path(audio_path).name, "wer": wer, "cer": cer, "time": elapsed})

        all_results[name] = results

        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_cer = sum(r["cer"] for r in results) / len(results)
        console.print(f"  Avg WER: {avg_wer:.1%}, Avg CER: {avg_cer:.1%}")

        # Free memory
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    console.print("\n")
    table = Table(title="BENCHMARK: All Models on PriMock57 Test Set", show_header=True, show_lines=True)
    table.add_column("Model", style="bold", min_width=25)
    table.add_column("Avg WER", justify="right")
    table.add_column("Avg CER", justify="right")
    table.add_column("Avg Time", justify="right")

    for name, results in all_results.items():
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_cer = sum(r["cer"] for r in results) / len(results)
        avg_time = sum(r["time"] for r in results) / len(results)
        table.add_row(name, f"{avg_wer:.1%}", f"{avg_cer:.1%}", f"{avg_time:.2f}s")

    console.print(table)

    # Save results
    out_path = Path("data/benchmark_v3.json")
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    console.print(f"\nResults saved to {out_path}")

    logger.info("STEP 6 DONE: Benchmark complete")

# ══════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════

async def run_pipeline():
    t_start = time.time()

    # Step 1: Synthetic audio generation
    t0 = time.time()
    synth_metadata = await generate_synthetic_audio()
    logger.info(f"Step 1 took {time.time()-t0:.0f}s")

    # Step 2: Data augmentation
    t0 = time.time()
    augmented_records = augment_primock57()
    logger.info(f"Step 2 took {time.time()-t0:.0f}s")

    # Step 3: Combine datasets
    t0 = time.time()
    combine_datasets(synth_metadata, augmented_records)
    logger.info(f"Step 3 took {time.time()-t0:.0f}s")

    # Step 4: Retrain Whisper v3
    t0 = time.time()
    try:
        train_whisper_v3()
        logger.info(f"Step 4 (Whisper v3) took {time.time()-t0:.0f}s")
    except Exception as e:
        logger.error(f"Step 4 failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 5: Retrain MedASR v2
    t0 = time.time()
    try:
        train_medasr_v2()
        logger.info(f"Step 5 (MedASR v2) took {time.time()-t0:.0f}s")
    except Exception as e:
        logger.error(f"Step 5 failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 6: Benchmark
    t0 = time.time()
    try:
        benchmark_all()
        logger.info(f"Step 6 (benchmark) took {time.time()-t0:.0f}s")
    except Exception as e:
        logger.error(f"Step 6 failed: {e}")
        import traceback
        traceback.print_exc()

    total = time.time() - t_start
    logger.info(f"\nTotal pipeline time: {total/60:.1f} minutes")
    logger.info("Pipeline complete!")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
