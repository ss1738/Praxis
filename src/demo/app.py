"""Streamlit demo for PRAXIS - AI Clinical Documentation System."""

import json
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when run via streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="PRAXIS - AI Clinical Documentation",
    layout="wide",
)

BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark_final_v2.json"
TRAINING_LOG_PATH = PROJECT_ROOT / "data" / "training_v3_full.log"

# sidebar
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Patient Management",
        "Live Demo",
        "SOAP Generator",
        "Model Comparison",
        "Clinical Evaluation",
        "GP Review",
        "Patient History",
        "Session History",
        "STT Benchmarks",
        "System Architecture",
        "Training Logs",
    ],
)


# --- Helpers ---
def get_soap_generator(model_value=None):
    from src.soap_generation.generator import SOAPGenerator, SOAPModel

    model_value = model_value or st.session_state.get("soap_model", "praxis-soap-v1")
    try:
        model = SOAPModel(model_value)
    except ValueError:
        model = SOAPModel.PRAXIS_PRIMARY
    return SOAPGenerator(model=model)


def get_pipeline(model_value=None):
    from src.soap_generation.generator import SOAPModel
    from src.soap_generation.pipeline import ClinicalScribePipeline

    model_value = model_value or st.session_state.get("soap_model", "praxis-soap-v1")
    try:
        model = SOAPModel(model_value)
    except ValueError:
        model = SOAPModel.PRAXIS_PRIMARY
    return ClinicalScribePipeline(soap_model=model)


def model_display(model_id: str) -> str:
    from src.soap_generation.generator import display_name
    return display_name(model_id)


SAMPLE_TRANSCRIPTS = {
    "Case 1 - Lower Back Pain": (
        "Patient: Good morning doctor. I've been having this pain in my lower back "
        "for about two weeks now. It's mostly there all the time but gets worse when "
        "I sit for too long.\n"
        "Doctor: Any numbness or tingling in your legs?\n"
        "Patient: No, nothing like that.\n"
        "Doctor: Let me examine your back. There's some tenderness over L4-L5 but no "
        "neurological deficit. I think this is mechanical back pain. I'll prescribe "
        "naproxen 500mg twice daily and refer you for physiotherapy. If you develop "
        "any leg weakness or bladder problems, come back immediately."
    ),
    "Case 2 - Persistent Cough (LRTI)": (
        "Patient: Good morning doctor. I've been having this persistent cough for "
        "about three weeks now. It started as a dry cough but now I'm bringing up "
        "some yellowish phlegm. I've also been feeling quite tired and had a mild "
        "temperature last week. I take ramipril 5mg for blood pressure and "
        "atorvastatin 20mg for cholesterol. No allergies that I know of.\n"
        "Doctor: Any shortness of breath or chest pain?\n"
        "Patient: A little breathless when climbing stairs but no chest pain.\n"
        "Doctor: Let me listen to your chest. I can hear some crackles at the right base. "
        "Your temperature is 37.8. I think this is a lower respiratory tract infection. "
        "I'll prescribe amoxicillin 500mg three times daily for five days. "
        "If your symptoms get worse, you develop chest pain, or you cough up blood, "
        "please come back immediately or call 111."
    ),
    "Case 3 - Diabetes Review": (
        "Patient: Hello doctor, I've come about my diabetes check. My blood sugar "
        "readings have been all over the place. Fasting is around 9 to 11 and "
        "after meals it goes up to 15 sometimes. I'm on metformin 500mg twice daily "
        "and gliclazide 80mg once daily. I've also noticed tingling in my feet, "
        "especially at night. And my vision seems a bit blurry sometimes.\n"
        "Doctor: When was your last HbA1c?\n"
        "Patient: About three months ago, it was 72.\n"
        "Doctor: That's above target. Let me check your feet. I can see reduced "
        "sensation to monofilament in both feet. I'd like to increase your metformin "
        "to 1000mg twice daily and refer you for diabetic eye screening. We'll repeat "
        "your HbA1c in 3 months. Your target is below 53."
    ),
    "Case 4 - Mental Health (Depression)": (
        "Patient: I've been feeling really low for the past month. I can't sleep properly, "
        "I've lost my appetite, and I just don't enjoy anything anymore. I've been calling "
        "in sick to work quite a lot.\n"
        "Doctor: I'm sorry to hear that. Have you had any thoughts of harming yourself?\n"
        "Patient: No, nothing like that. I just feel flat all the time.\n"
        "Doctor: How's your concentration? Are you able to manage day-to-day tasks?\n"
        "Patient: Not really. I keep forgetting things and I can't focus at work.\n"
        "Doctor: Based on what you've told me, I think you may be experiencing moderate "
        "depression. I'd like to start you on sertraline 50mg once daily and refer you "
        "for CBT through the IAPT service. We'll review in 2 weeks. If you do start "
        "having any thoughts of self-harm, please call 111 or go to A&E immediately."
    ),
}


# ── Helper: Render pipeline results ──────────────────────────────
def render_pipeline_results(pipe_result):
    """Display diarisation, entities, and SOAP note from a PipelineResult."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Speaker Diarisation")
        if pipe_result.diarisation and pipe_result.diarisation.segments:
            diar = pipe_result.diarisation
            st.caption(
                f"Method: **{'Pyannote' if diar.speaker_map else 'Rule-based'}** | "
                f"Speakers: **{', '.join(diar.speakers)}** | "
                f"Segments: **{len(diar.segments)}**"
            )
            st.code(diar.diarised_transcript or pipe_result.diarised_transcript, language=None)
        else:
            st.code(pipe_result.diarised_transcript, language=None)

        if pipe_result.entities:
            st.subheader("Extracted Medical Entities")

            method = "Regex (fallback)"
            if pipe_result.extraction_result:
                m = pipe_result.extraction_result.method
                if "medcat" in m:
                    method = "MedCAT + SNOMED CT"
                    if "medspacy" in m:
                        method += " + medspaCy negation"
            st.caption(f"Extraction method: **{method}**")

            entity_rows = []
            for e in pipe_result.entities:
                row = {
                    "Entity": e.text,
                    "Type": e.entity_type.value.replace("_", " ").title(),
                    "Confidence": e.confidence.value.upper(),
                    "Negated": "Yes" if e.negated else "",
                }
                if e.snomed_code:
                    row["SNOMED CT"] = f"{e.snomed_code} ({e.snomed_term})"
                else:
                    row["SNOMED CT"] = ""
                entity_rows.append(row)

            entity_df = pd.DataFrame(entity_rows)
            if not entity_df.empty:
                st.dataframe(entity_df, use_container_width=True, hide_index=True)

                n_symptoms = sum(1 for e in pipe_result.entities if e.entity_type.value == "symptom")
                n_meds = sum(1 for e in pipe_result.entities if e.entity_type.value == "medication")
                n_neg = sum(1 for e in pipe_result.entities if e.negated)
                n_snomed = sum(1 for e in pipe_result.entities if e.snomed_code)
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Symptoms", n_symptoms)
                mcol2.metric("Medications", n_meds)
                mcol3.metric("Negated", n_neg)
                mcol4.metric("SNOMED Linked", n_snomed)

    with col2:
        st.subheader("Generated SOAP Note")
        if pipe_result.soap_result:
            st.markdown(pipe_result.soap_result.soap_note)
            st.caption(
                f"Model: **{model_display(pipe_result.soap_result.model)}** | "
                f"Template: **{pipe_result.soap_result.template.upper()}** | "
                f"Latency: {pipe_result.soap_result.generation_time_secs}s | "
                f"Tokens: {pipe_result.soap_result.token_count}"
            )

    st.success(f"Full pipeline completed in {pipe_result.total_time_secs}s")


# =====================================================================
# PAGE 1: Home
# =====================================================================
if page == "Home":
    st.title("PRAXIS")
    st.markdown(
        "**P**ipeline for **R**eal-time **A**nalysis and e**X**traction "
        "of cl**I**nical documentation from **S**peech"
    )
    st.markdown("---")

    st.markdown("**Satyawan Singh** - MSc Artificial Intelligence for Business Intelligence")
    st.markdown("University of Leicester | MSc dissertation project")

    st.markdown("---")

    st.markdown(
        """
        ### Research Objective

        This dissertation investigates whether fine-tuned large language models can
        reliably generate structured clinical SOAP notes from British English GP
        consultation transcripts, with the goal of reducing GP administrative burden
        while maintaining clinical accuracy.

        ### Approach

        PRAXIS is an end-to-end pipeline that:

        1. **Transcribes** doctor-patient audio using fine-tuned speech-to-text models
           (Whisper, MedASR) evaluated on the PriMock57 British GP dataset
        2. **Identifies speakers** via Pyannote neural diarisation (with rule-based fallback)
        3. **Extracts medical entities** using MedCAT + medspaCy (SNOMED CT linking, negation detection)
        4. **Generates SOAP notes** using QLoRA fine-tuned medical LLMs (Med42-8B,
           Meditron3-8B) trained on a combined dataset of PriMock57, MTSamples,
           ChatDoctor, and synthetic GP consultations

        ### Key Contributions

        - First published evaluation of Google MedASR on British English GP audio
          (93.2% WER - effectively unusable due to accent mismatch)
        - Identification and mitigation of catastrophic hallucination in LoRA-tuned
          Whisper (514% WER reduced to 57.4% via encoder freezing)
        - QLoRA fine-tuning of Med42-8B and Meditron3-8B for UK GP SOAP note
          generation with SNOMED CT coding
        - Fully offline-capable system using quantised GGUF models via Ollama
        """
    )

    st.markdown("---")
    st.subheader("System Status")

    col1, col2, col3 = st.columns(3)

    from src.soap_generation.generator import SOAPGenerator
    gen = SOAPGenerator()
    ollama_ok = gen.check_ollama()
    models = gen.list_models() if ollama_ok else []

    with col1:
        if ollama_ok:
            st.success(f"Inference Server: Online ({len(models)} models)")
        else:
            st.error("Inference Server: Offline")
            st.caption("Start with: `ollama serve`")

    with col2:
        has_primary = any("praxis" in m for m in models)
        if has_primary:
            st.success("PRAXIS-SOAP-v1: Ready")
        else:
            st.warning("PRAXIS-SOAP-v1: Not loaded")

    with col3:
        display_models = [model_display(m.replace(":latest", "")) for m in models]
        st.info(f"Available: {len(models)} models")
        for dm in display_models:
            st.caption(f"- {dm}")


# =====================================================================
# PAGE: Patient Management
# =====================================================================
elif page == "Patient Management":
    st.title("Patient Management")
    st.markdown(
        "Register new patients and search existing records. "
        "Each consultation is linked to a patient so you can track their full visit history."
    )

    from src.clinical.patient_store import PatientStore
    from src.clinical.ontology import Patient

    patient_store = PatientStore(store_dir=PROJECT_ROOT / "data" / "patients")

    tab_register, tab_search, tab_all = st.tabs(["Register New Patient", "Search Patient", "All Patients"])

    with tab_register:
        st.subheader("Register a New Patient")
        with st.form("register_patient_form"):
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input("First Name *")
                last_name = st.text_input("Last Name *")
            with col2:
                nhs_number = st.text_input("NHS Number", placeholder="e.g. 943 476 5919")
                dob = st.date_input("Date of Birth", value=None)

            submitted = st.form_submit_button("Register Patient", type="primary")
            if submitted:
                if not first_name or not last_name:
                    st.error("First name and last name are required.")
                else:
                    # Check for duplicate NHS number
                    if nhs_number and patient_store.find_by_nhs_number(nhs_number):
                        st.error(f"A patient with NHS number {nhs_number} already exists.")
                    else:
                        patient = Patient(
                            first_name=first_name.strip(),
                            last_name=last_name.strip(),
                            nhs_number=nhs_number.strip(),
                            date_of_birth=str(dob) if dob else "",
                        )
                        patient_store.save(patient)
                        st.success(f"Patient registered: **{patient.full_name}** (ID: {patient.patient_id[:8]}...)")

    with tab_search:
        st.subheader("Find a Patient")
        search_type = st.radio("Search by:", ["Name", "NHS Number"], horizontal=True, key="patient_search_type")

        if search_type == "Name":
            query = st.text_input("Enter patient name:", key="patient_name_search")
            if query:
                results = patient_store.search_by_name(query)
                if results:
                    st.markdown(f"**{len(results)} patient(s) found:**")
                    for p in results:
                        st.markdown(
                            f"- **{p.full_name}** | NHS: {p.nhs_number or 'N/A'} | "
                            f"DOB: {p.date_of_birth or 'N/A'} | ID: `{p.patient_id[:8]}...`"
                        )
                else:
                    st.info("No patients found.")
        else:
            nhs_query = st.text_input("Enter NHS number:", key="patient_nhs_search")
            if nhs_query:
                found = patient_store.find_by_nhs_number(nhs_query)
                if found:
                    st.success(f"Found: **{found.full_name}** | DOB: {found.date_of_birth or 'N/A'}")
                else:
                    st.info("No patient found with that NHS number.")

    with tab_all:
        st.subheader("All Registered Patients")
        all_patients = patient_store.list_patients()
        if not all_patients:
            st.info("No patients registered yet. Use the 'Register New Patient' tab to add one.")
        else:
            st.markdown(f"**{len(all_patients)} patients registered**")
            patient_rows = []
            for p in all_patients:
                patient_rows.append({
                    "Name": p.full_name,
                    "NHS Number": p.nhs_number or "N/A",
                    "Date of Birth": p.date_of_birth or "N/A",
                    "Registered": p.registered_at[:10],
                    "ID": p.patient_id[:8],
                })
            st.dataframe(pd.DataFrame(patient_rows), use_container_width=True, hide_index=True)


# =====================================================================
# PAGE 2: Live Demo (full pipeline)
# =====================================================================
elif page == "Live Demo":
    st.title("Live Pipeline Demonstration")
    st.markdown(
        "Run the full PRAXIS pipeline: "
        "**Record/Upload Audio** → **Transcription** → **Diarisation** → **Entity Extraction** → **SOAP Generation**"
    )

    # --- Patient Selection ---
    st.markdown("---")
    st.subheader("Select Patient")
    from src.clinical.patient_store import PatientStore
    _patient_store = PatientStore(store_dir=PROJECT_ROOT / "data" / "patients")
    _all_patients = _patient_store.list_patients()

    if not _all_patients:
        st.warning("No patients registered. Go to **Patient Management** to register a patient first.")
        st.stop()

    _patient_options = {f"{p.full_name} (NHS: {p.nhs_number or 'N/A'})": p for p in _all_patients}
    _selected_patient_label = st.selectbox("Patient:", list(_patient_options.keys()), key="live_demo_patient")
    _selected_patient = _patient_options[_selected_patient_label]
    st.caption(f"Patient ID: {_selected_patient.patient_id[:8]}... | DOB: {_selected_patient.date_of_birth or 'N/A'}")

    # Store in session state for use by pipeline
    st.session_state["current_patient_id"] = _selected_patient.patient_id
    st.session_state["current_patient_name"] = _selected_patient.full_name

    st.markdown("---")

    tab_mic, tab_audio, tab_text = st.tabs(["Record Live", "Upload Audio", "Transcript Input"])

    # --- TAB: Record Live ---
    with tab_mic:
        st.markdown(
            "Record a live consultation using your microphone. "
            "Click the button below to start recording - click again to stop. "
            "The audio will be transcribed by Whisper and processed through the full pipeline."
        )

        from audio_recorder_streamlit import audio_recorder

        audio_bytes = audio_recorder(
            text="Click to record consultation",
            recording_color="#e74c3c",
            neutral_color="#2ecc71",
            icon_size="2x",
            pause_threshold=120.0,   # allow long recordings (2 min)
            sample_rate=16000,
        )

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            st.success("Recording captured. Click below to process.")

            if st.button("Transcribe & Generate SOAP Note", key="mic_soap", type="primary"):
                import tempfile
                import whisper

                with st.spinner("Step 1/4: Transcribing audio with Whisper..."):
                    # Save recorded bytes to a temp WAV file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name

                    # Transcribe with Whisper
                    model = whisper.load_model("small")
                    result = model.transcribe(
                        tmp_path,
                        language="en",
                        fp16=False,
                    )
                    transcript = result["text"].strip()
                    Path(tmp_path).unlink(missing_ok=True)

                st.subheader("Transcript (Whisper Small)")
                st.code(transcript, language=None)

                if not transcript:
                    st.warning("No speech detected. Try recording again, speaking clearly.")
                else:
                    with st.spinner("Step 2-4: Diarisation → Entities → SOAP Generation..."):
                        try:
                            pipeline = get_pipeline()
                            pipe_result = pipeline.run_from_transcript(transcript)
                            render_pipeline_results(pipe_result)

                            # Save session linked to patient
                            if pipe_result.session:
                                pipe_result.session.patient_id = st.session_state.get("current_patient_id", "")
                                pipe_result.session.patient_name = st.session_state.get("current_patient_name", "")
                                from src.clinical.session_store import SessionStore
                                _store = SessionStore(store_dir=PROJECT_ROOT / "data" / "sessions")
                                _store.save(pipe_result.session)
                                st.caption(f"Session saved for patient: **{pipe_result.session.patient_name}**")
                        except Exception as e:
                            st.error(f"Pipeline error: {e}")
                            st.caption("Ensure the inference server is running: `ollama serve`")

    # --- TAB: Upload Audio ---
    with tab_audio:
        uploaded = st.file_uploader("Upload consultation audio", type=["wav", "mp3", "m4a", "ogg", "flac"])
        if uploaded is not None:
            st.audio(uploaded)

            if st.button("Transcribe & Generate SOAP Note", key="upload_soap", type="primary"):
                import tempfile
                import whisper

                with st.spinner("Step 1/4: Transcribing audio with Whisper..."):
                    suffix = Path(uploaded.name).suffix if uploaded.name else ".wav"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name

                    model = whisper.load_model("small")
                    result = model.transcribe(
                        tmp_path,
                        language="en",
                        fp16=False,
                    )
                    transcript = result["text"].strip()
                    Path(tmp_path).unlink(missing_ok=True)

                st.subheader("Transcript (Whisper Small)")
                st.code(transcript, language=None)

                if not transcript:
                    st.warning("No speech detected in the uploaded file.")
                else:
                    with st.spinner("Step 2-4: Diarisation → Entities → SOAP Generation..."):
                        try:
                            pipeline = get_pipeline()
                            pipe_result = pipeline.run_from_transcript(transcript)
                            render_pipeline_results(pipe_result)

                            # Save session linked to patient
                            if pipe_result.session:
                                pipe_result.session.patient_id = st.session_state.get("current_patient_id", "")
                                pipe_result.session.patient_name = st.session_state.get("current_patient_name", "")
                                from src.clinical.session_store import SessionStore
                                _store = SessionStore(store_dir=PROJECT_ROOT / "data" / "sessions")
                                _store.save(pipe_result.session)
                                st.caption(f"Session saved for patient: **{pipe_result.session.patient_name}**")
                        except Exception as e:
                            st.error(f"Pipeline error: {e}")
                            st.caption("Ensure the inference server is running: `ollama serve`")

    with tab_text:
        from src.soap_generation.templates import NoteTemplate, TEMPLATE_DESCRIPTIONS
        from src.soap_generation.generator import template_display_name

        sample = st.selectbox(
            "Select a clinical scenario:",
            ["Custom Input"] + list(SAMPLE_TRANSCRIPTS.keys()),
        )

        demo_template_options = {template_display_name(t.value): t for t in NoteTemplate}
        demo_template_display = st.selectbox("Note template:", list(demo_template_options.keys()), key="demo_template")
        demo_template = demo_template_options[demo_template_display]
        st.caption(TEMPLATE_DESCRIPTIONS[demo_template.value])

        if sample != "Custom Input":
            transcript_input = st.text_area(
                "Consultation transcript:",
                value=SAMPLE_TRANSCRIPTS[sample],
                height=200,
            )
        else:
            transcript_input = st.text_area(
                "Paste consultation transcript:",
                height=200,
                placeholder="Patient: Good morning doctor...\nDoctor: How can I help you today?",
            )

        if transcript_input and st.button("Run Full Pipeline", key="text_pipeline", type="primary"):
            with st.spinner(f"Running: Diarisation → Entities → {demo_template_display}..."):
                try:
                    pipeline = get_pipeline()
                    result = pipeline.run_from_transcript(transcript_input, template=demo_template)
                    render_pipeline_results(result)

                    # Save session linked to patient
                    try:
                        from src.clinical.session_store import SessionStore
                        if result.session:
                            result.session.patient_id = st.session_state.get("current_patient_id", "")
                            result.session.patient_name = st.session_state.get("current_patient_name", "")
                            store = SessionStore(store_dir=PROJECT_ROOT / "data" / "sessions")
                            store.save(result.session)
                            st.caption(f"Session saved for patient: **{result.session.patient_name}** ({result.session.session_id[:8]}...)")
                    except Exception as e:
                        st.warning(f"Could not save session: {e}")

                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.caption("Ensure the inference server is running: `ollama serve`")


# =====================================================================
# PAGE 3: SOAP Generator (standalone)
# =====================================================================
elif page == "SOAP Generator":
    st.title("Clinical Note Generator")
    st.markdown("Generate structured clinical notes from consultation transcripts using fine-tuned clinical LLMs.")

    from src.soap_generation.generator import SOAPGenerator, SOAPModel, display_name, template_display_name
    from src.soap_generation.templates import NoteTemplate, TEMPLATE_DESCRIPTIONS

    gen = SOAPGenerator()
    available_models = gen.list_models()

    model_options = {}
    for m in SOAPModel:
        if any(m.value in am for am in available_models):
            model_options[display_name(m.value)] = m.value

    if not model_options:
        st.error("No models available. Ensure the inference server is running.")
        st.code("ollama serve", language="bash")
    else:
        col_model, col_template = st.columns(2)

        with col_model:
            selected_display = st.selectbox("Select model:", list(model_options.keys()))
            selected_model = model_options[selected_display]
            st.session_state["soap_model"] = selected_model

        with col_template:
            template_options = {template_display_name(t.value): t for t in NoteTemplate}
            selected_template_display = st.selectbox("Select note template:", list(template_options.keys()))
            selected_template = template_options[selected_template_display]

        # Show template description
        st.info(f"**{selected_template_display}:** {TEMPLATE_DESCRIPTIONS[selected_template.value]}")

        sample = st.selectbox(
            "Load clinical scenario:",
            ["Custom Input"] + list(SAMPLE_TRANSCRIPTS.keys()),
            key="soap_sample",
        )

        if sample != "Custom Input":
            text_input = st.text_area("Transcript:", value=SAMPLE_TRANSCRIPTS[sample], height=250)
        else:
            text_input = st.text_area(
                "Paste transcript:",
                height=250,
                placeholder="Patient: Good morning doctor...",
            )

        if text_input and st.button("Generate Clinical Note", type="primary"):
            with st.spinner(f"Generating {selected_template_display} with {selected_display}..."):
                try:
                    generator = SOAPGenerator(model=SOAPModel(selected_model), template=selected_template)
                    result = generator.generate(text_input)

                    st.markdown("---")
                    st.subheader(f"Generated {selected_template_display}")
                    st.markdown(result.soap_note)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Model", display_name(result.model))
                    col2.metric("Template", template_display_name(result.template))
                    col3.metric("Latency", f"{result.generation_time_secs}s")
                    col4.metric("Tokens", result.token_count)
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.caption("Check that the inference server is running: `ollama serve`")


# =====================================================================
# PAGE 4: Model Comparison
# =====================================================================
elif page == "Model Comparison":
    st.title("Model Comparison - SOAP Output Quality")
    st.markdown(
        "Compare SOAP note outputs across all available fine-tuned models "
        "on the same clinical scenario. This supports the evaluation chapter "
        "of the dissertation."
    )

    from src.soap_generation.generator import SOAPGenerator, display_name
    from src.soap_generation.templates import NoteTemplate, TEMPLATE_DISPLAY_NAMES, TEMPLATE_DESCRIPTIONS

    col_s, col_t = st.columns([1, 1])
    with col_s:
        sample = st.selectbox(
            "Select clinical scenario:",
            list(SAMPLE_TRANSCRIPTS.keys()),
            key="compare_sample",
        )
    with col_t:
        template_options = {TEMPLATE_DISPLAY_NAMES[t.value]: t for t in NoteTemplate}
        selected_template_name = st.selectbox(
            "Note template:",
            list(template_options.keys()),
            key="compare_template",
        )
        selected_template = template_options[selected_template_name]
        st.caption(TEMPLATE_DESCRIPTIONS[selected_template.value])

    transcript = SAMPLE_TRANSCRIPTS[sample]
    st.text_area("Input Transcript:", value=transcript, height=150, disabled=True)

    if st.button("Run Comparison Across All Models", type="primary"):
        gen = SOAPGenerator(template=selected_template)
        with st.spinner("Generating SOAP notes from all available models..."):
            results = gen.generate_comparison(transcript, template=selected_template)

        if not results:
            st.error("No models available. Ensure the inference server is running.")
        else:
            for result in results:
                with st.expander(
                    f"{display_name(result.model)} - {result.generation_time_secs}s, "
                    f"{result.token_count} tokens",
                    expanded=True,
                ):
                    st.markdown(result.soap_note)

            st.markdown("---")
            st.subheader("Quantitative Summary")

            from src.clinical.evaluator import STRUCTURE_PATTERNS
            template_key = selected_template.value
            section_keywords = list(STRUCTURE_PATTERNS.get(template_key, STRUCTURE_PATTERNS["soap"]).keys())

            summary_df = pd.DataFrame([
                {
                    "Model": display_name(r.model),
                    "Template": selected_template_name,
                    "Latency (s)": r.generation_time_secs,
                    "Tokens": r.token_count,
                    "Structure OK": all(
                        kw.upper() in r.soap_note.upper()
                        for kw in section_keywords
                    ),
                    "SNOMED CT Codes": "SNOMED" in r.soap_note.upper(),
                    "Safety-netting": "safety" in r.soap_note.lower(),
                }
                for r in results
            ])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Evaluation Criteria")
            st.markdown(
                f"""
                | Criterion | Description |
                |-----------|-------------|
                | **Structure OK** | Output contains all expected sections for **{selected_template_name}** |
                | **SNOMED CT Codes** | Clinical findings are mapped to SNOMED CT terminology |
                | **Safety-netting** | Plan includes appropriate safety-netting advice |
                | **Latency** | Wall-clock generation time (lower is better) |
                | **Token Count** | Output verbosity (model-dependent) |
                """
            )


# =====================================================================
# PAGE 5: STT Benchmarks
# =====================================================================
elif page == "STT Benchmarks":
    st.title("Speech-to-Text Benchmark Results")
    st.markdown(
        "Evaluated on 14 held-out test samples from the PriMock57 dataset "
        "(British English mock GP consultations). WER = Word Error Rate (lower is better)."
    )

    if BENCHMARK_PATH.exists():
        with open(BENCHMARK_PATH) as f:
            data = json.load(f)

        rows = []
        for model_name, samples in data.items():
            wers = [s["wer"] for s in samples]
            cers = [s["cer"] for s in samples]
            times = [s["time"] for s in samples]
            rows.append({
                "Model": model_name,
                "Avg WER (%)": round(sum(wers) / len(wers) * 100, 1),
                "Avg CER (%)": round(sum(cers) / len(cers) * 100, 1),
                "Avg Inference Time (s)": round(sum(times) / len(times), 2),
                "Test Samples": len(samples),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Word Error Rate Comparison")
        chart_df = df[["Model", "Avg WER (%)"]].set_index("Model")
        st.bar_chart(chart_df)

        st.markdown("---")
        st.subheader("Key Findings")
        st.markdown(
            """
            - **Google MedASR** achieves 93.2% WER on British GP audio - trained on
              US English radiology dictation, the accent mismatch renders it unusable
            - **Whisper Small (base)** performs best at 50.1% WER without any fine-tuning
            - **LoRA fine-tuning** on small data (90 samples) caused catastrophic
              hallucination (514% WER) due to seq2seq repetition loops
            - **Encoder freezing + generation constraints** mitigated hallucination
              (57.4% WER) but did not improve over the base model
            - **MedASR fine-tuning** (CTC architecture) showed 11.6% relative
              improvement - CTC models are safer for small-data fine-tuning
            """
        )
    else:
        st.warning(
            "Benchmark results file not found. Run `python scripts/final_benchmark.py` "
            "to generate results."
        )


# =====================================================================
# PAGE 6: System Architecture
# =====================================================================
elif page == "System Architecture":
    st.title("System Architecture")
    st.markdown("End-to-end pipeline from consultation audio to structured clinical documentation.")

    cols = st.columns(5)

    stages = [
        (
            "Audio Input",
            "done",
            "WAV/MP3/FLAC upload, 16 kHz resampling, spectral-gating noise reduction",
        ),
        (
            "Speech-to-Text",
            "done",
            "Whisper Small + Google MedASR, LoRA fine-tuned on PriMock57 (British GP)",
        ),
        (
            "Speaker Diarisation",
            "done",
            "Rule-based doctor/patient turn detection from lexical cues",
        ),
        (
            "Entity Extraction",
            "done",
            "Regex-based medical NER: symptoms, medications, conditions, negations",
        ),
        (
            "SOAP Generation",
            "done",
            "QLoRA fine-tuned Med42-8B / Meditron3-8B via Ollama (offline capable)",
        ),
    ]

    for col, (name, status, desc) in zip(cols, stages):
        with col:
            if status == "done":
                icon = "&#9989;"
                colour = "#d4edda"
            elif status == "in_progress":
                icon = "&#128992;"
                colour = "#fff3cd"
            else:
                icon = "&#9898;"
                colour = "#e2e3e5"

            st.markdown(
                f'<div style="background:{colour};padding:16px;border-radius:8px;'
                f'text-align:center;min-height:180px">'
                f'<p style="font-size:24px;margin:0">{icon}</p>'
                f'<p style="font-weight:bold;margin:8px 0 4px">{name}</p>'
                f'<p style="font-size:13px;color:#555">{desc}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.subheader("SOAP Generation - Model Details")
    st.markdown(
        """
        | Model | Base Architecture | Fine-tuning | Training Data | Role |
        |-------|-------------------|-------------|---------------|------|
        | **PRAXIS-SOAP-v1** | Med42-8B (Llama 3) | QLoRA 4-bit, Unsloth | PriMock57 + MTSamples + ChatDoctor + Synthetic GP | Primary model |
        | **Med42-8B** | Llama 3 (79.1% MedQA) | QLoRA 4-bit, Unsloth | Same combined dataset | Comparison |
        | **Meditron3-8B** | Llama 3.1 (48B+ medical tokens) | QLoRA 4-bit, Unsloth | Same combined dataset | Comparison |
        | **Mistral-7B** | Mistral v0.2 | None (zero-shot) | N/A | Baseline |
        """
    )

    st.markdown("---")
    st.subheader("Training Dataset Composition")
    st.markdown(
        """
        | Source | Type | Examples | Description |
        |--------|------|----------|-------------|
        | PriMock57 | Real GP transcripts | ~57 | British English mock GP consultations (Babylon Health) |
        | MTSamples | Medical transcriptions | ~10 | Curated GP-relevant medical transcription samples |
        | ChatDoctor | Medical Q&A | ~450 | HuggingFace GP-filtered medical dialogue |
        | Synthetic GP | Hand-crafted | 5 | Expert-written UK GP SOAP note examples |
        | **Total** | | **~520** | Instruction-tuned format (transcript → SOAP + SNOMED CT) |
        """
    )

    st.markdown("---")
    st.subheader("Technology Stack")
    st.markdown(
        """
        | Component | Technology |
        |-----------|------------|
        | Speech-to-Text | OpenAI Whisper, Google MedASR, HuggingFace Transformers |
        | Fine-tuning (STT) | LoRA via PEFT, PyTorch |
        | Fine-tuning (LLM) | QLoRA 4-bit via Unsloth, TRL SFTTrainer |
        | Inference | Ollama (GGUF quantised models, fully offline) |
        | Entity Extraction | Regex-based medical NER |
        | API | FastAPI |
        | Demo | Streamlit |
        | Experiment Tracking | Weights & Biases, MLflow |
        """
    )


# =====================================================================
# PAGE 7: Training Logs
# =====================================================================
elif page == "Training Logs":
    st.title("STT Fine-tuning - Training Progress")

    if not TRAINING_LOG_PATH.exists():
        st.warning("Training log not found at `data/training_v3_full.log`")
    else:
        log_text = TRAINING_LOG_PATH.read_text(errors="replace")

        progress_matches = re.findall(
            r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]", log_text
        )

        if progress_matches:
            last = progress_matches[-1]
            pct = int(last[0])
            current_step = int(last[1])
            total_steps = int(last[2])
            timing = last[3]

            st.metric("Current Step", f"{current_step} / {total_steps}")
            st.progress(pct / 100)
            st.markdown(f"**Progress:** {pct}% | **Timing:** {timing}")
        else:
            st.info("No progress data found in log yet.")

        st.markdown("---")
        st.subheader("Recent Log Output")
        lines = log_text.strip().split("\n")
        recent = "\n".join(lines[-50:])
        st.code(recent, language=None)


# =====================================================================
# PAGE: Clinical Evaluation
# =====================================================================
elif page == "Clinical Evaluation":
    st.title("Clinical Quality Evaluation")
    st.markdown(
        "Automated scoring of SOAP note quality across all available models. "
        "Each note is evaluated on structure, clinical content, coding, and safety."
    )

    from src.soap_generation.generator import SOAPGenerator, SOAPModel, display_name
    from src.soap_generation.templates import NoteTemplate, TEMPLATE_DISPLAY_NAMES, TEMPLATE_DESCRIPTIONS
    from src.clinical.evaluator import ClinicalEvaluator

    evaluator = ClinicalEvaluator()

    col_s, col_t = st.columns([1, 1])
    with col_s:
        sample = st.selectbox(
            "Select clinical scenario:",
            list(SAMPLE_TRANSCRIPTS.keys()),
            key="eval_sample",
        )
    with col_t:
        eval_template_options = {TEMPLATE_DISPLAY_NAMES[t.value]: t for t in NoteTemplate}
        eval_template_name = st.selectbox(
            "Note template:",
            list(eval_template_options.keys()),
            key="eval_template",
        )
        eval_template = eval_template_options[eval_template_name]
        st.caption(TEMPLATE_DESCRIPTIONS[eval_template.value])

    transcript = SAMPLE_TRANSCRIPTS[sample]
    st.text_area("Input Transcript:", value=transcript, height=150, disabled=True)

    if st.button("Evaluate All Models", type="primary"):
        gen = SOAPGenerator(template=eval_template)
        with st.spinner("Generating SOAP notes from all models and scoring..."):
            results = gen.generate_comparison(transcript, template=eval_template)

        if not results:
            st.error("No models available. Ensure Ollama is running.")
        else:
            scores = []
            for r in results:
                score = evaluator.evaluate(
                    r.soap_note,
                    model=display_name(r.model),
                    time_secs=r.generation_time_secs,
                    tokens=r.token_count,
                    template=eval_template.value,
                )
                scores.append(score)

            # Summary table
            st.subheader("Quality Scores")
            eval_df = pd.DataFrame([
                {
                    "Model": s.model_used,
                    "Structure": f"{s.structure_score:.0%}",
                    "Content": f"{s.content_score:.0%}",
                    "Overall": f"{s.overall_score:.0%}",
                    "SNOMED Codes": s.snomed_code_count,
                    "Safety-netting": "Yes" if s.has_safety_netting else "No",
                    "UK English": "Yes" if s.uses_uk_english else "No",
                    "Latency (s)": s.generation_time_secs,
                    "Tokens": s.token_count,
                }
                for s in scores
            ])
            st.dataframe(eval_df, use_container_width=True, hide_index=True)

            # Bar chart
            st.subheader("Overall Quality Score Comparison")
            chart_df = pd.DataFrame([
                {"Model": s.model_used, "Overall Score": round(s.overall_score, 2)}
                for s in scores
            ]).set_index("Model")
            st.bar_chart(chart_df)

            # Show full SOAP outputs
            st.markdown("---")
            st.subheader("Generated Notes")
            for r, s in zip(results, scores):
                with st.expander(f"{display_name(r.model)} - Overall: {s.overall_score:.0%}"):
                    st.markdown(r.soap_note)

            st.markdown("---")
            st.subheader("Evaluation Criteria")
            st.markdown(
                """
                | Criterion | Weight | Description |
                |-----------|--------|-------------|
                | **Structure** | 30% | Presence of all four SOAP sections (S, O, A, P) |
                | **Clinical Content** | 50% | Chief complaint, history, medications, examination, diagnosis, treatment, safety-netting, follow-up |
                | **SNOMED Coding** | 10% | Inclusion of SNOMED CT concept codes |
                | **Safety** | 10% | Absence of hallucinated content (penalty if flagged) |

                Additional checks: UK English terminology, negation handling, safety-netting advice.
                """
            )


# =====================================================================
# PAGE: GP Review / Edit Workflow
# =====================================================================
elif page == "GP Review":
    st.title("GP Review & Approval Workflow")
    st.markdown(
        "Review, edit, and approve/reject AI-generated clinical notes. "
        "This implements the human-in-the-loop workflow required for clinical safety. "
        "Scores are recorded using the GP blind-scoring criteria (7 Likert scales)."
    )

    from src.clinical.session_store import SessionStore
    from src.clinical.gp_scoring import ScoringCriteria, GPScoringSession

    store = SessionStore(store_dir=PROJECT_ROOT / "data" / "sessions")
    sessions = store.list_sessions(limit=50)

    # Filter to sessions with SOAP notes
    reviewable = [s for s in sessions if s["has_soap"]]

    if not reviewable:
        st.info("No sessions with SOAP notes available for review. Run a pipeline from the Live Demo page first.")
    else:
        # GP identification
        gp_name = st.text_input("Reviewer name (e.g. Dr Smith):", key="gp_reviewer_name")

        st.markdown(f"**{len(reviewable)} sessions available for review**")

        for s in reviewable:
            created = s["created_at"][:19].replace("T", " ")
            review_label = "Reviewed" if s["reviewed"] else "Pending review"
            with st.expander(f"{created} - {s['entity_count']} entities - {review_label}", expanded=False):
                full_data = store.load_full(s["session_id"])
                if not full_data:
                    st.error("Failed to load session.")
                    continue

                soap = full_data.get("soap_note", {})
                transcript = full_data.get("raw_transcript", "")

                tab_note, tab_score, tab_edit = st.tabs(["SOAP Note", "Score & Approve", "Edit Note"])

                with tab_note:
                    st.subheader("Original Transcript")
                    st.code(transcript[:1000] if transcript else "N/A", language=None)
                    st.subheader("Generated SOAP Note")
                    st.markdown(soap.get("raw_text", "No SOAP note.") if soap else "No SOAP note.")
                    if soap:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Model", soap.get("model_used", "N/A"))
                        col2.metric("Latency", f"{soap.get('generation_time_secs', 0)}s")
                        col3.metric("SNOMED Codes", "Yes" if soap.get("has_snomed_codes") else "No")

                with tab_score:
                    if not gp_name:
                        st.warning("Please enter your name above to score notes.")
                    else:
                        sid = s["session_id"]
                        st.markdown("**Rate each criterion (1 = Poor, 5 = Excellent):**")

                        c_acc = st.slider("Clinical accuracy", 1, 5, 3, key=f"acc_{sid}")
                        c_comp = st.slider("Completeness", 1, 5, 3, key=f"comp_{sid}")
                        c_struct = st.slider("Structure quality", 1, 5, 3, key=f"struct_{sid}")
                        c_lang = st.slider("Language appropriateness", 1, 5, 3, key=f"lang_{sid}")
                        c_safe = st.slider("Safety-netting", 1, 5, 3, key=f"safe_{sid}")
                        c_code = st.slider("Coding quality (SNOMED)", 1, 5, 3, key=f"code_{sid}")
                        c_overall = st.slider("Overall acceptability", 1, 5, 3, key=f"overall_{sid}")

                        would_edit = st.checkbox("Would you need to edit this before signing?", key=f"edit_{sid}")
                        comments = st.text_area("Comments:", key=f"comments_{sid}")
                        critical = st.text_input("Critical errors (comma-separated):", key=f"critical_{sid}")

                        col_approve, col_reject = st.columns(2)
                        with col_approve:
                            if st.button("Approve", key=f"approve_{sid}", type="primary"):
                                criteria = ScoringCriteria(
                                    clinical_accuracy=c_acc, completeness=c_comp,
                                    structure_quality=c_struct, language_appropriateness=c_lang,
                                    safety_netting=c_safe, coding_quality=c_code,
                                    overall_acceptability=c_overall, comments=comments,
                                    would_edit=would_edit,
                                    critical_errors=[e.strip() for e in critical.split(",") if e.strip()],
                                )
                                gp_session = GPScoringSession(gp_id=gp_name, output_dir=PROJECT_ROOT / "data" / "gp_scores")
                                gp_session.score_note(
                                    session_id=sid,
                                    soap_text=soap.get("raw_text", ""),
                                    model_id=soap.get("model_used", ""),
                                    transcript=transcript,
                                    scores=criteria,
                                )
                                gp_session.export_scores()

                                # Update session via ontology review_soap
                                session_obj = store.load(sid)
                                if session_obj:
                                    from src.clinical.ontology import SOAPNote as SOAPNoteObj
                                    session_obj.soap_note = SOAPNoteObj(
                                        raw_text=soap.get("raw_text", ""),
                                        model_used=soap.get("model_used", ""),
                                    )
                                    session_obj.review_soap(gp_name, accepted=True)
                                    store.save(session_obj)

                                st.success(f"Note approved by {gp_name}. Scores saved.")

                        with col_reject:
                            if st.button("Reject", key=f"reject_{sid}"):
                                criteria = ScoringCriteria(
                                    clinical_accuracy=c_acc, completeness=c_comp,
                                    structure_quality=c_struct, language_appropriateness=c_lang,
                                    safety_netting=c_safe, coding_quality=c_code,
                                    overall_acceptability=c_overall, comments=comments,
                                    would_edit=would_edit,
                                    critical_errors=[e.strip() for e in critical.split(",") if e.strip()],
                                )
                                gp_session = GPScoringSession(gp_id=gp_name, output_dir=PROJECT_ROOT / "data" / "gp_scores")
                                gp_session.score_note(
                                    session_id=sid,
                                    soap_text=soap.get("raw_text", ""),
                                    model_id=soap.get("model_used", ""),
                                    transcript=transcript,
                                    scores=criteria,
                                )
                                gp_session.export_scores()

                                session_obj = store.load(sid)
                                if session_obj:
                                    from src.clinical.ontology import SOAPNote as SOAPNoteObj
                                    edits = [e.strip() for e in critical.split(",") if e.strip()]
                                    session_obj.soap_note = SOAPNoteObj(
                                        raw_text=soap.get("raw_text", ""),
                                        model_used=soap.get("model_used", ""),
                                    )
                                    session_obj.review_soap(gp_name, accepted=False, edits=edits)
                                    store.save(session_obj)

                                st.warning(f"Note rejected by {gp_name}. Scores saved.")

                with tab_edit:
                    st.markdown("Edit the SOAP note before approving:")
                    edited_text = st.text_area(
                        "Editable SOAP note:",
                        value=soap.get("raw_text", "") if soap else "",
                        height=300,
                        key=f"edited_{s['session_id']}",
                    )
                    if st.button("Save Edits", key=f"save_edit_{s['session_id']}"):
                        # Update the session file with edited text
                        if full_data and full_data.get("soap_note"):
                            full_data["soap_note"]["raw_text"] = edited_text
                            filepath = store.store_dir / f"{s['session_id']}.json"
                            import json as _json
                            filepath.write_text(_json.dumps(full_data, indent=2))
                            st.success("SOAP note updated. You can now score and approve the edited version.")


# =====================================================================
# PAGE: Patient History
# =====================================================================
elif page == "Patient History":
    st.title("Patient Visit History")
    st.markdown(
        "View all consultations for a specific patient. "
        "See how many times they have visited, review past SOAP notes, and track their clinical journey."
    )

    from src.clinical.patient_store import PatientStore
    from src.clinical.session_store import SessionStore

    patient_store = PatientStore(store_dir=PROJECT_ROOT / "data" / "patients")
    session_store = SessionStore(store_dir=PROJECT_ROOT / "data" / "sessions")

    all_patients = patient_store.list_patients()

    if not all_patients:
        st.info("No patients registered yet. Go to **Patient Management** to register a patient.")
    else:
        # Patient selector
        patient_options = {f"{p.full_name} (NHS: {p.nhs_number or 'N/A'})": p for p in all_patients}
        selected_label = st.selectbox("Select Patient:", list(patient_options.keys()), key="history_patient")
        selected_patient = patient_options[selected_label]

        st.markdown("---")

        # Patient info card
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Name", selected_patient.full_name)
        col2.metric("NHS Number", selected_patient.nhs_number or "N/A")
        col3.metric("Date of Birth", selected_patient.date_of_birth or "N/A")

        # Get all sessions for this patient
        patient_sessions = session_store.get_patient_sessions(selected_patient.patient_id)
        col4.metric("Total Visits", len(patient_sessions))

        st.markdown("---")

        if not patient_sessions:
            st.info(
                f"No consultations recorded for **{selected_patient.full_name}** yet. "
                "Go to **Live Demo** to record a consultation for this patient."
            )
        else:
            st.subheader(f"Consultation History ({len(patient_sessions)} visits)")

            for i, s in enumerate(patient_sessions, 1):
                created = s["created_at"][:19].replace("T", " ")
                status_parts = []
                if s["has_soap"]:
                    status_parts.append("SOAP generated")
                if s["reviewed"]:
                    status_parts.append("Reviewed")
                status = " | ".join(status_parts) if status_parts else "In progress"

                with st.expander(f"Visit {i} - {created} - {s['entity_count']} entities - {status}", expanded=(i == 1)):
                    full_data = session_store.load_full(s["session_id"])
                    if full_data:
                        tab_t, tab_e, tab_s, tab_a = st.tabs(["Transcript", "Entities", "SOAP Note", "Audit Log"])

                        with tab_t:
                            st.code(full_data.get("raw_transcript", "N/A"), language=None)
                            if full_data.get("diarised_transcript"):
                                st.subheader("Diarised")
                                st.code(full_data["diarised_transcript"], language=None)

                        with tab_e:
                            entities = full_data.get("entities", [])
                            if entities:
                                edf = pd.DataFrame(entities)
                                display_cols = [c for c in ["type", "text", "confidence", "negated", "snomed_code"] if c in edf.columns]
                                st.dataframe(edf[display_cols], use_container_width=True, hide_index=True)
                            else:
                                st.info("No entities extracted.")

                        with tab_s:
                            soap = full_data.get("soap_note")
                            if soap:
                                st.markdown(soap.get("raw_text", ""))
                                sc1, sc2, sc3 = st.columns(3)
                                sc1.metric("Model", soap.get("model_used", ""))
                                sc2.metric("Latency", f"{soap.get('generation_time_secs', 0)}s")
                                sc3.metric("Reviewed", "Yes" if soap.get("reviewed") else "No")
                            else:
                                st.info("No SOAP note generated.")

                        with tab_a:
                            audit = full_data.get("audit_log", [])
                            if audit:
                                for entry in audit:
                                    ts = entry.get("timestamp", "")[:19].replace("T", " ")
                                    st.markdown(
                                        f"**{ts}** - `{entry.get('event', '')}` - "
                                        f"{entry.get('details', '')}"
                                    )
                            else:
                                st.info("No audit events.")

                        st.caption(f"Session ID: {s['session_id']}")


# =====================================================================
# PAGE: Session History
# =====================================================================
elif page == "Session History":
    st.title("Session History")
    st.markdown("Browse past consultation sessions with full audit trails.")

    try:
        from src.clinical.session_store import SessionStore

        store = SessionStore(store_dir=PROJECT_ROOT / "data" / "sessions")
        sessions = store.list_sessions(limit=50)

        if not sessions:
            st.info("No sessions recorded yet. Run a pipeline from the Live Demo page to create one.")
        else:
            st.markdown(f"**{len(sessions)} sessions recorded**")

            # Export button
            col_exp1, col_exp2 = st.columns([1, 4])
            with col_exp1:
                if st.button("Export All"):
                    export_path = store.export_all(PROJECT_ROOT / "data" / "sessions_export.jsonl")
                    st.success(f"Exported to {export_path.name}")

            st.markdown("---")

            for s in sessions:
                created = s["created_at"][:19].replace("T", " ")
                patient_label = f" - {s['patient_name']}" if s.get("patient_name") else ""
                status_parts = []
                if s["has_soap"]:
                    status_parts.append("SOAP generated")
                if s["reviewed"]:
                    status_parts.append("Reviewed")
                status = " | ".join(status_parts) if status_parts else "In progress"

                with st.expander(
                    f"{created}{patient_label} - {s['entity_count']} entities - {status}",
                    expanded=False,
                ):
                    full_data = store.load_full(s["session_id"])
                    if full_data:
                        tab_t, tab_e, tab_s, tab_a = st.tabs(["Transcript", "Entities", "SOAP Note", "Audit Log"])

                        with tab_t:
                            st.code(full_data.get("raw_transcript", "N/A"), language=None)
                            if full_data.get("diarised_transcript"):
                                st.subheader("Diarised")
                                st.code(full_data["diarised_transcript"], language=None)

                        with tab_e:
                            entities = full_data.get("entities", [])
                            if entities:
                                edf = pd.DataFrame(entities)
                                display_cols = [c for c in ["type", "text", "confidence", "negated", "snomed_code"] if c in edf.columns]
                                st.dataframe(edf[display_cols], use_container_width=True, hide_index=True)
                            else:
                                st.info("No entities extracted.")

                        with tab_s:
                            soap = full_data.get("soap_note")
                            if soap:
                                st.markdown(soap.get("raw_text", ""))
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Model", soap.get("model_used", ""))
                                col2.metric("Latency", f"{soap.get('generation_time_secs', 0)}s")
                                col3.metric("Reviewed", "Yes" if soap.get("reviewed") else "No")
                            else:
                                st.info("No SOAP note generated.")

                        with tab_a:
                            audit = full_data.get("audit_log", [])
                            if audit:
                                for entry in audit:
                                    ts = entry.get("timestamp", "")[:19].replace("T", " ")
                                    st.markdown(
                                        f"**{ts}** - `{entry.get('event', '')}` - "
                                        f"{entry.get('details', '')}"
                                    )
                            else:
                                st.info("No audit events.")

                        st.caption(f"Session ID: {s['session_id']}")

    except Exception as e:
        st.error(f"Error loading sessions: {e}")
