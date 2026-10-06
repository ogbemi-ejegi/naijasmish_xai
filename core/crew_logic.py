import os
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from database.chroma_engine import query_local_scams

load_dotenv()

def run_inspection(preprocessed_text: str):
    # 1. Retrieve Context from local RAG
    historical_context = query_local_scams(preprocessed_text, top_k=3)
    context_str = "\n".join([f"- {msg}" for msg in historical_context]) if historical_context else "No historical matches found."

    # 2. Define Agents
    linguistic_agent = Agent(
        role="Nigerian Cyber-Linguistics Analyst",
        goal="Parse SMS syntax to isolate Nigerian slang, Pidgin (Naija) patterns, code-switching, and cultural emotional coercion models.",
        backstory="You are a specialist in West African digital linguistics. Your function is to analyze texts for structural code-switching, use of indigenous urgency markers (e.g., 'sharp sharp', 'abeg', 'fast-fast'), and artificial social pressure designed to bypass standard English filters.",
        llm="openrouter/google/gemini-2.5-flash",
        allow_delegation=False,
        verbose=False
    )

    pattern_agent = Agent(
        role="Localized Cyber Threat Pattern Analyst",
        goal="Cross-reference metadata, claims, and formatting layout against historical Nigerian banking and security fraud typologies.",
        backstory="You are a technical cyber threat intelligence officer. You map the structural elements of incoming communications. You look for spoofed entity headers, mismatching sender shortcodes, masking techniques, and layouts matching common local fraud types such as BVN restrictions, fake credit alerts, and federal palliative scams.",
        llm="openrouter/google/gemini-2.5-flash",
        allow_delegation=False,
        verbose=False
    )

    scoring_agent = Agent(
        role="Cybersecurity Meta-Judge",
        goal="Evaluate the discrete payloads of the worker agents, calculate a final threat probability score, resolve conflicting analysis, and write the human-readable explanation layer.",
        backstory="You are the final arbiter in a security operations center multi-agent system. You do not look at raw un-vetted text data. You ingest the structured analysis from your Linguistic and Pattern analysts. Your job is to balance their perspectives, heavily weight structural indicators against linguistic formatting to prevent clean prose from sneaking past, and draft the definitive Explainable AI text summary. If the message matches historical fraud schemas, requests sensitive actions (like clicking a link or calling an unknown number), or creates artificial urgency, you MUST assign a threat probability > 75%.",
        llm="openrouter/google/gemini-2.5-flash",
        allow_delegation=False,
        verbose=False
    )

    # 3. Define Tasks
    task_linguistic = Task(
        description=f"Inspect the following preprocessed SMS text and output a structured JSON payload identifying the linguistic profile, presence of localized pressure variables, and a qualitative assessment of the language used.\n\nText: {preprocessed_text}",
        expected_output="A structured JSON payload identifying the linguistic profile, presence of localized pressure variables, and a qualitative assessment.",
        agent=linguistic_agent
    )

    task_pattern = Task(
        description=f"Inspect structural placeholders and entity claims in the SMS text. Compare it against these historically similar scams from our database:\n{context_str}\n\nSMS Text: {preprocessed_text}\n\nOutput a structured JSON identifying layout compliance, historical fraud schema matches, and structural abnormalities.",
        expected_output="A structured JSON identifying layout compliance, historical fraud schema matches, and structural abnormalities.",
        agent=pattern_agent
    )

    task_scoring = Task(
        description="Review the JSON payloads provided by the Linguistic Agent and the Pattern Agent. Return a consolidated output containing a 'final_verdict' (Boolean/Binary), a 'threat_probability' (0-100%), and a clean Markdown 'explanation_summary' that details why the message was categorized that way based on the specific evidence discovered. Note that the output should explicitly include the threat probability integer value somewhere so it can be parsed.",
        expected_output="A consolidated output containing final_verdict (boolean), threat_probability (0-100 integer), and an explanation_summary (Markdown text).",
        agent=scoring_agent
    )

    # 4. Form Crew and Execute
    crew = Crew(
        agents=[linguistic_agent, pattern_agent, scoring_agent],
        tasks=[task_linguistic, task_pattern, task_scoring],
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()
    
    # We should return the final result and the outputs of the intermediate tasks for the UI traces
    
    # Get raw output strings from tasks (handles different crewai versions)
    ling_trace = getattr(task_linguistic.output, 'raw', str(task_linguistic.output)) if hasattr(task_linguistic, 'output') else "No output"
    pat_trace = getattr(task_pattern.output, 'raw', str(task_pattern.output)) if hasattr(task_pattern, 'output') else "No output"
    
    return {
        "final_result": str(result),
        "linguistic_trace": ling_trace,
        "pattern_trace": pat_trace
    }
