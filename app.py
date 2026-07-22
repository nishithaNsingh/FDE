import os
import json
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

load_dotenv()  

app = Flask(__name__)

OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY")

MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are a triage engine for Rapido's customer support queue.
Rapido is a bike-taxi / auto / cab platform in India. You read one incoming
complaint and return ONLY a JSON object, no markdown fences, no preamble,
no explanation before or after, with exactly these keys:

{
  "category": one of ["Fraud/Overcharge", "Safety", "Lost Item", "Driver Behavior", "App/Technical", "Refund/Billing", "Other"],
  "escalation_risk": integer 1-10 (10 = likely to go viral / legal / safety incident if ignored for days),
  "risk_reason": one short sentence explaining the score,
  "recommended_queue": one of ["Priority - respond within 1 hour", "Standard - respond within 24 hours", "Low - batch review"],
  "draft_response": a short, empathetic first-response message to send the customer immediately (2-3 sentences, in the voice of Rapido support, acknowledging the issue and stating next steps)
}

Score escalation_risk high when: physical safety was threatened, fraud/theft occurred,
the customer explicitly threatens to post publicly or already has, or money was lost
with no resolution path. Score it low for routine app glitches or minor delays.

Return raw JSON only. Nothing else."""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json(force=True)
    complaint = (data.get("complaint") or "").strip()

    if not complaint:
        return jsonify({"error": "No complaint text provided."}), 400

    if not OPENROUTER_KEY:
        return jsonify({"error": "Server is missing OPENROUTER_KEY. Set it in your hosting provider's environment variables."}), 500

   
    candidates = [MODEL, "z-ai/glm-4.5-air:free", "openai/gpt-oss-20b:free"]
    candidates = list(dict.fromkeys(candidates))  # dedupe, keep order

    last_error = None
    payload = None

    for model_id in candidates:
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": complaint},
                    ],
                    "temperature": 0.3,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                payload = resp.json()
                break
            last_error = f"{model_id} -> ({resp.status_code}) {resp.text[:200]}"
        except requests.exceptions.RequestException as e:
            last_error = f"{model_id} -> network error: {str(e)}"

    if payload is None:
        return jsonify({"error": f"All models unavailable. Last error: {last_error}"}), 502

    try:
        raw_text = payload["choices"][0]["message"]["content"].strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].strip()

   
        if not raw_text.startswith("{"):
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1:
                raw_text = raw_text[start:end + 1]

        result = json.loads(raw_text)
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "Could not parse model response. Free models are occasionally inconsistent with JSON — try again."}), 502
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Network error calling OpenRouter: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)