from flask import Flask, jsonify, render_template, request

from inference import generate_text, load_model


app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    try:
        _, tokenizer, device, config = load_model()
        return jsonify(
            {
                "ok": True,
                "device": device,
                "vocab_size": tokenizer.vocab_size,
                "block_size": config["block_size"],
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/generate")
def generate():
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"error": "Please enter a prompt."}), 400

    try:
        result = generate_text(
            prompt=prompt,
            max_tokens=payload.get("max_tokens", 750),
            temperature=payload.get("temperature", 0.8),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Generation failed: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
