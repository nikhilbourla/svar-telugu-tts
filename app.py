import spaces  # Import before torch/Gradio so ZeroGPU can initialize its CUDA handling.
import gradio as gr
from huggingface_hub import snapshot_download

from svar_inference import Svar, SynthSettings


MODEL_REPO_ID = "copycat1/svar-telugu-tts"
ENCODEC_REPO_ID = "facebook/encodec_24khz"

print(f"Downloading Svar inference files from {MODEL_REPO_ID}...")

MODEL_DIR = snapshot_download(
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    allow_patterns=[
        "dit.safetensors",
        "dit_config.json",
        "latent_stats.safetensors",
        "t2s.safetensors",
        "t2s_config.json",
        "telugu_char.model",
    ],
)

print("Svar files are ready.")
print("Loading Svar for ZeroGPU...")

# ZeroGPU requires CUDA model placement at module scope.
tts = Svar.from_pretrained(
    model_dir=MODEL_DIR,
    device="cuda",
    encodec_path=ENCODEC_REPO_ID,
)

print("Svar loaded successfully!")


@spaces.GPU(duration=180)
def generate_audio(text: str, speaker: str):
    if not text or not text.strip():
        raise gr.Error("Please enter Telugu text.")
    if speaker not in ("male", "female"):
        raise gr.Error("Please select male or female voice.")

    try:
        settings = SynthSettings(
            seed=42,
            dit_steps=65,
            dit_fp16=True,
            verbose=False,
        )
        audio, sample_rate = tts.synthesize(
            text=text.strip(),
            speaker=speaker,
            settings=settings,
        )
        return (sample_rate, audio)
    except gr.Error:
        raise
    except Exception as exc:
        print(f"Svar generation error: {exc}")
        raise gr.Error(f"Audio generation failed: {exc}") from exc


with gr.Blocks(title="Svar - Telugu Text-to-Speech") as demo:
    gr.Markdown(
        """
        # Svar - Telugu Text-to-Speech

        Generate Telugu speech using the Svar voice model.

        Enter Telugu text, select a voice, and generate audio.
        """
    )
    text_input = gr.Textbox(
        label="Enter Telugu Text",
        placeholder="ఇక్కడ మీ తెలుగు వాక్యాన్ని నమోదు చేయండి...",
        lines=5,
    )
    speaker_input = gr.Radio(
        choices=["male", "female"],
        value="male",
        label="Select Voice",
    )
    generate_button = gr.Button("Generate Speech", variant="primary")
    audio_output = gr.Audio(label="Generated Speech", type="numpy")

    generate_button.click(
        fn=generate_audio,
        inputs=[text_input, speaker_input],
        outputs=audio_output,
    )


if __name__ == "__main__":
    demo.queue().launch()
