"""
Pipeline de transcrição com diarização.

Fluxo:
1. Converte áudio para WAV 16kHz mono (ffmpeg)
2. Separa voz do ruído com Demucs (isola o stem vocal)
3. Diarização com pyannote (identifica quem fala quando)
4. Transcrição com Whisper API da OpenAI
5. Junta tudo: transcrição final com timestamps e speaker labels

Uso:
    python pipeline.py audio_original/reuniao.mp3 --num-speakers 3
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import OpenAI
from pyannote.audio import Pipeline as DiarizationPipeline
from pydub import AudioSegment

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not OPENAI_API_KEY:
    sys.exit("ERRO: defina OPENAI_API_KEY no arquivo .env")
if not HF_TOKEN:
    sys.exit("ERRO: defina HF_TOKEN no arquivo .env")


# ---------- utilidades ----------

def log(msg: str):
    print(f"\n>>> {msg}", flush=True)


def format_timestamp(seconds: float) -> str:
    """Converte segundos em HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


# ---------- etapa 1: conversão ----------

def convert_to_wav(input_path: Path, output_path: Path):
    """Converte qualquer áudio para WAV 16kHz mono usando ffmpeg."""
    log(f"[1/4] Convertendo {input_path.name} para WAV 16kHz mono...")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ac", "1",            # mono
        "-ar", "16000",        # 16kHz
        "-c:a", "pcm_s16le",   # PCM 16-bit
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    log(f"    OK: {output_path}")


# ---------- etapa 2: denoise com Demucs ----------

def denoise_with_demucs(input_wav: Path, output_wav: Path, work_dir: Path):
    """Roda Demucs pra isolar o stem vocal do áudio."""
    log("[2/4] Separando voz do ruído com Demucs (usa GPU se disponível)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"    Device: {device}")

    demucs_out = work_dir / "demucs_tmp"
    demucs_out.mkdir(exist_ok=True)

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "-n", "htdemucs",
        "-d", device,
        "-o", str(demucs_out),
        str(input_wav),
    ]
    subprocess.run(cmd, check=True)

    # demucs salva em: <out>/htdemucs/<nome_sem_ext>/vocals.wav
    vocals_path = demucs_out / "htdemucs" / input_wav.stem / "vocals.wav"
    if not vocals_path.exists():
        raise FileNotFoundError(f"Demucs não gerou vocals.wav esperado em {vocals_path}")

    # Demucs gera em 44.1kHz stereo — reconverte pra 16kHz mono pro pyannote/whisper
    log("    Reconvertendo stem vocal para 16kHz mono...")
    cmd = [
        "ffmpeg", "-y", "-i", str(vocals_path),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(output_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # limpa os arquivos temporários do demucs
    shutil.rmtree(demucs_out)
    log(f"    OK: {output_wav}")


# ---------- etapa 3: diarização ----------

def diarize(input_wav: Path, output_rttm: Path, num_speakers: int | None = None):
    """Identifica quem fala quando usando pyannote."""
    log("[3/4] Rodando diarização (pyannote)...")

    pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        log("    Diarização rodando na GPU")

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
        log(f"    Forçando num_speakers={num_speakers}")

    diarization = pipeline(str(input_wav), **kwargs)

    # salva arquivo RTTM padrão
    with open(output_rttm, "w") as f:
        diarization.write_rttm(f)

    # retorna lista de segmentos: (start, end, speaker)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    log(f"    OK: {len(segments)} segmentos de fala, {output_rttm}")
    return segments


# ---------- etapa 4: transcrição ----------

def transcribe_with_whisper(input_wav: Path) -> dict:
    """Transcreve o áudio completo com Whisper API."""
    log("[4/4] Transcrevendo com Whisper API da OpenAI...")

    # Whisper API tem limite de 25MB por request.
    # Como convertemos pra 16kHz mono, 1h ≈ 115MB — precisamos chunkar.
    audio = AudioSegment.from_wav(input_wav)
    duration_ms = len(audio)
    chunk_ms = 10 * 60 * 1000  # 10 minutos por chunk (seguro, ~19MB)

    client = OpenAI(api_key=OPENAI_API_KEY)
    all_segments = []

    tmp_dir = input_wav.parent / "whisper_chunks"
    tmp_dir.mkdir(exist_ok=True)

    try:
        for i, start_ms in enumerate(range(0, duration_ms, chunk_ms)):
            end_ms = min(start_ms + chunk_ms, duration_ms)
            chunk = audio[start_ms:end_ms]
            chunk_path = tmp_dir / f"chunk_{i:03d}.wav"
            chunk.export(chunk_path, format="wav")

            log(f"    Chunk {i+1}: {start_ms/1000:.0f}s - {end_ms/1000:.0f}s")

            with open(chunk_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="pt",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            # ajusta timestamps somando o offset do chunk
            offset = start_ms / 1000
            for seg in result.segments:
                all_segments.append({
                    "start": seg.start + offset,
                    "end": seg.end + offset,
                    "text": seg.text.strip(),
                })
    finally:
        shutil.rmtree(tmp_dir)

    log(f"    OK: {len(all_segments)} segmentos transcritos")
    return {"segments": all_segments}


# ---------- etapa 5: merge diarização + transcrição ----------

def merge_diarization_and_transcription(
    diarization_segments: list,
    transcription: dict,
) -> list:
    """
    Para cada segmento transcrito pelo Whisper, encontra o falante
    que mais se sobrepõe temporalmente com ele na diarização.
    """
    log("Juntando diarização e transcrição...")

    merged = []
    for trans_seg in transcription["segments"]:
        t_start = trans_seg["start"]
        t_end = trans_seg["end"]

        # calcula overlap com cada segmento de diarização
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for diar_seg in diarization_segments:
            overlap_start = max(t_start, diar_seg["start"])
            overlap_end = min(t_end, diar_seg["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg["speaker"]

        merged.append({
            "start": t_start,
            "end": t_end,
            "speaker": best_speaker,
            "text": trans_seg["text"],
        })

    return merged


def write_final_transcription(segments: list, output_path: Path):
    """Escreve a transcrição final em texto legível."""
    lines = []
    current_speaker = None

    for seg in segments:
        timestamp = format_timestamp(seg["start"])
        if seg["speaker"] != current_speaker:
            lines.append(f"\n[{timestamp}] {seg['speaker']}:")
            current_speaker = seg["speaker"]
        lines.append(f"  {seg['text']}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"    OK: transcrição final em {output_path}")


# ---------- orquestração ----------

def main():
    parser = argparse.ArgumentParser(description="Pipeline de transcrição com diarização")
    parser.add_argument("audio", type=str, help="Caminho do arquivo de áudio de entrada")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Número de falantes (se souber). Ex: 3")
    parser.add_argument("--skip-denoise", action="store_true",
                        help="Pula a etapa de Demucs (usa o WAV original)")
    parser.add_argument("--resume", action="store_true",
                        help="Pula etapas cujos arquivos de saída já existem")
    args = parser.parse_args()

    input_path = Path(args.audio).resolve()
    if not input_path.exists():
        sys.exit(f"Arquivo não encontrado: {input_path}")

    # cria pasta de saída específica pro áudio
    project_root = Path(__file__).parent
    output_dir = project_root / "output" / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Pasta de saída: {output_dir}")

    # etapa 1
    converted_wav = output_dir / "01_converted.wav"
    if args.resume and converted_wav.exists():
        log(f"[1/4] Pulando conversão — {converted_wav.name} já existe (--resume)")
    else:
        convert_to_wav(input_path, converted_wav)

    # etapa 2
    if args.skip_denoise:
        audio_for_processing = converted_wav
        log("[2/4] Pulando denoise (flag --skip-denoise)")
    else:
        denoised_wav = output_dir / "02_denoised.wav"
        if args.resume and denoised_wav.exists():
            log(f"[2/4] Pulando denoise — {denoised_wav.name} já existe (--resume)")
        else:
            denoise_with_demucs(converted_wav, denoised_wav, output_dir)
        audio_for_processing = denoised_wav

    # etapa 3
    rttm_path = output_dir / "03_diarization.rttm"
    diar_segments = diarize(audio_for_processing, rttm_path, num_speakers=args.num_speakers)

    # etapa 4
    transcription = transcribe_with_whisper(audio_for_processing)
    (output_dir / "04_transcription.json").write_text(
        json.dumps(transcription, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # merge
    merged = merge_diarization_and_transcription(diar_segments, transcription)
    (output_dir / "05_merged.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    final_txt = output_dir / "06_transcricao_final.txt"
    write_final_transcription(merged, final_txt)

    # move o áudio original pra pasta de já transcritos
    archive_dir = project_root / "already_transcripted"
    archive_dir.mkdir(exist_ok=True)
    archived_path = archive_dir / input_path.name

    if archived_path.exists():
        # se já existe um arquivo com o mesmo nome, adiciona um sufixo
        stem = input_path.stem
        suffix = input_path.suffix
        counter = 2
        while archived_path.exists():
            archived_path = archive_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    shutil.move(str(input_path), str(archived_path))
    log(f"Áudio original movido para: {archived_path}")

    log("=" * 60)
    log(f"CONCLUÍDO! Transcrição final: {final_txt}")
    log("=" * 60)


if __name__ == "__main__":
    main()