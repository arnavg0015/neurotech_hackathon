"""Run with: conda activate neurotech && python demo/demo.py"""

import errno
import gzip
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cloudpickle
import mne
import numpy as np
import pandas as pd


DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
STAGES = {
    "activation": ("hackathon_pipeline", "activate_classifier.pkl", 5, [21, 500], 15.6),
    "blink": ("blink_pipeline", "blink_classifier.pkl", 3, [2, 100], 0.0),
}


class Replay:
    def __init__(self, stage="activation"):
        folder, model_file, trial, shape, clock_offset = STAGES[stage]
        results = ROOT / "results" / folder
        self.stage = stage
        with (results / model_file).open("rb") as file:
            artifact = cloudpickle.load(file)
        self.model = artifact["pipeline"]
        settings = artifact["preprocessing"]
        report = json.loads((results / "metrics.json").read_text())
        manifest = pd.read_csv(results / "epoch_split.csv")
        source = next(row for row in report["sources"] if row["trial"] == trial)
        log = pd.read_csv(ROOT / source["stimulus_file"], keep_default_na=False)
        raw = mne.io.read_raw_fif(
            results / f"trial_{trial}_preprocessed_raw.fif", preload=True, verbose="error"
        )
        selected = manifest["trial"].eq(trial).to_numpy()
        rows = manifest.loc[selected].reset_index(drop=True)
        with np.load(results / "epochs.npz") as data:
            if not np.array_equal(data["trial"], manifest["trial"]):
                raise ValueError("Epochs and split manifest do not match. Rerun the notebook.")
            if not np.array_equal(data["split"], manifest["split"]):
                raise ValueError("Epoch split assignments do not match the manifest.")
            if data["channels"].tolist() != artifact["channels"] or float(data["sfreq"]) != shape[1]:
                raise ValueError("Saved epoch channels or sample rate do not match the model.")
            samples = data["X"][selected]
        if samples.shape != (len(rows), *shape) or not np.isfinite(samples).all():
            raise ValueError(f"Expected finite {stage} epochs with shape {shape}.")
        if artifact["input_units"] != "microvolts" or artifact["input_shape"] != shape:
            raise ValueError("The saved model has a different input format.")
        display_channels = artifact.get("display_channels", artifact["channels"])
        if raw.ch_names != display_channels or len(display_channels) != 21 or raw.info["sfreq"] != shape[1]:
            raise ValueError("Continuous EEG channels or sample rate do not match the model.")
        if settings["clock_offset_s"] != clock_offset or not rows["split"].eq("Test").all():
            raise ValueError(f"Expected the agreed {stage} clock correction and test split.")
        if settings["sample_rate_hz"] != shape[1] or settings["epoch_seconds"] != 1:
            raise ValueError("Expected one-second epochs at the saved sample rate.")
        if not rows["event_id"].is_unique or not log["seq"].is_unique:
            raise ValueError("Event IDs must be unique within the recording.")

        origin_ms = float(log["onset_epoch_ms"].iloc[0])
        eeg_start_ms = pd.Timestamp(raw.info["meas_date"]).timestamp() * 1000
        origin_in_eeg = (origin_ms - eeg_start_ms) / 1000 - settings["clock_offset_s"]
        duration = (float(log["offset_epoch_ms"].iloc[-1]) - origin_ms) / 1000
        sfreq = int(raw.info["sfreq"])
        self.inputs = {int(row.event_id): samples[i:i + 1] for i, row in enumerate(rows.itertuples())}
        epochs = rows.set_index("event_id")
        self.events = {}
        for row in log.itertuples():
            event_id = int(row.seq)
            start = (row.onset_epoch_ms - origin_ms) / 1000
            end = (row.offset_epoch_ms - origin_ms) / 1000
            ready_at = input_start = None
            if event_id in self.inputs:
                epoch = epochs.loc[event_id]
                input_start = float(epoch.start_s - origin_in_eeg)
                ready_at = float(epoch.end_s - origin_in_eeg)
                if not (start - 1e-6 <= input_start < ready_at <= end + 1e-6):
                    raise ValueError(f"Epoch {event_id} crosses its stimulus interval.")
                if not np.isclose(epoch.onset_in_eeg_s - origin_in_eeg, start, atol=1e-6):
                    raise ValueError(f"Timestamp correction does not match event {event_id}.")
                if not np.isclose(ready_at - input_start, 1) or epoch.start_sample < round(0.2 * sfreq):
                    raise ValueError(f"Epoch {event_id} lacks a full second or its preceding baseline.")
            self.events[event_id] = {
                "id": event_id, "word": str(row.word) if row.word else "Blank",
                "kind": "blank" if not row.stimulus_type else "word",
                "start": start, "end": end, "input_start": input_start, "ready_at": ready_at,
            }
        if not set(self.inputs).issubset(self.events):
            raise ValueError("Some epochs have no matching stimulus event.")

        start_sample = max(0, int(np.ceil(origin_in_eeg * sfreq)))
        stop_sample = min(raw.n_times, int(np.ceil((origin_in_eeg + duration) * sfreq)))
        values = raw.get_data(start=start_sample, stop=stop_sample) * 1e6
        if not values.size or not np.isfinite(values).all():
            raise ValueError("The replay signal is missing or contains invalid samples.")
        bucket = round(0.02 * sfreq)
        boundaries = np.arange(0, values.shape[1], bucket)
        signal = {
            "start": start_sample / sfreq - origin_in_eeg, "step": bucket / sfreq,
            "min": np.minimum.reduceat(values, boundaries, axis=1).round(3).tolist(),
            "max": np.maximum.reduceat(values, boundaries, axis=1).round(3).tolist(),
            "scales": np.maximum(10, np.ceil(1.5 * np.percentile(np.abs(values), 99, axis=1) / 10) * 10).tolist(),
        }
        session = {
            "stage": stage, "trial": trial, "duration": duration, "sample_rate": sfreq,
            "channels": raw.ch_names, "events": list(self.events.values()),
            "eligible_count": len(self.inputs), "signal": signal,
        }
        self.session = json.dumps(session, separators=(",", ":"), allow_nan=False).encode()
        self.compressed_session = gzip.compress(self.session)

    def predict(self, event_id):
        if event_id not in self.inputs:
            raise KeyError("This segment has no eligible one-second epoch.")
        predicted = int(self.model.predict(self.inputs[event_id])[0])
        return {"stage": self.stage, "event_id": event_id, "predicted": predicted,
                "ready_at": self.events[event_id]["ready_at"]}


class Handler(BaseHTTPRequestHandler):
    def reply(self, body, content_type="application/json", status=200, compressed=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def error(self, message, status):
        self.reply(json.dumps({"error": message}).encode(), status=status)

    def do_GET(self):
        url = urlsplit(self.path)
        path = url.path
        if path in ("/", "/demo.html"):
            self.reply((DEMO_DIR / "demo.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/session":
            compressed = "gzip" in self.headers.get("Accept-Encoding", "")
            stage = parse_qs(url.query).get("stage", ["activation"])[0]
            if stage not in self.server.replays:
                return self.error("stage must be activation or blink.", 400)
            replay = self.server.replays[stage]
            self.reply(replay.compressed_session if compressed else replay.session, compressed=compressed)
        elif path == "/favicon.ico":
            self.reply(b"", status=204)
        else:
            self.error("Not found.", 404)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/predict":
            return self.error("Not found.", 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024:
                raise ValueError("Send a JSON object containing an event_id.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or type(body.get("event_id")) is not int:
                raise ValueError("event_id must be an integer.")
            stage = body.get("stage", "activation")
            if not isinstance(stage, str) or stage not in self.server.replays:
                raise ValueError("stage must be activation or blink.")
            result = self.server.replays[stage].predict(body["event_id"])
            self.reply(json.dumps(result).encode())
        except KeyError as error:
            self.error(str(error.args[0]), 404)
        except (ValueError, UnicodeDecodeError) as error:
            self.error(str(error), 400)
        except Exception as error:
            print(f"Inference failed: {error}", flush=True)
            self.error("Model inference failed. Check the Python terminal and restart playback.", 500)

    def log_message(self, format, *args):
        if len(args) > 1 and str(args[1]) not in ("200", "204"):
            super().log_message(format, *args)


if __name__ == "__main__":
    try:
        print("Loading the activation and double-blink recordings and models...", flush=True)
        replays = {stage: Replay(stage) for stage in STAGES}
        with ThreadingHTTPServer(("127.0.0.1", 8000), Handler) as server:
            server.replays = replays
            print("Open http://127.0.0.1:8000 — Ctrl+C to stop.", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nReplay server stopped.")
    except FileNotFoundError as error:
        raise SystemExit(f"Missing demo data: {error}\nSee demo/readme.md for the required saved outputs. The current blink training notebook builds a separate analysis model.")
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            raise SystemExit(
                "Port 8000 is already in use. If the demo is already running, open "
                "http://127.0.0.1:8000 or stop that server with Ctrl+C before starting again."
            )
        raise SystemExit(f"Could not start the demo: {error}")
    except Exception as error:
        raise SystemExit(f"Could not start the demo: {error}")
