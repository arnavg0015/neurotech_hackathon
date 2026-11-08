# Demo

`demo.py` serves `demo.html` and runs the saved models. It replays spoken Trial 5 until activation is detected, then switches to double-blink Trial 3. A double-blink prediction lights the selected bulb and stops the replay.

From the project root:

```bash
conda activate neurotech
python demo/demo.py
```

Open http://127.0.0.1:8000. Use **Play**, **Pause**, and **Restart**. 

