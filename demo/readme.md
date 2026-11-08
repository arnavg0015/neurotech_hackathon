# Evoke demo

`demo.py` serves `demo.html` and runs the saved models. An activation starts the appliance selector. A double blink toggles the selected appliance, highlights it for one second, and resumes activation detection. Both recordings continue from where they stopped.

The appliances are Light, Fan, TV, Speaker, Heater, and Kettle. Fan, Speaker, and Kettle start on. A yellow box means on; a neutral box means off. Appliance states stay as selected until another blink changes them or you press Restart.

The two central model blocks highlight the current stage. The recorded word appears below Activation Model; the eyes below Double Blink Model blink twice at each recorded double-blink cue. The animation represents the instruction, not measured eyelid timing. Only model predictions change the stage or an appliance.

From the project root:

```bash
conda activate neurotech
python demo/demo.py
```

Open http://127.0.0.1:8000. The page fits the browser window for screen recording. Use **Play**, **Pause**, and **Restart**. Pause freezes playback and the selection highlight. Restart resets both recordings and the appliances.

[replay](replay/readme.md) contains 60 seconds of activation EEG, joined from six Trial 5 clips. Activate cues appear near 8, 18, 28, 38, 48, and 58 seconds; predictions follow one second later. Blink selection and its one-second highlight add to the total viewing time. Playback stops when either recording ends.

This is a curated demo using known successful predictions, not a new accuracy test. The activation model comes from [hackathon_pipeline](../results/hackathon_pipeline/readme.md); the two-channel blink model and its full Trial 3 recording come from [blink_pipeline](../results/blink_pipeline/readme.md).

To rebuild the activation clips from the saved outputs:

```bash
python demo/prepare_replay.py
```

