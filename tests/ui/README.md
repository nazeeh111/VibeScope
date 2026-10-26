# Browser smoke check

This optional check drives the actual local instrument. It does not mock the analysis endpoint or supply fabricated results.

Start VibeScope, then in a second terminal:

```sh
cd tests/ui
npm ci
npx playwright install chromium
npm test
```

Set `VIBESCOPE_URL` if using another loopback port. Set `CHROME_EXECUTABLE` to use an already installed Chromium-based browser. The check loads the synthetic demo, runs measurement, checks native form validity, plays and seeks the synchronized videos, fetches the three export artifacts, checks mobile overflow, and verifies invalid-band feedback. It can take several minutes on a slow CPU.

When the repository benchmark reference is present, the check also imports it, verifies the acquisition-confirmation gate, runs actual reference screening, and checks that changing the axis clears the confirmation. Set `VIBESCOPE_BASELINE_JSON` to test another compatible synthetic reference. A browser check is functional evidence, not validation of a real machine.
