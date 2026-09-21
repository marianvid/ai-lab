import { app } from '../../scripts/app.js';

app.registerExtension({
  name: 'ai-lab.instance-preset',
  async setup() {
    const params = new URLSearchParams(window.location.search);
    if (params.get('ai_lab_preset') !== '1') return;
    const response = await fetch('/ai-lab/preset');
    if (!response.ok) {
      console.error('AI-Lab preset unavailable:', response.status);
      return;
    }
    try {
      const preset = await response.json();
      if (Array.isArray(preset.nodes)) {
        await app.loadGraphData(preset, true, true);
      } else {
        await app.loadApiJson(preset, 'AI-Lab preset.json');
      }
    } catch (error) {
      console.error('AI-Lab could not open its ComfyUI preset:', error);
    }
  },
});
