context.addInitScript(() => {
    const OriginalAudio = window.AudioContext || window.webkitAudioContext;
    window.AudioContext = function() {
      const ctx = new OriginalAudio();
      // stub createAnalyser and any other methods LinkedIn probes
      ctx.createAnalyser = () => ({ /* return dummy frequency data matching your Windows run */ });
      return ctx;
    };
  });