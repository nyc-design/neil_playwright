Object.defineProperty(navigator, 'connection', {
    get: () => ({ downlink: 3.6, effectiveType: '4g', rtt: 200, saveData: false })
  });