// in stealth_js/user_agent_data.js
if (navigator.userAgentData) {
    // override the low-entropy property
    Object.defineProperty(navigator.userAgentData, 'platform', {
      get: () => 'Windows',
      configurable: true
    });
  
    // override the high-entropy API
    const orig = navigator.userAgentData.getHighEntropyValues;
    navigator.userAgentData.getHighEntropyValues = (hints) =>
      orig.call(navigator.userAgentData, hints)
          .then(data => ({ ...data, platform: 'Windows' }));
  }