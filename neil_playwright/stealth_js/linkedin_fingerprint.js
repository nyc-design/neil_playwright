(() => {
    const subtle = window.crypto.subtle;
    // 1) Intercept generateKey to return a constant key object
    const originalGenerateKey = subtle.generateKey.bind(subtle);
    subtle.generateKey = async function(algorithm, extractable, keyUsages) {
      // Create one “master” key once and always return it
      if (!window._fixedFingerprintKey) {
        window._fixedFingerprintKey = await originalGenerateKey(algorithm, extractable, keyUsages);
      }
      return window._fixedFingerprintKey;
    };
  
    // 2) Intercept wrapKey to return a dummy wrapped key ArrayBuffer
    const originalWrapKey = subtle.wrapKey.bind(subtle);
    subtle.wrapKey = async function(format, key, wrappingKey, wrapAlgorithm) {
      // Return a zeroed ArrayBuffer of a plausible length
      const length = 256; // or whatever your Windows run returns
      return new Uint8Array(length).buffer;
    };
  
    // 3) Leave encrypt/decrypt alone so other code works
  })();