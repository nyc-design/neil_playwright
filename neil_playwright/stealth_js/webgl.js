(() => {
    // WebGL vendor/renderer spoof to match Windows values
    const originalGet = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(parameter) {
      // 37445 = UNMASKED_VENDOR_WEBGL
      if (parameter === 37445) {
        return 'Google Inc. (AMD)';
      }
      // 37446 = UNMASKED_RENDERER_WEBGL
      if (parameter === 37446) {
        return 'ANGLE (AMD, AMD Radeon(TM) Graphics (0x00001638) Direct3D11 vs_5_0 ps_5_0, D3D11-31.0.12027.9001)';
      }
      return originalGet.call(this, parameter);
    };
  
    // Apply same spoof in Workers
    const spoofFunc = () => {
      const origGet = WebGLRenderingContext.prototype.getParameter;
      WebGLRenderingContext.prototype.getParameter = function(param) {
        if (param === 37445) return 'Google Inc. (AMD)';
        if (param === 37446) return 'ANGLE (AMD, AMD Radeon(TM) Graphics (0x00001638) Direct3D11 vs_5_0 ps_5_0, D3D11-31.0.12027.9001)';
        return origGet.call(this, param);
      };
    };
  
    const workerBlob = new Blob([
      '(' + spoofFunc.toString() + ')();'
    ], { type: 'application/javascript' });
  
    const OriginalWorker = Worker;
    window.Worker = function(url, options) {
      // Inject spoof code before executing worker script
      const blobUrl = URL.createObjectURL(workerBlob);
      return new OriginalWorker(blobUrl, options);
    };
  })();