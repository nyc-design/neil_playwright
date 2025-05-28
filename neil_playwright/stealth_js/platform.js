  // 1) main world
  Object.defineProperty(navigator, 'platform', { get: () => 'Win32', configurable: true });

  // helper to inject inside workers
  const injectPlatform = () => {
    Object.defineProperty(navigator, 'platform', { get: () => 'Win32', configurable: true });
  };

  // 2) wrap Worker
  const NativeWorker = Worker;
  window.Worker = function(scriptURL, options) {
    // build a blob that first spoofs platform, then loads the real script
    const blob = new Blob([
      '(' + injectPlatform.toString() + ')();',
      'importScripts("' + scriptURL + '");'
    ], { type: 'application/javascript' });
    return new NativeWorker(URL.createObjectURL(blob), options);
  };

  // 3) wrap SharedWorker
  if (window.SharedWorker) {
    const NativeShared = SharedWorker;
    window.SharedWorker = function(scriptURL, name) {
      const blob = new Blob([
        '(' + injectPlatform.toString() + ')();',
        'importScripts("' + scriptURL + '");'
      ], { type: 'application/javascript' });
      return new NativeShared(URL.createObjectURL(blob), name);
    };
  }