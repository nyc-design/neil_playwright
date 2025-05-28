// battery.js
Object.defineProperty(navigator, 'getBattery', {
    value: () => Promise.resolve({
      charging: true,
      chargingTime: 0,
      dischargingTime: Infinity,
      level: 1
    })
  });  