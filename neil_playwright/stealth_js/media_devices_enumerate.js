navigator.mediaDevices.enumerateDevices = async () => [
    { deviceId: 'default', kind: 'audioinput',  label: 'Microphone (Realtek(R) Audio)' },
    { deviceId: 'default', kind: 'audiooutput', label: 'Speakers (Realtek(R) Audio)' }
  ];  