const origSpeak = window.speechSynthesis.speak;
window.speechSynthesis.speak = (utterance) => {
  if (utterance && typeof utterance.dispatchEvent === 'function') {
    setTimeout(() => utterance.dispatchEvent(new Event('start')), 10);
    setTimeout(() => utterance.dispatchEvent(new Event('end')), 1100);
  }
};