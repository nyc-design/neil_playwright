const fakeList = [
    {voiceURI:'Microsoft David - English (United States)', name:'Microsoft David - English (United States)', lang:'en-US', localService:true, default:true},
    {voiceURI:'Microsoft Zira - English (United States)', name:'Microsoft Zira - English (United States)', lang:'en-US', localService:true, default:false},
    {voiceURI:'Microsoft Mark - English (United States)', name:'Microsoft Mark - English (United States)', lang:'en-US', localService:true, default:false}
  ];
  window.speechSynthesis.getVoices = () => fakeList;  