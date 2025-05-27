const fakeList = [
    {voiceURI:'Microsoft David Desktop - English (United States)', name:'Microsoft David Desktop - English (United States)', lang:'en-US', localService:true, default:true},
    {voiceURI:'Microsoft Zira Desktop - English (United States)', name:'Microsoft Zira Desktop - English (United States)', lang:'en-US', localService:true, default:false},
    {voiceURI:'Microsoft Mark Desktop - English (United States)', name:'Microsoft Mark Desktop - English (United States)', lang:'en-US', localService:true, default:false}
  ];
  window.speechSynthesis.getVoices = () => fakeList;  